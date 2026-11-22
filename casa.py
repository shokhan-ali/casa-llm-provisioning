"""
CASA: Cost-Aware Split-Adaptive conformal provisioning for
prefill/decode (P/D) disaggregated LLM serving.

All provisioning policies share one interface: at the start of window t they
receive the point forecasts and scale estimates for both pools and return the
capacity bounds U_P, U_D (tokens/minute). After the window they observe the
realised demand and update their internal state.
"""
import numpy as np

POOLS = ("P", "D")


N_MAX = 128  # cluster limit per pool (4x the training peak requirement)


def n_instances(U, mu, n_min=1, n_max=N_MAX):
    """Instances needed so that n*mu >= U, clipped to [n_min, n_max]
    (an infinite bound, i.e. ACI level <= 0, provisions the cluster limit)."""
    U = np.where(np.isfinite(U), np.maximum(U, 0.0), np.where(U > 0, np.inf, 0.0))
    n = np.ceil(np.minimum(U / mu, n_max))
    return np.clip(n, n_min, n_max).astype(int)


# ---------------------------------------------------------------- baselines
class PointForecast:
    name = "Point"
    def bounds(self, yhat, sig):  return yhat.copy()
    def update(self, y, yhat, sig, cap_miss): pass


class Headroom:
    def __init__(self, h=0.25): self.h = h; self.name = f"Headroom{int(h*100)}"
    def bounds(self, yhat, sig):  return yhat * (1 + self.h)
    def update(self, y, yhat, sig, cap_miss): pass


class SplitCPBonferroni:
    """Static split conformal, alpha/2 per pool, calibrated once offline."""
    name = "SplitCP-Bonf"
    def __init__(self, alpha, z_cal):
        n = len(z_cal)
        k = int(np.ceil((n + 1) * (1 - alpha / 2)))
        self.q = np.array([np.sort(z_cal[:, j])[min(k, n) - 1] for j in range(2)])
    def bounds(self, yhat, sig): return yhat + self.q * sig
    def update(self, y, yhat, sig, cap_miss): pass


def _quantile(scores, level):
    """Conformal (1-level) quantile with ACI boundary conventions."""
    if level <= 0:  return np.inf
    if level >= 1:  return -np.inf
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - level)))
    if k > n: return np.inf
    return np.sort(scores)[k - 1]


class ACIBonferroni:
    """Per-pool Adaptive Conformal Inference (Gibbs & Candes 2021), alpha/2 each."""
    name = "ACI-Bonf"
    def __init__(self, alpha, z_init, gamma=0.005, window=288):
        self.a_target = alpha / 2
        self.a = np.full(2, alpha / 2)
        self.g, self.W = gamma, window
        self.buf = [list(z_init[-window:, j]) for j in range(2)]
    def bounds(self, yhat, sig):
        q = np.array([_quantile(np.array(self.buf[j]), self.a[j]) for j in range(2)])
        self._q = q
        return yhat + q * sig
    def update(self, y, yhat, sig, cap_miss):
        z = (y - yhat) / sig
        for j in range(2):
            err = float(cap_miss[j])
            self.a[j] += self.g * (self.a_target - err)
            self.buf[j].append(z[j]); self.buf[j] = self.buf[j][-self.W:]


class JointACI:
    """
    Joint ACI on a split-scaled max score.  With a fixed split phi this is the
    'JointACI-sym' ablation (phi = 0).  CASA subclasses it and adapts phi.

    score_t(phi) = max( z_P * e^{-phi},  z_D * e^{+phi} )
    U_P = yhat_P + theta * e^{+phi} * sig_P ;  U_D = yhat_D + theta * e^{-phi} * sig_D
    A window is covered iff theta >= score_t(phi).
    """
    name = "JointACI-sym"
    def __init__(self, alpha, z_init, gamma=0.005, window=288, phi=0.0):
        self.alpha, self.a, self.g, self.W = alpha, alpha, gamma, window
        self.Z = [tuple(r) for r in z_init[-window:]]
        self.phi = phi
        self.theta_hist, self.phi_hist, self.a_hist = [], [], []
    @staticmethod
    def _scores(Z, phi):
        Z = np.asarray(Z)
        return np.maximum(Z[:, 0] * np.exp(-phi), Z[:, 1] * np.exp(phi))
    def choose_phi(self, yhat, sig):
        return self.phi
    def bounds(self, yhat, sig):
        self.phi = self.choose_phi(yhat, sig)
        th = _quantile(self._scores(self.Z, self.phi), self.a)
        self.theta = th
        self.theta_hist.append(th); self.phi_hist.append(self.phi); self.a_hist.append(self.a)
        s = np.array([np.exp(self.phi), np.exp(-self.phi)])
        return yhat + th * s * sig
    def update(self, y, yhat, sig, cap_miss):
        err = float(cap_miss[0] or cap_miss[1])  # joint SLO violation
        self.a += self.g * (self.alpha - err)
        self.Z.append(tuple((y - yhat) / sig)); self.Z = self.Z[-self.W:]


class CASA(JointACI):
    """
    Proposed.  Validity loop: joint ACI on the level a_t (any split is allowed).
    Efficiency loop: every step pick the split phi that minimises the
    expected cost-weighted capacity on the recent score window, at the
    current level a_t:
        phi* = argmin_phi  sum_k c_k/mu_k * ( yhat_k + Q_{1-a_t}(score(phi)) * s_k(phi) * sig_k )
    The split is smoothed with step eta to avoid oscillation.
    """
    name = "CASA"
    def __init__(self, alpha, z_init, cost, mu, gamma=0.005, window=288,
                 phi_grid=np.linspace(-1.5, 1.5, 61), eta=0.5):
        super().__init__(alpha, z_init, gamma, window, 0.0)
        self.c_over_mu = np.asarray(cost) / np.asarray(mu)
        self.grid, self.eta = phi_grid, eta
    def choose_phi(self, yhat, sig):
        a = min(max(self.a, 1e-3), 0.999)  # use a valid level to *rank* splits
        best, best_cost = self.phi, np.inf
        for phi in self.grid:
            th = _quantile(self._scores(self.Z, phi), a)
            s = np.array([np.exp(phi), np.exp(-phi)])
            c = np.sum(self.c_over_mu * (yhat + th * s * sig))
            if c < best_cost: best, best_cost = phi, c
        return (1 - self.eta) * self.phi + self.eta * best


class HeadroomCalibrated:
    """Fixed multiplicative headroom tuned on the calibration day so that the
    empirical joint violation there equals alpha (strong, practical baseline)."""
    name = "Headroom-cal"
    def __init__(self, alpha, Ycal, Yhcal, mu):
        best = None
        for h in np.arange(0.0, 3.0001, 0.01):
            cap = n_instances((Yhcal * (1 + h)), mu) * mu
            v = np.mean(np.any(Ycal > cap, axis=1))
            if v <= alpha: best = h; break
        self.h = best if best is not None else 3.0
    def bounds(self, yhat, sig): return yhat * (1 + self.h)
    def update(self, y, yhat, sig, cap_miss): pass
