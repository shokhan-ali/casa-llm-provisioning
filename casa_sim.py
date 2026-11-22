"""
End-to-end experiments on the Azure LLM inference traces (2024 code / conversation,
2025 multimodal).  Produces results/*.json, results/*.csv and figures/*.png.
"""
import json, time, sys, itertools
from collections import deque
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from casa import (PointForecast, Headroom, SplitCPBonferroni, ACIBonferroni,
                  JointACI, CASA, HeadroomCalibrated, n_instances)

import os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = HERE  # the per-minute CSV files sit next to the code
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
WIN = 5            # provisioning window (minutes)
DAY = 24 * 60 // WIN
TRAIN_DAYS, CAL_DAYS = 2, 1
N_INST_PEAK = 32   # per-instance capacity chosen so the training peak needs 32 instances
SCALE_W = 36       # windows used for the residual scale estimate (3 h)

TRACES = {"Code": "minute_code.csv", "Conversation": "minute_conv.csv",
          "Multimodal": "minute_mm.csv"}


# ------------------------------------------------------------------ data
def load_windows(fname):
    d = pd.read_csv(f"{DATA}/{fname}", index_col=0, parse_dates=True)
    g = d.resample(f"{WIN}min")
    out = pd.DataFrame({"P": g["ctx"].max(), "D": g["gen"].max(),      # peak minute
                        "Pm": g["ctx"].mean(), "Dm": g["gen"].mean()})  # mean minute
    return out


def features(w, k, h):
    """Features for predicting w[k] at window t using information up to t-h."""
    y, m = np.log1p(w[k].values), np.log1p(w[k + "m"].values)
    o = np.log1p(w["D" if k == "P" else "P"].values)  # other pool
    T = len(y)
    cols = {}
    def lag(a, L):
        r = np.full(T, np.nan); r[L:] = a[:-L]; return r
    for L in range(h, h + 6): cols[f"y{L}"] = lag(y, L)
    for L in (h, h + 1):      cols[f"m{L}"] = lag(m, L); cols[f"o{L}"] = lag(o, L)
    cols["y_h12"] = lag(y, h + 11)
    cols["y_day"] = lag(y, DAY) if DAY >= h else np.nan
    tod = (np.arange(T) % DAY) / DAY
    cols["sin"], cols["cos"] = np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)
    X = pd.DataFrame(cols).values
    return X, y


def forecast(w, h, model="hgb"):
    """Return point forecasts for both pools (tokens/min), trained on TRAIN_DAYS only."""
    tr = slice(h + 6, TRAIN_DAYS * DAY)
    preds = {}
    for k in ("P", "D"):
        X, y = features(w, k, h)
        if model == "hgb":
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                              max_leaf_nodes=31, random_state=0)
            m.fit(X[tr], y[tr]); p = m.predict(np.nan_to_num(X, nan=np.nan))
        elif model == "naive":      # last observed peak (persistence)
            p = X[:, 0]
        p = np.expm1(p); p[: h + 6] = np.nan
        preds[k] = p
    return np.column_stack([preds["P"], preds["D"]])


def scales(Y, Yh, h):
    """Causal residual scale: mean |residual| over the last SCALE_W *observed* windows."""
    R = np.abs(Y - Yh)
    S = np.full_like(Y, np.nan)
    for t in range(len(Y)):
        lo, hi = t - h - SCALE_W + 1, t - h + 1
        if lo >= 0: S[t] = np.nanmean(R[lo:hi], axis=0)
    return np.maximum(S, 0.01 * np.abs(Yh))


# ------------------------------------------------------------------ simulation
def simulate(policy, Y, Yh, S, mu, cost, t0, h):
    T = len(Y); rec = []
    pending = deque()
    for t in range(t0, T):
        while pending and pending[0][0] <= t - h:      # feedback with delay h
            _, args = pending.popleft(); policy.update(*args)
        t_start = time.perf_counter()
        U = policy.bounds(Yh[t], S[t])
        dt = time.perf_counter() - t_start
        n = n_instances(U, mu)
        cap = n * mu
        miss = Y[t] > cap
        pending.append((t, (Y[t], Yh[t], S[t], miss)))
        rec.append(dict(t=t, nP=n[0], nD=n[1], missP=miss[0], missD=miss[1],
                        miss=miss[0] | miss[1], cost=float(np.dot(cost, n)),
                        shortP=max(0, Y[t, 0] - cap[0]) / mu[0],
                        shortD=max(0, Y[t, 1] - cap[1]) / mu[1], dt=dt))
    return pd.DataFrame(rec)


def oracle(Y, mu, cost, t0):
    n = n_instances(Y[t0:], mu)
    return float((n @ cost).sum())


def longest_run(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0; best = max(best, cur)
    return best


def summarise(df, orc):
    hours = WIN / 60
    daily = df.assign(day=(df.t // DAY)).groupby("day")["miss"].mean()
    return dict(viol=float(df.miss.mean()), violP=float(df.missP.mean()),
                violD=float(df.missD.mean()),
                worst_day_viol=float(daily.max()),
                longest_burst_min=int(longest_run(df.miss.values) * WIN),
                cost_inst_h=float(df.cost.sum() * hours),
                overprov_pct=float(100 * (df.cost.sum() / orc - 1)),
                short_inst=float((df.shortP + df.shortD)[df.miss].mean()) if df.miss.any() else 0.0,
                us_per_decision=float(1e6 * df.dt.mean()))


def make_policies(alpha, Zcal, cost, mu, gamma=0.005, Ycal=None, Yhcal=None):
    return [PointForecast(), Headroom(0.25), Headroom(0.5),
            HeadroomCalibrated(alpha, Ycal, Yhcal, mu), SplitCPBonferroni(alpha, Zcal),
            ACIBonferroni(alpha, Zcal, gamma), JointACI(alpha, Zcal, gamma),
            CASA(alpha, Zcal, cost, mu, gamma)]


def run_trace(name, alpha=0.05, h=1, cost=(1.0, 1.0), model="hgb", gamma=0.005,
              keep_series=False, policies=None):
    w = load_windows(TRACES[name])
    Y = w[["P", "D"]].values.astype(float)
    Yh = forecast(w, h, model)
    S = scales(Y, Yh, h)
    mu = Y[: TRAIN_DAYS * DAY].max(axis=0) / N_INST_PEAK
    cal = slice(TRAIN_DAYS * DAY, (TRAIN_DAYS + CAL_DAYS) * DAY)
    Zcal = ((Y - Yh) / S)[cal]
    t0 = (TRAIN_DAYS + CAL_DAYS) * DAY
    cost = np.asarray(cost)
    orc = oracle(Y, mu, cost, t0)
    out, series = {}, {}
    pols = policies(alpha, Zcal, cost, mu, gamma) if policies else make_policies(alpha, Zcal, cost, mu, gamma, Y[cal], Yh[cal])
    for p in pols:
        df = simulate(p, Y, Yh, S, mu, cost, t0, h)
        out[p.name] = summarise(df, orc)
        if keep_series:
            series[p.name] = df
            if hasattr(p, "phi_hist"):
                series[p.name + "_phi"] = np.array(p.phi_hist)
                series[p.name + "_a"] = np.array(p.a_hist)
    out["_oracle_inst_h"] = orc * WIN / 60
    mape = {k: float(np.nanmean(np.abs(Yh[t0:, i] - Y[t0:, i]) / Y[t0:, i]) * 100)
            for i, k in enumerate("PD")}
    out["_forecast_mape"] = mape
    out["_mu"] = mu.tolist()
    if keep_series:
        return out, series, (w, Y, Yh, S, mu, t0)
    return out




if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the main comparison of the paper on one trace.")
    ap.add_argument("--trace", default="Code", choices=list(TRACES))
    ap.add_argument("--alpha", type=float, default=0.05, help="target joint violation rate")
    ap.add_argument("--lead", type=int, default=1, help="provisioning lead time h (5-minute windows)")
    ap.add_argument("--gamma", type=float, default=0.005, help="ACI step size")
    ap.add_argument("--cost", type=float, nargs=2, default=[1.0, 1.0], metavar=("C_P", "C_D"))
    ap.add_argument("--model", default="hgb", choices=["hgb", "naive"])
    a = ap.parse_args()
    out = run_trace(a.trace, alpha=a.alpha, h=a.lead, cost=tuple(a.cost), model=a.model, gamma=a.gamma)
    meta = {k: out.pop(k) for k in list(out) if k.startswith("_")}
    df = pd.DataFrame(out).T
    tag = f"{a.trace.lower()}_a{a.alpha}_h{a.lead}_c{a.cost[0]}-{a.cost[1]}_{a.model}"
    df.to_csv(f"{RES}/results_{tag}.csv")
    pd.set_option("display.width", 200)
    print(df[["viol", "violP", "violD", "worst_day_viol", "longest_burst_min",
              "cost_inst_h", "overprov_pct"]].round(4).to_string())
    print(f"\noracle cost: {meta['_oracle_inst_h']:.1f} instance-hours | "
          f"forecast MAPE: {meta['_forecast_mape']}\nsaved: results/results_{tag}.csv")
