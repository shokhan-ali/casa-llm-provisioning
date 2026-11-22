"""Figures and motivating statistics for the paper."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from casa_sim import run_trace, load_windows, TRACES, DAY, RES
import os
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Serif", "axes.grid": True,
                     "grid.alpha": .3, "figure.dpi": 200})
C = {"Point": "#999999", "Headroom-cal": "#b07aa1", "SplitCP-Bonf": "#edc948",
     "ACI-Bonf": "#59a14f", "JointACI-sym": "#4e79a7", "CASA": "#e15759"}
stats = {}

# Fig 2: demand series (whole week, normalised)
fig, ax = plt.subplots(3, 1, figsize=(7, 5.2), sharex=False)
for i, t in enumerate(TRACES):
    w = load_windows(TRACES[t]); x = np.arange(len(w)) / DAY
    ax[i].plot(x, w.P / w.P.max(), lw=.6, label="Prefill demand (peak input tokens/min)")
    ax[i].plot(x, w.D / w.D.max(), lw=.6, label="Decode demand (peak output tokens/min)", alpha=.8)
    ax[i].axvspan(0, 2, color="k", alpha=.06); ax[i].axvspan(2, 3, color="orange", alpha=.08)
    ax[i].set_title(f"{t} trace", fontsize=9); ax[i].set_ylabel("normalised")
    rho = spearmanr(w.P, w.D).correlation; stats[f"{t}_level_spearman"] = float(rho)
ax[0].legend(loc="upper right", fontsize=7); ax[2].set_xlabel("day (grey: training, orange: calibration, white: test)")
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_traces.png"); plt.close()

# Run once with full series for the remaining figures
series_all = {}
for t in TRACES:
    o, s, (w, Y, Yh, S, mu, t0) = run_trace(t, keep_series=True)
    series_all[t] = (o, s, Y, Yh, S, t0)
    Z = ((Y - Yh) / S)[t0:]
    stats[f"{t}_resid_spearman"] = float(spearmanr(Z[:, 0], Z[:, 1]).correlation)
    stats[f"{t}_resid_pearson"] = float(pearsonr(Z[:, 0], Z[:, 1])[0])
    # joint exceedance check: P(both large) vs independence
    qP, qD = np.quantile(Z[:, 0], .95), np.quantile(Z[:, 1], .95)
    stats[f"{t}_joint_tail"] = float(np.mean((Z[:, 0] > qP) & (Z[:, 1] > qD)))

# Fig 3: residual scatter
fig, ax = plt.subplots(1, 3, figsize=(7, 2.4))
for i, t in enumerate(TRACES):
    o, s, Y, Yh, S, t0 = series_all[t]; Z = ((Y - Yh) / S)[t0:]
    ax[i].scatter(Z[:, 0], Z[:, 1], s=2, alpha=.35)
    ax[i].set_title(f"{t}\nSpearman ρ = {stats[t+'_resid_spearman']:.2f}", fontsize=8)
    ax[i].set_xlabel("normalised prefill residual"); 
ax[0].set_ylabel("normalised decode residual")
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_residuals.png"); plt.close()

# Fig 4: rolling 1-day joint violation rate
fig, ax = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)
for i, t in enumerate(TRACES):
    o, s, *_ = series_all[t]
    for m in ["Headroom-cal", "SplitCP-Bonf", "ACI-Bonf", "JointACI-sym", "CASA"]:
        r = s[m].miss.rolling(DAY, min_periods=DAY // 4).mean()
        ax[i].plot(np.arange(len(r)) / DAY, r, lw=1, color=C[m], label=m)
    ax[i].axhline(.05, color="k", ls="--", lw=.8); ax[i].set_title(t, fontsize=9)
    ax[i].set_xlabel("test day"); ax[i].set_ylim(0, .25)
ax[0].set_ylabel("rolling 24 h joint\nviolation rate"); ax[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_rolling.png"); plt.close()

# Fig 5: cost vs violation across alpha
R = json.load(open(f"{RES}/all_results.json"))
fig, ax = plt.subplots(1, 3, figsize=(7, 2.6))
for i, t in enumerate(TRACES):
    for m in ["Headroom-cal", "SplitCP-Bonf", "ACI-Bonf", "JointACI-sym", "CASA"]:
        pts = [(R["alpha"][f"{t}|{a}"][m]["viol"], R["alpha"][f"{t}|{a}"][m]["cost_inst_h"]) for a in [0.01, .02, .05, .1]]
        pts = np.array(pts); ax[i].plot(pts[:, 0], pts[:, 1], "o-", ms=3, lw=1, color=C[m], label=m)
    ax[i].set_title(t, fontsize=9); ax[i].set_xlabel("realised joint violation rate")
    ax[i].set_xscale("log")
ax[0].set_ylabel("cost (instance-hours)"); ax[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_tradeoff.png"); plt.close()

# Fig 6: CASA internals on Conversation, heterogeneous cost
o, s, *_ = run_trace("Conversation", cost=(1, 0.5), keep_series=True)
fig, ax = plt.subplots(2, 1, figsize=(7, 3.2), sharex=True)
x = np.arange(len(s["CASA_phi"])) / DAY
ax[0].plot(x, s["CASA_phi"], lw=.8, color=C["CASA"]); ax[0].set_ylabel("split φ_t")
ax[1].plot(x, s["CASA_a"], lw=.8, color=C["CASA"]); ax[1].axhline(.05, ls="--", color="k", lw=.8)
ax[1].set_ylabel("level α_t"); ax[1].set_xlabel("test day")
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_casa_state.png"); plt.close()
stats["casa_phi_mean_conv_het"] = float(np.mean(s["CASA_phi"]))

json.dump(stats, open(f"{RES}/stats.json", "w"), indent=1); print(json.dumps(stats, indent=1))
