"""Runs every experiment reported in the paper and stores raw results as JSON."""
import json, numpy as np
from casa_sim import run_trace, RES
from casa import JointACI, CASA, ACIBonferroni

TR = ["Code", "Conversation", "Multimodal"]
R = {}

# E1: main comparison, alpha=0.05, h=1, two cost settings
R["main"] = {f"{t}|{c}": run_trace(t, cost=c) for t in TR for c in [(1, 1), (1, 0.5)]}

# E2: target level sweep
R["alpha"] = {f"{t}|{a}": run_trace(t, alpha=a) for t in TR for a in [0.01, 0.02, 0.05, 0.10]}

# E3: provisioning lead-time sweep (h windows of 5 min)
R["horizon"] = {f"{t}|{h}": run_trace(t, h=h) for t in TR for h in [1, 3, 6]}

# E4: forecaster ablation (persistence forecaster)
R["naive"] = {t: run_trace(t, model="naive") for t in TR}

# E5: step-size sensitivity for the conformal methods
def conf_only(alpha, Z, cost, mu, g):
    return [ACIBonferroni(alpha, Z, g), JointACI(alpha, Z, g), CASA(alpha, Z, cost, mu, g)]
R["gamma"] = {f"{t}|{g}": run_trace(t, gamma=g, policies=conf_only)
              for t in TR for g in [0.001, 0.005, 0.01, 0.05]}

# E6: cost-asymmetry sweep (decode cost relative to prefill)
def split_only(alpha, Z, cost, mu, g):
    return [JointACI(alpha, Z, g), CASA(alpha, Z, cost, mu, g)]
R["costratio"] = {f"{t}|{r}": run_trace(t, cost=(1, r), policies=split_only)
                  for t in TR for r in [0.25, 0.5, 1.0, 2.0, 4.0]}

json.dump(R, open(f"{RES}/all_results.json", "w"), indent=1)
print("saved")
