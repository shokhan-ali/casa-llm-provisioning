# CASA: Cost-Aware Split-Adaptive Conformal Provisioning for Disaggregated LLM Serving

Code for the paper **"CASA: Split-Adaptive Online Conformal Provisioning with Distribution-Free Joint SLO Guarantees for Prefill–Decode Disaggregated LLM Serving"** ([Author names]; Dukan Technical Institute, Sulaimani Polytechnic University; 2026).

CASA sizes the two GPU pools of a prefill–decode (P/D) disaggregated LLM service jointly. It wraps any
forecaster, calibrates one joint score (the split-scaled maximum of the two normalized forecast residuals)
with adaptive conformal inference, and chooses the split of the risk budget between the pools to minimize
cost. The long-run fraction of windows in which either pool runs short is provably held at a target level α,
for any data sequence and any split policy. This repository contains the complete evaluation pipeline used in the paper.

## What is compared

All methods use the same forecaster, the same scale estimate, the same rounding rule and the same data splits.

| Group | Methods |
|---|---|
| No margin | Point (provision the point forecast) |
| Fixed headroom | Headroom25, Headroom50, Headroom-cal (tuned on the calibration day) |
| Per-pool conformal (Bonferroni) | SplitCP-Bonf (static), ACI-Bonf (adaptive) |
| Joint conformal, fixed split (ablation) | JointACI-sym |
| Proposed | CASA (joint conformal + cost-aware split) |
| Lower bound | Oracle (exact demand in hindsight) |

## Evaluation protocol

* Data: three week-long Microsoft Azure LLM inference traces (Code, Conversation, Multimodal; 45.1 million requests).
* Requests are aggregated to 5-minute provisioning windows; the demand of each pool is the peak-minute input tokens (prefill) or output tokens (decode).
* Each trace is split chronologically: days 1–2 train the forecaster, day 3 is the calibration day, days 4–7 (1,152 windows) are the test period.
* Every decision uses only data available h windows earlier (strictly causal replay). The forecaster is trained once and never retrained.
* A window is a joint SLO violation if either pool lacks whole-instance capacity.
* Per-instance capacity is normalized so that the training-period peak of each pool needs 32 instances; the cluster limit is 128 instances per pool.
* Default settings: α = 0.05, h = 1 window (5 minutes), γ = 0.005, buffer W = 288 windows.
* Metrics: joint and per-pool violation rate, worst-day violation rate, longest violation burst, cost in instance-hours, over-provisioning relative to the oracle, decision time.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or newer. No GPU is needed; the full suite runs in about 5 minutes on one CPU core.

## Data

The processed per-minute series used in the paper are included in the repository, so all experiments run without downloading anything:

* `minute_code.csv`, `minute_conv.csv`, `minute_mm.csv` (columns: minute, requests, input tokens, output tokens).

To rebuild them from the raw traces (CC-BY, Azure Public Dataset):

* **Code**: https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv
* **Conversation**: https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv
* **Multimodal**: https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLMMInferenceTrace_multimodal.csv.gz

```bash
python casa_aggregate.py AzureLLMInferenceTrace_code_1week.csv minute_code.csv
python casa_aggregate.py AzureLLMInferenceTrace_conv_1week.csv minute_conv.csv
python casa_aggregate.py AzureLMMInferenceTrace_multimodal.csv.gz minute_mm.csv
```

## Usage

```bash
# Main comparison on one trace (Table 3 of the paper)
python casa_sim.py --trace Code
python casa_sim.py --trace Conversation
python casa_sim.py --trace Multimodal

# Heterogeneous hardware: decode costs half of prefill
python casa_sim.py --trace Conversation --cost 1 0.5

# Other target level, lead time (in 5-minute windows) or forecaster
python casa_sim.py --trace Code --alpha 0.01
python casa_sim.py --trace Code --lead 6
python casa_sim.py --trace Code --model naive
```

## Reproducing all tables and figures

```bash
# All experiments of the paper (main comparison, alpha sweep, lead time, forecaster ablation,
# step-size sensitivity, cost-ratio sweep) -> results/all_results.json
python casa_all.py

# Figures 2-6 and the dependence statistics of Section 7.1 -> figures/, results/stats.json
python casa_figures.py

# Figure 1 (architecture)
python casa_fig_architecture.py
```

## Files

* `casa.py`: CASA and all baseline provisioning policies.
* `casa_sim.py`: windowing, forecaster, scale estimate, causal simulator and metrics; command-line entry point.
* `casa_all.py`: runs every experiment reported in the paper.
* `casa_figures.py`, `casa_fig_architecture.py`: figure scripts.
* `casa_aggregate.py`: converts the raw Azure traces into per-minute series.

## Output files

* `results/results_<trace>_a<alpha>_h<lead>_c<cP>-<cD>_<model>.csv`: one row per method with all metrics (from `casa_sim.py`);
* `results/all_results.json`: raw results of every experiment (from `casa_all.py`);
* `results/stats.json`: correlation and joint-tail statistics of Section 7.1;
* `figures/fig1_architecture.png` … `figures/fig6_casa_state.png`.

## Notes

* All randomness is seeded; running `casa_all.py` reproduces the numbers in the paper exactly.
* The guarantee of Theorem 1 is on the long-run average; individual days can exceed α (see the worst-day metric).
* Absolute instance counts depend on the capacity normalization; relative comparisons between methods are the main result.

## Citation

```
The BibTeX entry will be added after publication.
```

## License

MIT License (see the LICENSE file).
