# analysis

Turns the runs recorded by [`experiments/runner.py`](../experiments/runner.py) into the statistics of the report.

```bash
pip install -r requirements.txt
python analyze.py ../experiments/results/A          # -> reports/A/report.md + tables + figures
python -m unittest discover -s tests
```

## What it does

| Step | File | Details |
| --- | --- | --- |
| Merge | [`data.py`](data.py) | `index.csv` + every run's detections → one row per run; factor levels kept in design order |
| Score | [`scoring.py`](scoring.py) | Detections vs. [`test_sequence/labels.csv`](../test_sequence/labels.csv): hits, misses, substitutions, false alarms, duplicates, recall, precision, F1, detection latency from `voiced_end` (window: up to 1 s after the word) |
| Describe | [`stats.py`](stats.py) | Mean, SD, 95 % CI, P95 and max per treatment |
| ANOVA | `stats.anova` | Factorial, all interactions (main effects only with 1 run per treatment), Type III with sum-to-zero contrasts, F, p and partial η² |
| Assumptions | `stats.assumptions` | Shapiro-Wilk on residuals and Levene (median); if they fail: same ANOVA on log(response) and Kruskal-Wallis |
| Comparisons | `stats.tukey` | Tukey HSD for every significant term (interaction cells or main-effect levels) |
| Proportions | `stats.logistic` | Quasi-binomial GLM for blocks lost / total and keywords missed / keywords (Wald χ² per term) |
| Figures | [`plots.py`](plots.py) | Box plot per treatment; interaction plot with 95 % CI (experiment A: priority × affinity, one panel per load level) |

Each observation is the summary of one run (no pseudoreplication). Failed runs are listed and excluded. Allocation failures (experiment C) are reported as a result.

| Experiment | Responses analysed by default |
| --- | --- |
| A | detection latency, audio loss %, F1 |
| B | mean inference time, detection latency, F1, minimum free heap |
| C | largest free block (+ allocation failures) |
| D | detection latency, audio loss %, F1 |

`--responses col1,col2` analyses other columns of `runs.csv`.

## With simulated data

```bash
cd ../experiments
python sim_board.py --port 5555 &
python runner.py designs/A.yaml --port socket://localhost:5555 --out sim_results/A --done-timeout 5
cd ../analysis
python analyze.py ../experiments/sim_results/A --out reports/sim_A
```

Reports from simulated data go to `reports/sim_*` and are not versioned.
