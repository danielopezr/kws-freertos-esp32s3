"""Plan step 12: number of replicates from the pilot runs (the base configuration repeated).

    python pilot.py ../experiments/results/pilot [--effect 0.10] [--power 0.8] [--alpha 0.05]

Prints each run's responses, their run-to-run mean and SD, and for each response the
replicates per treatment that detect a difference of --effect x its mean between two
treatments (two-sided two-sample t test) with the given power. The plan keeps at least 5.
"""

import argparse
import math
import sys
import warnings

from statsmodels.stats.power import TTestIndPower
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import data

RESPONSES = ["det_latency_ms", "f1", "recall", "precision", "false_alarms", "loss_pct", "inf_mean_us"]
MIN_REPLICATES = 5


def replicates_needed(mean, sd, effect, power, alpha):
    """Replicates per treatment, or None when the response does not vary between runs."""
    delta = abs(effect * mean)
    if sd == 0 or delta == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        n = TTestIndPower().solve_power(effect_size=delta / sd, alpha=alpha, power=power, alternative="two-sided")
    # The solver does not converge when the effect is so large that fewer than 2 runs would do
    return 2 if math.isnan(n) else max(2, math.ceil(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="experiments/results/pilot")
    ap.add_argument("--effect", type=float, default=0.10, help="smallest relevant difference, fraction of the mean")
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df, _ = data.load(args.results)
    ok = data.ok_runs(df)
    if len(ok) < 2:
        sys.exit(f"only {len(ok)} successful runs: need at least 2")
    cols = [c for c in RESPONSES if c in ok.columns]
    print(ok[["run_id"] + cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n{len(ok)} runs ok of {len(df)}; difference to detect: {args.effect:.0%} of the mean, "
          f"power {args.power}, alpha {args.alpha}\n")
    print(f"{'response':16s} {'mean':>10s} {'sd':>10s} {'cv %':>7s} {'replicates':>11s}")
    for c in cols:
        x = ok[c].astype(float)
        mean, sd = x.mean(), x.std(ddof=1)
        n = replicates_needed(mean, sd, args.effect, args.power, args.alpha)
        cv = 100 * sd / mean if mean else float("nan")
        shown = "no spread" if n is None else str(n)
        print(f"{c:16s} {mean:10.4g} {sd:10.4g} {cv:7.2f} {shown:>11s}")
    lat = ok["det_latency_ms"].astype(float)
    n = replicates_needed(lat.mean(), lat.std(ddof=1), args.effect, args.power, args.alpha)
    use = max(MIN_REPLICATES, n or 0)
    print(f"\nDetection latency (the main response) needs {n} replicates; use {use} "
          f"(the plan's minimum is {MIN_REPLICATES}).")


if __name__ == "__main__":
    main()
