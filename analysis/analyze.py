"""Analyse one experiment and write a report with tables and figures.

    python analyze.py ../experiments/results/A                 # -> reports/A/
    python analyze.py ../experiments/sim_results/A --out reports/sim_A
    python analyze.py RESULTS --responses det_latency_ms,loss_pct

For each response: descriptive statistics per treatment, factorial ANOVA with
partial eta squared, assumption checks (Shapiro-Wilk, Levene) with a log
transform and Kruskal-Wallis as fallbacks, Tukey HSD on the significant terms,
box plot and interaction plot. Counts out of a total (audio blocks lost,
keywords missed) also get a quasi-binomial logistic regression.

Writes report.md, runs.csv (the merged table) and one CSV per table into --out.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import data
import plots
import stats

HERE = Path(__file__).resolve().parent

# Main responses per experiment (the design name is the results folder's plan)
RESPONSES = {
    "A": ["det_latency_ms", "loss_pct", "f1"],
    "B": ["inf_mean_us", "det_latency_ms", "f1", "heap_free_min"],
    "C": ["heap_largest_block"],
    "D": ["det_latency_ms", "loss_pct", "f1"],
}
DEFAULT_RESPONSES = ["det_latency_ms", "loss_pct", "f1"]
# (successes column, total column) analysed as proportions
PROPORTIONS = {"loss": ("blocks_lost", "blocks_total"), "missed keywords": ("misses", "keywords")}
# Central hypothesis of experiment A: the priority effect depends on whether tasks share a core
INTERACTION_AXES = {"A": ("prio", "affinity", "load")}


def md_table(df, floatfmt=".4g"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).to_numpy()]
    return "\n".join([head, sep, *rows])


def p_text(p):
    return "n/a (needs 2+ runs per treatment)" if pd.isna(p) else f"{p:.3g}"


def interaction_axes(name, factors):
    if name in INTERACTION_AXES and set(INTERACTION_AXES[name]) <= set(factors):
        return INTERACTION_AXES[name]
    if len(factors) >= 2:
        return factors[0], factors[1], (factors[2] if len(factors) > 2 else None)
    return None


def analyse_response(df, factors, response, out, name):
    lines = [f"## {plots.label(response)} (`{response}`)", ""]
    desc = stats.descriptive(df, factors, response)
    desc.to_csv(out / f"{response}_descriptive.csv", index=False)
    lines += ["### Descriptive statistics per treatment", "", md_table(desc), ""]
    if df[response].dropna().nunique() < 2:
        return lines + ["No variation in this response: ANOVA not applicable.", ""]

    table, model = stats.anova(df, factors, response)
    check = stats.assumptions(model, df, factors, response)
    reps = stats.replicates(df.dropna(subset=[response]), factors)
    kind = "with all interactions" if reps >= 2 else "main effects only (1 run per treatment)"
    lines += [f"### Factorial ANOVA ({kind}, Type III)", "",
              md_table(table.reset_index(names="term")[["term", "df", "sum_sq", "F", "p", "partial_eta2",
                                                        "significant"]]), "",
              f"Assumptions: Shapiro-Wilk on residuals p = {p_text(check['shapiro_p'])}, "
              f"Levene (median) p = {p_text(check['levene_p'])} -> "
              f"{'OK' if check['ok'] else '**not met**'}.", ""]
    table.to_csv(out / f"{response}_anova.csv")
    final = table

    if not check["ok"]:
        if (df[response].dropna() > 0).all():
            logged = df.assign(**{f"log_{response}": np.log(df[response])})
            ltable, lmodel = stats.anova(logged, factors, f"log_{response}")
            lcheck = stats.assumptions(lmodel, logged, factors, f"log_{response}")
            ltable.to_csv(out / f"{response}_anova_log.csv")
            lines += ["#### Same ANOVA on log(response)", "",
                      md_table(ltable.reset_index(names="term")[["term", "df", "F", "p", "partial_eta2",
                                                                 "significant"]]), "",
                      f"Assumptions on the log scale: Shapiro-Wilk p = {p_text(lcheck['shapiro_p'])}, "
                      f"Levene p = {p_text(lcheck['levene_p'])} -> "
                      f"{'OK' if lcheck['ok'] else 'still not met'}.", ""]
            if lcheck["ok"]:
                final = ltable
        kw = stats.kruskal(df, factors, response)
        lines += [f"Kruskal-Wallis across treatments (non-parametric check): H = {kw['H']:.3g}, "
                  f"p = {kw['p']:.3g}.", ""]

    terms = stats.significant_terms(final, factors)
    if terms:
        lines += ["### Tukey HSD on significant terms", ""]
        if not check["ok"] and final is table:
            lines += ["> Assumptions are not met even on the log scale: read these comparisons with care and "
                      "rely on the Kruskal-Wallis test and, for proportions, the logistic regression below.", ""]
        for term in terms:
            tk = stats.tukey(df, term, response)
            fname = f"{response}_tukey_{'_x_'.join(term)}.csv"
            tk.to_csv(out / fname, index=False)
            sig = tk[tk["significant"]]
            lines += [f"**{' x '.join(term)}**: {len(sig)} of {len(tk)} pairs differ "
                      f"([all pairs]({fname})).", ""]
            if len(sig):
                lines += [md_table(sig.head(15)), ""]
    else:
        lines += ["No significant term at alpha = 0.05.", ""]

    box = f"{response}_box.png"
    plots.boxplot(df, factors, response, out / box)
    lines += [f"![box plot]({box})", ""]
    axes = interaction_axes(name, factors)
    if axes:
        fig = f"{response}_interaction.png"
        plots.interaction(df, axes[0], axes[1], response, out / fig, panel=axes[2])
        lines += [f"![interaction {axes[0]} x {axes[1]}]({fig})", ""]
    return lines


def analyse(results_dir, out=None, responses=None):
    results_dir = Path(results_dir)
    df_all, factors = data.load(results_dir)
    name = df_all["run_id"].iloc[0].split("-")[0]
    out = Path(out) if out else HERE / "reports" / name
    out.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out / "runs.csv", index=False)
    df = data.ok_runs(df_all)
    responses = responses or [r for r in RESPONSES.get(name, DEFAULT_RESPONSES) if r in df.columns]

    status = df_all["status"].value_counts().to_dict()
    lines = [f"# Experiment {name}", "",
             f"Source: `{results_dir}`. Factors: {', '.join(f'`{f}`' for f in factors)}. "
             f"Runs: {len(df_all)} recorded, {len(df)} ok "
             f"({', '.join(f'{k}: {v}' for k, v in sorted(status.items()))}). "
             "Each observation is the summary of one run (no pseudoreplication); failed runs are excluded.", ""]
    if "alloc_fail" in df.columns and df["alloc_fail"].fillna(0).any():
        fails = df[df["alloc_fail"] == 1]["treatment_label"].value_counts()
        lines += ["**Allocation failures** (a result, not an error): "
                  + ", ".join(f"{k} ({v})" for k, v in fails.items()), ""]

    for response in responses:
        lines += analyse_response(df, factors, response, out, name)

    lines += ["## Proportions (quasi-binomial logistic regression)", ""]
    for title, (k, n) in PROPORTIONS.items():
        if {k, n} <= set(df.columns):
            wald = stats.logistic(df, factors, k, n)
            if wald is None:
                lines += [f"- {title}: no variation or not estimable.", ""]
            else:
                wald.to_csv(out / f"logistic_{k}.csv")
                lines += [f"### {title} (`{k}` / `{n}`)", "", md_table(wald.reset_index(names="term")), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out, df_all, factors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="experiments/results/<design> (with plan.json and index.csv)")
    ap.add_argument("--out", help="report folder (default analysis/reports/<design>)")
    ap.add_argument("--responses", help="comma-separated response columns")
    args = ap.parse_args()
    responses = args.responses.split(",") if args.responses else None
    try:
        out, df, _ = analyse(args.results, args.out, responses)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    print(f"{len(df)} runs -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
