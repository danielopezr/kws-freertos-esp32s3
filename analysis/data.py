"""Load one experiment's results into a table: one row per run, one column per factor and response."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scoring


def load(results_dir, labels=None):
    """(DataFrame of all recorded runs, list of factor names).

    Adds the scored detection metrics (see scoring.py) and the audio loss in %.
    Failed runs are kept with their status so they can be reported; the analysis
    uses only status == 'ok'.
    """
    results_dir = Path(results_dir)
    plan = json.loads((results_dir / "plan.json").read_text())
    factors = list(plan["runs"][0]["factors"])
    df = pd.read_csv(results_dir / "index.csv", dtype={f: str for f in factors})
    labels = labels if labels is not None else scoring.load_labels()

    scored = []
    for run_id in df["run_id"]:
        rec = json.loads((results_dir / "runs" / f"{run_id}.json").read_text())
        full_sequence = rec["cfg"].get("seq", "full") == "full"
        if rec["status"] == "ok" and full_sequence and not rec["result"].get("alloc_fail"):
            scored.append(scoring.score(rec["detections"], labels))
        else:
            scored.append({})
    df = pd.concat([df, pd.DataFrame(scored, index=df.index)], axis=1)
    if {"blocks_lost", "blocks_total"} <= set(df.columns):
        df["loss_pct"] = 100 * df["blocks_lost"] / df["blocks_total"].replace(0, np.nan)
    df["treatment_label"] = df[factors].astype(str).agg(" / ".join, axis=1)
    # Keep the level order of the design file (lower, equal, higher), not alphabetical
    for f in factors:
        order = list(dict.fromkeys(r["factors"][f] for r in sorted(plan["runs"], key=lambda r: r["treatment"])))
        df[f] = pd.Categorical(df[f], categories=order, ordered=True)
    return df, factors


def ok_runs(df):
    return df[df["status"] == "ok"].copy()
