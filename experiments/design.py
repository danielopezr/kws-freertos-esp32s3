"""Factorial designs: load a designs/*.yaml file and expand it into a randomized run plan.

A design has fixed settings, factors with levels and (optionally) presets: a factor
whose levels stand for whole groups of settings (experiment D's strategies).
Every run is one treatment x one replicate; the order of all runs is shuffled with
the design's seed and each run gets its own seed, also derived from it.
"""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from protocol import PARAMS, ProtocolError, normalize, validate


class DesignError(ValueError):
    pass


def load(path):
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "replicates", "seed", "factors"):
        if key not in d:
            raise DesignError(f"{path}: missing '{key}'")
    d.setdefault("fixed", {})
    d.setdefault("presets", {})
    d["fixed"] = {k: normalize(v) for k, v in d["fixed"].items()}
    d["factors"] = {k: [normalize(v) for v in levels] for k, levels in d["factors"].items()}
    for factor, table in d["presets"].items():
        if factor not in d["factors"]:
            raise DesignError(f"preset for '{factor}', which is not a factor")
        undefined = [lvl for lvl in d["factors"][factor] if not table.get(lvl)]
        if undefined:
            raise DesignError(f"{d['name']}: preset {factor}={undefined} not defined yet (see the design file)")
    return d


def treatments(d):
    """[(treatment_id, factor levels, full CFG settings)] in factor order."""
    names = list(d["factors"])
    out = []
    for i, levels in enumerate(itertools.product(*d["factors"].values()), start=1):
        factors = dict(zip(names, levels))
        cfg = dict(d["fixed"])
        for name, level in factors.items():
            if name in d["presets"]:
                cfg.update({k: normalize(v) for k, v in d["presets"][name][level].items()})
            elif name in PARAMS:
                cfg[name] = level
            else:
                raise DesignError(f"factor '{name}' is neither a protocol key nor a preset")
        try:
            cfg = validate(cfg)
        except ProtocolError as e:
            raise DesignError(f"{d['name']} treatment {factors}: {e}") from None
        if cfg["demo"] != "off":
            raise DesignError(f"{d['name']}: demo=on only for the presentation; the OLED would perturb the runs")
        out.append((i, factors, cfg))
    return out


def plan(d):
    """Randomized list of runs: dicts with order, run_id, treatment, replicate, seed, factors, cfg."""
    trts = treatments(d)
    runs = [(t, r) for t in trts for r in range(1, d["replicates"] + 1)]
    rng = np.random.default_rng(d["seed"])
    order = rng.permutation(len(runs))
    seeds = rng.integers(0, 2**31 - 1, len(runs))
    out = []
    for pos, (idx, seed) in enumerate(zip(order, seeds), start=1):
        (tid, factors, cfg), rep = runs[idx]
        out.append({"order": pos, "run_id": f"{d['name']}-T{tid:02d}-R{rep}", "treatment": tid,
                    "replicate": rep, "seed": int(seed), "factors": factors, "cfg": cfg})
    return out


def fingerprint(d):
    """Hash of everything that defines the runs, to refuse resuming with an edited design."""
    keys = ("name", "replicates", "seed", "fixed", "factors", "presets")
    blob = json.dumps({k: d[k] for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
