"""Score the detections of one run against the test sequence labels.

Each utterance owns a window from its voiced start until MAX_LATENCY_MS after its
voiced end (cut at the start of the next utterance). For every keyword:

  hit           first detection of the right word inside its window
                (latency = detection sample - voiced_end, in ms; can be negative)
  substitution  only wrong words detected inside its window
  miss          nothing detected inside its window

Every other detection is either a duplicate (the same right word again inside a
window that already has its hit: a debounce problem, reported apart) or a false
alarm (wrong word, anything during an unknown word, anything between words).

  recall = hits / keywords   precision = hits / (hits + false alarms)   F1
"""

import csv
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
MAX_LATENCY_MS = 1000
FIRST_KEYWORD_ID = 2  # 0 silence, 1 unknown
LABELS_CSV = Path(__file__).resolve().parents[1] / "test_sequence" / "labels.csv"


def load_labels(path=LABELS_CSV):
    with open(path, newline="") as f:
        rows = [{k: (v if k in ("label", "word", "source") else int(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]
    return sorted(rows, key=lambda r: r["start"])


def windows(labels, max_latency_ms=MAX_LATENCY_MS):
    """[(lo, hi, label_id)] in sample indices, one per utterance."""
    out = []
    for i, r in enumerate(labels):
        hi = r["voiced_end"] + max_latency_ms * SAMPLE_RATE // 1000
        if i + 1 < len(labels):
            hi = min(hi, labels[i + 1]["start"])
        out.append((r["voiced_start"], hi, r["label_id"]))
    return out


def score(detections, labels, max_latency_ms=MAX_LATENCY_MS):
    """detections: [{'sample', 'label_id', ...}] -> dict of per-run metrics."""
    wins = windows(labels, max_latency_ms)
    starts = np.array([w[0] for w in wins])
    used = [False] * len(wins)
    hit_latency = {}
    wrong_in_window = set()
    false_alarms = duplicates = 0
    for d in sorted(detections, key=lambda d: d["sample"]):
        i = int(np.searchsorted(starts, d["sample"], side="right")) - 1
        inside = i >= 0 and d["sample"] < wins[i][1]
        if not inside or wins[i][2] < FIRST_KEYWORD_ID:
            false_alarms += 1
        elif d["label_id"] == wins[i][2]:
            if used[i]:
                duplicates += 1
            else:
                used[i] = True
                hit_latency[i] = (d["sample"] - labels[i]["voiced_end"]) * 1000 / SAMPLE_RATE
        else:
            wrong_in_window.add(i)
            false_alarms += 1

    keywords = [i for i, w in enumerate(wins) if w[2] >= FIRST_KEYWORD_ID]
    hits = len(hit_latency)
    substitutions = sum(1 for i in keywords if i not in hit_latency and i in wrong_in_window)
    misses = len(keywords) - hits - substitutions
    recall = hits / len(keywords) if keywords else float("nan")
    precision = hits / (hits + false_alarms) if hits + false_alarms else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    lat = np.array(list(hit_latency.values()))
    return {
        "keywords": len(keywords), "hits": hits, "misses": misses, "substitutions": substitutions,
        "false_alarms": false_alarms, "duplicates": duplicates,
        "recall": recall, "precision": precision, "f1": f1,
        "det_latency_ms": float(lat.mean()) if len(lat) else float("nan"),
        "det_latency_median_ms": float(np.median(lat)) if len(lat) else float("nan"),
        "det_latency_p95_ms": float(np.percentile(lat, 95)) if len(lat) else float("nan"),
    }
