"""Calibrate the on-board measurements of one pipeline run against a logic analyzer capture.

    sigrok-cli -d fx2lafw --config samplerate=250k --time 325000 -C D0,D1,D2,D3,D4,D5 -o calib.sr
    python board.py COM7 "CFG model=cnn affinity=none prio=equal load=off mem=sram alloc=static frag=0 seq=full demo=off seed=1" --log calib.log
    python calibrate.py calib.sr calib.log --seed 1

Start the capture first: it must hold the whole run, from the first T1 block to the last. The
board measures with esp_timer (1 us); the analyzer sees the same events on the GPIO probes of
probes.h with its own clock. This tool compares, run-wide:

  clock         analyzer time vs board time of every detection (linear fit): rate in ppm and the
                residual disagreement of single instants
  detections    every DET of the board paired with an LED rising edge; the detection latency
                (analysis/scoring.py) computed from both
  external      the latency seen only on the pins, end-of-word MARK -> LED, plus the known
                position of voiced_end inside its 20 ms block (independent of the board's
                arithmetic)
  tasks         T1-T4 response time (probe pulse width) vs the TASK lines; T1 block period vs
                t1_gap_max_us; pulse counts vs activations

Default wiring: D0..D5 = T1, T2, T3, T4, MARK, LED (as check_probes.py).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.scoring import FIRST_KEYWORD_ID, SAMPLE_RATE, load_labels, score, windows  # noqa: E402
from check_probes import DEFAULT_MAP, edges, pulses  # noqa: E402
from experiments.protocol import parse_line  # noqa: E402

BLOCK = 320          # samples per I2S block (20 ms), pipeline.cc kBlock
JITTER_MAX = 1600    # pipeline.cc kJitterMax
JITTER_LEAD = 1600   # pipeline.cc kJitterLead
PAIR_TOL_S = 0.050   # a board DET and an LED edge are the same detection within this


# ---------------------------------------------------------------- firmware timeline (pipeline.cc)
class Timeline:
    """Silence inserted before each utterance, exactly as timeline_build() does on the board."""

    def __init__(self, labels, seed):
        x = seed or 0x9E3779B9
        self.at, self.len, self.cum = [], [], []
        total = 0
        for item in labels:
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= x >> 17
            x ^= (x << 5) & 0xFFFFFFFF
            self.at.append(item["start"] - JITTER_LEAD)
            self.len.append(x % JITTER_MAX)
            total += self.len[-1]
            self.cum.append(total)
        self.stream_at = [a + (self.cum[i - 1] if i else 0) for i, a in enumerate(self.at)]

    def insertion_before(self, s):
        j = int(np.searchsorted(self.stream_at, s, side="right")) - 1
        return j

    def to_seq(self, s):
        """Stream sample -> sequence sample (stream_to_seq)."""
        j = self.insertion_before(s)
        if j < 0:
            return s
        if s < self.stream_at[j] + self.len[j]:
            return self.at[j]
        return s - self.cum[j]

    def to_stream(self, q, near):
        """Sequence sample -> stream sample, in the insertion interval of stream sample `near`.

        None when q is an insertion point inside a silence (ambiguous)."""
        j = self.insertion_before(near)
        if j < 0:
            return q
        if q == self.at[j]:
            return None
        return q + self.cum[j]


# ---------------------------------------------------------------- helpers
def stats(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return None
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std()),
            "min": float(x.min()), "max": float(x.max())}


def read_log(path):
    dets, tasks, result = [], {}, None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        kind, payload = parse_line(line)
        if kind == "DET":
            dets.append(payload)
        elif kind == "TASK":
            tasks[payload["name"]] = payload
        elif kind == "RESULT":
            result = payload
    if result is None:
        raise ValueError(f"{path}: no RESULT line (did the run finish?)")
    return dets, tasks, result


def run_segment(rising, rate, gap_s=0.2):
    """(first, last) index of the longest stretch of T1 rises without a gap > gap_s.

    The capture may also hold the probe test that ran before the reset and boot glitches;
    the run is the long uninterrupted 20 ms activity."""
    if len(rising) < 2:
        raise ValueError("no T1 activity in the capture")
    breaks = np.nonzero(np.diff(rising) > gap_s * rate)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(rising) - 1]))
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def hit_latencies(dets, labels):
    """{utterance index: (latency ms, detection)} of the hits, with the rules of scoring.score()."""
    wins = windows(labels)
    starts = np.array([w[0] for w in wins])
    out = {}
    for d in sorted(dets, key=lambda d: d["sample"]):
        i = int(np.searchsorted(starts, d["sample"], side="right")) - 1
        if i < 0 or d["sample"] >= wins[i][1] or wins[i][2] < FIRST_KEYWORD_ID:
            continue
        if d["label_id"] == wins[i][2] and i not in out:
            out[i] = ((d["sample"] - labels[i]["voiced_end"]) * 1000 / SAMPLE_RATE, d)
    return out


# ---------------------------------------------------------------- calibration
def calibrate(capture, log, seed, labels=None, mapping=None):
    labels = labels or load_labels()
    mapping = mapping or DEFAULT_MAP
    tl = Timeline(labels, seed)
    dets, tasks, result = read_log(log)
    rate, total, found = edges(capture, sorted(set(mapping.values())))
    sig = {name: found[ch] for name, ch in mapping.items()}

    t1_rise, _ = sig["T1"]
    i0, i1 = run_segment(t1_rise, rate)
    t_first, t_last = t1_rise[i0], t1_rise[i1]
    if i0 == 0 or t_last + 0.1 * rate >= total:
        raise ValueError("the capture does not hold the whole run (T1 active at its start or end)")

    def window(rise_fall, margin_s=1.0):
        r, f = rise_fall
        keep = lambda e: e[(e >= t_first) & (e <= t_last + margin_s * rate)]
        return keep(r), keep(f)

    # ---- detections: board DET vs LED rising edges
    led_rise, _ = window(sig["LED"])
    led_dt = (led_rise - t_first) / rate                        # analyzer seconds since t_first

    def pair(scale, tol_s):
        """Board DET <-> LED edge, with analyzer time divided by `scale` (clock rate ratio)."""
        stream = BLOCK + np.floor(led_dt / scale * SAMPLE_RATE).astype(np.int64)
        seq = np.array([tl.to_seq(int(s)) for s in stream])
        out, used = [], set()
        for d in dets:
            q = d["sample"]
            k = int(np.argmin(np.abs(seq - q))) if len(seq) else -1
            if k < 0 or k in used or abs(seq[k] - q) > tol_s * SAMPLE_RATE:
                continue
            s = tl.to_stream(q, int(stream[k]))
            if s is None:
                continue
            used.add(k)
            # board time of the decision: stream = 320 + floor(dt * 16000), so dt is in [.., + 62.5 us)
            board_dt = (s - BLOCK + 0.5) / SAMPLE_RATE
            out.append((board_dt, float(led_dt[k]), d, int(seq[k]), int(led_rise[k])))
        if len(out) < 3:
            raise ValueError(f"only {len(out)} detections paired with an LED edge")
        return out

    # The two crystals differ by ~100 ppm, which adds up to tens of ms over a 5-min run: fit the
    # rate on a loose pairing, then pair again with analyzer time on the board's time base.
    pairs = pair(1.0, PAIR_TOL_S)
    bx = np.array([p[0] for p in pairs])
    ay = np.array([p[1] for p in pairs])
    slope, offset = np.polyfit(bx, ay, 1)
    resid_us = (ay - (offset + slope * bx)) * 1e6
    drift_ms = float((slope - 1) * bx.max() * 1e3)
    pairs = pair(slope, 0.005)

    led_of = {}  # id of a board detection -> its LED rising edge (capture sample)
    analyzer_dets, board_dets = [], []
    for p in pairs:
        board_dets.append(p[2])
        analyzer_dets.append({"sample": p[3], "label_id": p[2]["label_id"]})
        led_of[id(p[2])] = p[4]
    lat_board = hit_latencies(board_dets, labels)
    lat_analyzer = hit_latencies(analyzer_dets, labels)
    common = sorted(set(lat_board) & set(lat_analyzer))
    lat_diff = [lat_analyzer[i][0] - lat_board[i][0] for i in common]

    # ---- external latency: MARK of utterance i -> LED of its hit, plus voiced_end's place in its block
    mark_rise, _ = window(sig["MARK"])
    n_marks = len(mark_rise)
    ext = []
    if n_marks == len(labels):  # one MARK per utterance, in order
        for i, (lat, d) in lat_board.items():
            ve_stream = labels[i]["voiced_end"] + tl.cum[i]
            block_end = (ve_stream // BLOCK + 1) * BLOCK          # T1 handles the block once it is full
            lead_s = (block_end - ve_stream) / SAMPLE_RATE
            ext_ms = ((led_of[id(d)] - mark_rise[i]) / rate + lead_s) * 1e3
            ext.append((i, ext_ms, ext_ms - lat))

    # ---- tasks
    task_rows = {}
    for name in ("T1", "T2", "T3", "T4"):
        r, f = window(sig[name], margin_s=0.5)
        per, wid = pulses(r, f)
        w_us = stats(wid / rate * 1e6)
        t = tasks.get(name, {})
        task_rows[name] = {
            "pulses": int(len(r)), "activations": t.get("activations"),
            "width_us": w_us, "board_resp_mean_us": t.get("resp_mean_us"),
            "board_resp_max_us": t.get("resp_max_us"),
        }
    t1_per = np.diff(t1_rise[i0:i1 + 1]) / rate * 1e3

    return {
        "capture": {"samplerate": rate, "samples": int(total), "seconds": total / rate,
                    "run_seconds": float((t_last - t_first) / rate)},
        "clock": {"ppm": float((slope - 1) * 1e6), "offset_us": float(offset * 1e6), "drift_ms": drift_ms,
                  "residual_us": stats(resid_us), "pairs": len(pairs),
                  "board_dets": len(dets), "led_edges": int(len(led_rise))},
        "latency": {
            "board": score(board_dets, labels), "analyzer": score(analyzer_dets, labels),
            "hits_compared": len(common), "diff_ms": stats(lat_diff),
        },
        "external": {"marks": n_marks, "utterances": len(labels),
                     "ext_ms": stats([e[1] for e in ext]),
                     "ext_minus_board_ms": stats([e[2] for e in ext])},
        "tasks": task_rows,
        "t1_period_ms": stats(t1_per),
        "board": {"t1_gap_max_us": result.get("t1_gap_max_us"), "blocks_total": result.get("blocks_total"),
                  "blocks_lost": result.get("blocks_lost"), "inf_mean_us": result.get("inf_mean_us"),
                  "inferences": result.get("inferences")},
    }


def fmt(s, unit, nd=1):
    if s is None:
        return "-"
    return f"{s['mean']:.{nd}f} {unit} (sd {s['sd']:.{nd}f}, {s['min']:.{nd}f} .. {s['max']:.{nd}f}, n={s['n']})"


def print_report(rep):
    c, k, lat, ext = rep["capture"], rep["clock"], rep["latency"], rep["external"]
    print(f"capture: {c['seconds']:.1f} s at {c['samplerate'] / 1e3:g} kHz "
          f"({1e6 / c['samplerate']:.0f} us resolution); run on the pins: {c['run_seconds']:.2f} s")
    print(f"\nclock      analyzer vs board: {k['ppm']:+.1f} ppm ({k['drift_ms']:+.1f} ms over the run, removed below), "
          f"offset {k['offset_us']:+.1f} us")
    print(f"           residual per instant: {fmt(k['residual_us'], 'us')}")
    print(f"           paired {k['pairs']} of {k['board_dets']} board detections ({k['led_edges']} LED edges)")
    b, a = lat["board"], lat["analyzer"]
    print(f"\nlatency    board    {b['det_latency_ms']:.2f} ms mean, {b['det_latency_p95_ms']:.2f} ms P95, "
          f"{b['hits']} hits, F1 {b['f1']:.3f}")
    print(f"           analyzer {a['det_latency_ms']:.2f} ms mean, {a['det_latency_p95_ms']:.2f} ms P95, "
          f"{a['hits']} hits, F1 {a['f1']:.3f}")
    print(f"           analyzer - board per hit: {fmt(lat['diff_ms'], 'ms', 3)}")
    print(f"\nexternal   MARK -> LED + block lead: {fmt(ext['ext_ms'], 'ms', 2)}")
    print(f"           minus board latency:      {fmt(ext['ext_minus_board_ms'], 'ms', 3)}")
    print(f"           {ext['marks']} MARK pulses for {ext['utterances']} utterances")
    print("\ntask       pulses / activations   width on the pins (us)                 board resp mean / max (us)")
    for name, t in rep["tasks"].items():
        w = t["width_us"]
        wtxt = f"{w['mean']:8.1f} (max {w['max']:8.1f})" if w else "-"
        print(f"{name:10} {t['pulses']:6} / {t['activations']!s:6}        {wtxt:38} "
              f"{t['board_resp_mean_us']!s:>7} / {t['board_resp_max_us']!s}")
    p = rep["t1_period_ms"]
    print(f"\nT1 period  {fmt(p, 'ms', 3)}; board t1_gap_max {rep['board']['t1_gap_max_us']} us")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="sigrok session file (.sr) of the whole run")
    ap.add_argument("log", help="board.py --log of the same run")
    ap.add_argument("--seed", type=int, required=True, help="seed of the CFG line")
    ap.add_argument("--json", help="also write the report here")
    args = ap.parse_args()
    rep = calibrate(args.capture, args.log, args.seed)
    print_report(rep)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
