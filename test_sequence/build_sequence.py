"""Build the fixed test sequence that every experiment run plays back.

    python build_sequence.py --data-dir ../training/data/speech_commands [--download] [--out out]

Takes, from the Speech Commands v2 **test** split only, 10 utterances of each
keyword and 50 unknown words (spread evenly over the non-keyword words), each
from a different speaker, shuffles them with a fixed seed and lays them out as

    [1 s gap][1 s utterance slot][1 s gap][1 s utterance slot] ...

over a constant low-level noise floor (no digital zeros, like a real
microphone). 150 utterances -> 300 s, 16 kHz, 16-bit.

Outputs in --out:
  sequence.raw            int16 little-endian, written to the `sequence` flash partition
  sequence.wav            same audio, to listen to (skip with --no-wav)
  labels.csv              one row per utterance: class, sample ranges, source file
  kws_sequence_labels.h   the same table as C constants for the firmware (T1 playback, marker pin)
  manifest.json           parameters, counts, size and SHA-256 of sequence.raw

`start`/`end` are where the clip audio sits; `voiced_start`/`voiced_end` are an
energy-based estimate of the spoken word inside the clip (see voiced_range).
Detection latency is measured from `voiced_end`.
"""

import argparse
import csv
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from kws import dataset as D  # noqa: E402
from kws import features as F  # noqa: E402

PARTITIONS = ROOT / "firmware" / "partitions.csv"
PARTITION_NAME = "sequence"
ENERGY_FRAME = F.SAMPLE_RATE // 50  # 20 ms, one capture block
VOICED_REL_DB = 25.0
VOICED_ABOVE_FLOOR_DB = 10.0
VOICED_MAX_GAP = 5  # frames (100 ms)


def speaker(rel_path):
    """Speech Commands file names are <speaker>_nohash_<n>.wav."""
    return Path(rel_path).name.split("_nohash_")[0]


def select(test_files, rng, per_keyword=10, n_unknown=50):
    """[(rel_path, word, label_id)] with one utterance per speaker, keywords first then unknown."""
    by_word = {}
    for rel, word in test_files:
        by_word.setdefault(word, []).append(rel)
    unknown_words = sorted(w for w in by_word if w not in D.KEYWORDS)
    missing = [w for w in D.KEYWORDS if w not in by_word]
    if missing or not unknown_words:
        raise ValueError(f"test split lacks keywords {missing} or unknown words")

    # Unknown quota: as even as possible, the remainder goes to randomly chosen words
    quota = {w: n_unknown // len(unknown_words) for w in unknown_words}
    for i in rng.choice(len(unknown_words), n_unknown % len(unknown_words), replace=False):
        quota[unknown_words[i]] += 1

    used, chosen = set(), []
    for word, k in [(w, per_keyword) for w in D.KEYWORDS] + [(w, quota[w]) for w in unknown_words]:
        cands = by_word[word]
        got = 0
        for j in rng.permutation(len(cands)):
            if got == k:
                break
            if speaker(cands[j]) in used:
                continue
            used.add(speaker(cands[j]))
            label = D.LABELS.index(word) if word in D.KEYWORDS else D.UNKNOWN
            chosen.append((cands[j], word, label))
            got += 1
        if got < k:
            raise ValueError(f"only {got} distinct-speaker clips of '{word}', need {k}")
    return chosen


def voiced_range(x):
    """(start, end) sample of the spoken word inside one clip, in 20 ms frames.

    A frame is "active" when it is within VOICED_REL_DB of the loudest frame and at
    least VOICED_ABOVE_FLOOR_DB above the clip's own noise floor (10th percentile),
    so noisy recordings are not marked active end to end. The word is the run of
    active frames around the loudest one, bridging pauses of up to VOICED_MAX_GAP
    frames (stop closures); isolated clicks further away are ignored.
    """
    n = max(1, len(x) // ENERGY_FRAME)
    frames = x[:n * ENERGY_FRAME].reshape(n, -1) if len(x) >= ENERGY_FRAME else x[None, :]
    rms = np.sqrt((np.asarray(frames, dtype=np.float64) ** 2).mean(axis=1))
    if rms.max() == 0:
        return 0, len(x)
    db = 20 * np.log10(rms + 1e-12)
    thr = min(max(db.max() - VOICED_REL_DB, np.percentile(db, 10) + VOICED_ABOVE_FLOOR_DB), db.max())
    active = np.nonzero(db >= thr)[0]
    peak = int(db.argmax())
    lo = hi = int(np.searchsorted(active, peak))
    while hi + 1 < len(active) and active[hi + 1] - active[hi] <= VOICED_MAX_GAP + 1:
        hi += 1
    while lo > 0 and active[lo] - active[lo - 1] <= VOICED_MAX_GAP + 1:
        lo -= 1
    return int(active[lo] * ENERGY_FRAME), int(min(len(x), (active[hi] + 1) * ENERGY_FRAME))


def label_voiced(pcm, rows):
    """Fill voiced_start / voiced_end from the final int16 audio (so --relabel reproduces it)."""
    for r in rows:
        v0, v1 = voiced_range(pcm[r["start"]:r["end"]].astype(np.float64) / 32768)
        r["voiced_start"], r["voiced_end"] = r["start"] + v0, r["start"] + v1
    return rows


def build(data_dir, seed=2026, per_keyword=10, n_unknown=50, gap_ms=1000, floor_dbfs=-60.0):
    """Return (int16 audio, rows). Fully determined by the dataset and the arguments."""
    rng = np.random.default_rng(seed)
    items = select(D.split_files(data_dir)["test"], rng, per_keyword, n_unknown)
    items = [items[i] for i in rng.permutation(len(items))]

    gap = F.SAMPLE_RATE * gap_ms // 1000
    slot = F.CLIP_SAMPLES
    audio = rng.normal(0.0, 10 ** (floor_dbfs / 20), len(items) * (gap + slot))
    rows = []
    for i, (rel, word, label) in enumerate(items):
        x = D.read_wav(Path(data_dir) / rel)[:slot]
        start = i * (gap + slot) + gap
        audio[start:start + len(x)] += x
        rows.append({"index": i, "label": D.LABELS[label], "label_id": label, "word": word,
                     "start": start, "end": start + len(x),
                     "voiced_start": 0, "voiced_end": 0, "source": rel})
    pcm = np.clip(np.round(audio * 32768), -32768, 32767).astype("<i2")
    return pcm, label_voiced(pcm, rows)


def relabel(out):
    """Recompute the voiced ranges of an existing build from sequence.raw (no dataset needed)."""
    out = Path(out)
    pcm = np.fromfile(out / "sequence.raw", dtype="<i2")
    with open(out / "labels.csv", newline="") as f:
        rows = [{k: (v if k in ("label", "word", "source") else int(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]
    params = json.loads((out / "manifest.json").read_text())["params"]
    return write_outputs(out, pcm, label_voiced(pcm, rows), params, wav=False)


def partition_size(name=PARTITION_NAME, path=PARTITIONS):
    for line in Path(path).read_text().splitlines():
        cols = [c.strip() for c in line.split("#")[0].split(",")]
        if cols[0] == name:
            return int(cols[4], 0)
    raise ValueError(f"partition '{name}' not in {path}")


def render_header(rows, total_samples):
    lines = [
        "/* Generated by test_sequence/build_sequence.py. Do not edit. */",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define KWS_SEQ_SAMPLE_RATE {F.SAMPLE_RATE}",
        f"#define KWS_SEQ_TOTAL_SAMPLES {total_samples}u",
        f"#define KWS_SEQ_N_ITEMS {len(rows)}",
        "",
        "/* Sample indices into the `sequence` partition; label is the model class id. */",
        "typedef struct {",
        "    uint32_t start, end, voiced_start, voiced_end;",
        "    uint8_t label;",
        "} kws_seq_item_t;",
        "",
        f"static const char *const kws_seq_label_names[{len(D.LABELS)}] = {{"
        + ", ".join(f'"{n}"' for n in D.LABELS) + "};",
        "",
        "static const kws_seq_item_t kws_seq_items[KWS_SEQ_N_ITEMS] = {",
    ]
    lines += [f"    {{{r['start']}u, {r['end']}u, {r['voiced_start']}u, {r['voiced_end']}u, {r['label_id']}}},"
              f"  /* {r['index']:3d} {r['word']} */" for r in rows]
    lines += ["};", ""]
    return "\n".join(lines)


def write_outputs(out, pcm, rows, params, wav=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = pcm.tobytes()
    limit = partition_size()
    if len(raw) > limit:
        raise ValueError(f"sequence is {len(raw)} B but partition '{PARTITION_NAME}' holds {limit} B")
    (out / "sequence.raw").write_bytes(raw)
    if wav:
        with wave.open(str(out / "sequence.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(F.SAMPLE_RATE)
            w.writeframes(raw)
    with open(out / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "kws_sequence_labels.h").write_text(render_header(rows, len(pcm)), newline="\n")
    counts = {name: sum(r["label"] == name for r in rows) for name in D.LABELS}
    manifest = {
        "params": params, "sample_rate": F.SAMPLE_RATE, "samples": len(pcm),
        "seconds": len(pcm) / F.SAMPLE_RATE, "bytes": len(raw), "partition_bytes": limit,
        "sha256": hashlib.sha256(raw).hexdigest(), "items": len(rows),
        "counts": {k: v for k, v in counts.items() if v}, "speakers": len({speaker(r["source"]) for r in rows}),
        "dataset": D.DATA_URL, "split": "test",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "training" / "data" / "speech_commands"))
    ap.add_argument("--download", action="store_true", help="download Speech Commands v2 if missing")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--per-keyword", type=int, default=10)
    ap.add_argument("--unknown", type=int, default=50)
    ap.add_argument("--gap-ms", type=int, default=1000)
    ap.add_argument("--floor-dbfs", type=float, default=-60.0, help="RMS of the constant noise floor")
    ap.add_argument("--no-wav", action="store_true")
    ap.add_argument("--relabel", action="store_true",
                    help="only recompute voiced ranges of the build in --out from its sequence.raw")
    args = ap.parse_args()

    if args.relabel:
        m = relabel(args.out)
        print(f"relabelled {m['items']} utterances in {args.out} (sha256 {m['sha256']})")
        return
    if args.download:
        D.download(args.data_dir)
    params = {k: getattr(args, k) for k in ("seed", "per_keyword", "unknown", "gap_ms", "floor_dbfs")}
    pcm, rows = build(args.data_dir, args.seed, args.per_keyword, args.unknown, args.gap_ms, args.floor_dbfs)
    m = write_outputs(args.out, pcm, rows, params, wav=not args.no_wav)
    print(f"{m['items']} utterances from {m['speakers']} speakers, {m['seconds']:.1f} s, "
          f"{m['bytes'] / 1e6:.2f} MB of {m['partition_bytes'] / 1e6:.2f} MB partition")
    print(f"counts: {m['counts']}")
    print(f"sha256: {m['sha256']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
