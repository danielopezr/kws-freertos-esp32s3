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
energy-based estimate of the spoken word inside the clip (20 ms frames within
VOICED_DB of the loudest frame). Detection latency is measured from `voiced_end`.
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
VOICED_DB = 30.0


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
    """(first, last+1) sample of the 20 ms frames within VOICED_DB of the loudest one."""
    n = max(1, len(x) // ENERGY_FRAME)
    frames = x[:n * ENERGY_FRAME].reshape(n, -1) if len(x) >= ENERGY_FRAME else x[None, :]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    if rms.max() == 0:
        return 0, len(x)
    db = 20 * np.log10(rms / rms.max() + 1e-12)
    idx = np.nonzero(db >= -VOICED_DB)[0]
    return int(idx[0] * ENERGY_FRAME), int(min(len(x), (idx[-1] + 1) * ENERGY_FRAME))


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
        v0, v1 = voiced_range(x)
        rows.append({"index": i, "label": D.LABELS[label], "label_id": label, "word": word,
                     "start": start, "end": start + len(x),
                     "voiced_start": start + v0, "voiced_end": start + v1, "source": rel})
    pcm = np.clip(np.round(audio * 32768), -32768, 32767).astype("<i2")
    return pcm, rows


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
    args = ap.parse_args()

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
