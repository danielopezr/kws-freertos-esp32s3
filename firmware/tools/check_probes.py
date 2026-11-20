"""Check a logic analyzer capture of the probe test (plan step 5) against firmware/main/probes.h.

    sigrok-cli -d fx2lafw --config samplerate=1m --time 5000 -C D0,D1,D2,D3,D4,D5 -o probes.sr
    python check_probes.py probes.sr

Reads a sigrok session file (.sr, as saved by PulseView or sigrok-cli) in chunks,
finds the rising and falling edges of each channel and compares the mean period
and pulse width with the values the firmware is built with. A channel passes when
both means are within 1 % of nominal (or 2 samples, whichever is larger) and at
least 3 periods were seen. Exit code 0 only if every channel passes.

Default wiring: D0..D5 = T1, T2, T3, T4, MARK, LED (change with --map).
"""

import argparse
import configparser
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

PROBES_H = Path(__file__).resolve().parents[1] / "main" / "probes.h"
SIGNALS = ("T1", "T2", "T3", "T4", "MARK", "LED")
DEFAULT_MAP = {name: i for i, name in enumerate(SIGNALS)}
TOLERANCE = 0.01
MIN_PERIODS = 3


def expectations(path=PROBES_H):
    """{signal: (period_s, width_s)} from the #defines in probes.h."""
    text = Path(path).read_text()
    defs = {k: int(v) for k, v in re.findall(r"#define\s+PROBE_(\w+)\s+(\d+)\b", text)}
    out = {}
    for name in SIGNALS:
        period = defs[f"{name}_PERIOD_MS"] / 1e3
        width = defs[f"{name}_BUSY_US"] / 1e6
        out[name] = (period, width)
    return out


def parse_samplerate(text):
    m = re.fullmatch(r"\s*([\d.]+)\s*([kMG]?)Hz\s*", text)
    if not m:
        raise ValueError(f"unrecognised samplerate {text!r}")
    return float(m.group(1)) * {"": 1, "k": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]


def read_sr(path):
    """Yield (samplerate, unitsize) first, then uint arrays of samples, chunk by chunk."""
    with zipfile.ZipFile(path) as z:
        meta = configparser.ConfigParser()
        meta.read_string(z.read("metadata").decode())
        dev = meta["device 1"]
        rate = parse_samplerate(dev["samplerate"])
        unitsize = int(dev.get("unitsize", "1"))
        prefix = dev.get("capturefile", "logic-1")
        chunks = sorted((n for n in z.namelist() if n == prefix or n.startswith(prefix + "-")),
                        key=lambda n: int(n.rsplit("-", 1)[1]) if n != prefix else 0)
        if not chunks:
            raise ValueError(f"{path}: no logic data")
        yield rate, unitsize
        dtype = {1: np.uint8, 2: "<u2", 4: "<u4"}[unitsize]
        for name in chunks:
            yield np.frombuffer(z.read(name), dtype=dtype)


def edges(path, channels):
    """(samplerate, total samples, {channel: (rising, falling)}) as sample indices."""
    it = read_sr(path)
    rate, _ = next(it)
    found = {c: ([], []) for c in channels}
    prev = {c: None for c in channels}
    offset = 0
    for data in it:
        for c in channels:
            bits = ((data >> c) & 1).astype(np.int8)
            first = bits[0] if prev[c] is None else prev[c]
            d = np.diff(np.concatenate(([first], bits)))
            found[c][0].append(np.nonzero(d == 1)[0] + offset)
            found[c][1].append(np.nonzero(d == -1)[0] + offset)
            prev[c] = bits[-1]
        offset += len(data)
    return rate, offset, {c: (np.concatenate(r), np.concatenate(f)) for c, (r, f) in found.items()}


def pulses(rising, falling):
    """Periods (rise to rise) and widths (rise to next fall), in samples."""
    periods = np.diff(rising)
    if len(rising) and len(falling):
        idx = np.searchsorted(falling, rising, side="right")
        ok = idx < len(falling)
        widths = falling[idx[ok]] - rising[ok]
    else:
        widths = np.array([], dtype=np.int64)
    return periods, widths


def summarize(samples, rate):
    if len(samples) == 0:
        return None
    s = samples / rate
    return {"n": int(len(s)), "mean": float(s.mean()), "sd": float(s.std()),
            "min": float(s.min()), "max": float(s.max())}


def within(measured, nominal, rate):
    return measured is not None and abs(measured["mean"] - nominal) <= max(TOLERANCE * nominal, 2 / rate)


def check(path, mapping=None, expected=None):
    mapping = mapping or DEFAULT_MAP
    expected = expected or expectations()
    rate, total, found = edges(path, sorted(set(mapping.values())))
    report = []
    for name, ch in mapping.items():
        period, width = pulses(*found[ch])
        p, w = summarize(period, rate), summarize(width, rate)
        exp_p, exp_w = expected[name]
        ok = within(p, exp_p, rate) and within(w, exp_w, rate) and p["n"] >= MIN_PERIODS
        report.append({"signal": name, "channel": ch, "period": p, "width": w,
                       "expected_period": exp_p, "expected_width": exp_w, "ok": bool(ok)})
    return rate, total, report


def fmt(stat, unit, scale):
    if stat is None:
        return "no edges"
    return (f"{stat['mean'] * scale:9.3f} {unit} (sd {stat['sd'] * scale:.3f}, "
            f"{stat['min'] * scale:.3f}-{stat['max'] * scale:.3f}, n={stat['n']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="sigrok session file (.sr)")
    ap.add_argument("--map", help="signal=channel pairs, e.g. T1=0,T2=1,T3=2,T4=3,MARK=4,LED=5")
    args = ap.parse_args()
    mapping = None
    if args.map:
        mapping = {k: int(v.lstrip("D")) for k, v in (p.split("=") for p in args.map.split(","))}
    rate, total, report = check(args.capture, mapping)
    print(f"{args.capture}: {total} samples at {rate / 1e6:g} MHz = {total / rate:.2f} s")
    for r in report:
        print(f"{r['signal']:5} D{r['channel']}  {'PASS' if r['ok'] else 'FAIL'}")
        print(f"      period {fmt(r['period'], 'ms', 1e3)}   expected {r['expected_period'] * 1e3:g} ms")
        print(f"      width  {fmt(r['width'], 'us', 1e6)}   expected {r['expected_width'] * 1e6:g} us")
    ok = all(r["ok"] for r in report)
    print("PASS: the measurement chain is valid" if ok else "FAIL: check wiring, ground and sample rate")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
