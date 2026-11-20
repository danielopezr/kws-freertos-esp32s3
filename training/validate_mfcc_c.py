"""Compare the C MFCC front end (firmware/components/kws_mfcc) with kws/features.py.

    python validate_mfcc_c.py --exe path/to/host_test[.exe] [--quant-report artifacts/cnn/quant_report.json]

Feeds a fixed set of synthetic clips (silence, near-silent noise, noise, tones,
chirps, clipped square waves, impulses, speech-like harmonic bursts) plus any WAVs
from --wav-dir through both implementations and reports:

  * max / mean absolute MFCC error, overall and per signal type
  * agreement of the INT8 model input after normalisation and quantisation
    (this is what the network actually sees)

Pass criteria: every INT8 input value within 1 step and >= 99.9 % identical.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from kws import features as F

# Defaults from the first Colab run (identical for the three models)
DEFAULT_QUANT = {"norm": {"mean": -0.6560660446818979, "std": 6.152801217507387},
                 "input": {"scale": 0.08214017003774643, "zero_point": 44}}


def synthetic_clips(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(F.CLIP_SAMPLES) / F.SAMPLE_RATE
    clips = []

    def add(kind, x):
        clips.append((kind, np.clip(np.round(x), -32768, 32767).astype(np.int16)))

    add("silence", np.zeros(F.CLIP_SAMPLES))
    for amp in (1, 3, 10):
        add("near-silent noise", rng.normal(0, amp, F.CLIP_SAMPLES))
    for amp in (100, 1000, 8000, 30000):
        add("noise", rng.normal(0, amp, F.CLIP_SAMPLES))
    for f in (60, 250, 440, 1000, 2500, 3900, 6000):
        add("tone", 10000 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)))
    for _ in range(4):
        fs = rng.uniform(80, 4000, 5)
        add("multitone", sum(3000 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3)) for f in fs))
    for f0, f1 in ((20, 8000), (300, 3400), (4000, 100)):
        k = (f1 - f0) / t[-1]
        add("chirp", 12000 * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2)))
    for f in (150, 900):
        add("clipped square", 40000 * np.sign(np.sin(2 * np.pi * f * t)))
    for period in (160, 1600):
        x = np.zeros(F.CLIP_SAMPLES)
        x[::period] = 25000
        add("impulses", x)
    for _ in range(12):  # voiced-speech-like: harmonics of a varying pitch, formant-shaped, in bursts
        f0 = rng.uniform(90, 250) * (1 + 0.1 * np.sin(2 * np.pi * rng.uniform(1, 4) * t))
        phase = 2 * np.pi * np.cumsum(f0) / F.SAMPLE_RATE
        formants = rng.uniform([300, 900, 2200], [900, 2200, 3500])
        x = np.zeros(F.CLIP_SAMPLES)
        for h in range(1, 30):
            fh = h * f0.mean()
            gain = sum(np.exp(-((fh - fm) / 150) ** 2) for fm in formants) + 0.02
            x += gain * np.sin(h * phase)
        start = rng.integers(0, 6000)
        env = np.zeros(F.CLIP_SAMPLES)
        env[start:start + rng.integers(4000, 9000)] = 1.0
        env = np.convolve(env, np.hanning(800) / np.hanning(800).sum(), mode="same")
        x = x / np.abs(x).max() * rng.uniform(2000, 20000) * env + rng.normal(0, rng.uniform(5, 200), F.CLIP_SAMPLES)
        add("speech-like", x)
    return clips


def wav_clips(wav_dir):
    from kws.dataset import read_wav
    return [("wav", np.round(F.fit_length(read_wav(p)) * 32768).clip(-32768, 32767).astype(np.int16))
            for p in sorted(Path(wav_dir).rglob("*.wav"))]


def quantize(x, q):
    z = (x - q["norm"]["mean"]) / q["norm"]["std"]
    return np.clip(np.round(z / q["input"]["scale"] + q["input"]["zero_point"]), -128, 127).astype(np.int16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True, help="compiled host_test")
    ap.add_argument("--quant-report", help="quant_report.json with norm and input scale (defaults: first run)")
    ap.add_argument("--wav-dir", help="optional folder of real 16 kHz WAVs to add")
    args = ap.parse_args()

    q = json.loads(Path(args.quant_report).read_text()) if args.quant_report else DEFAULT_QUANT
    clips = synthetic_clips() + (wav_clips(args.wav_dir) if args.wav_dir else [])
    kinds = [k for k, _ in clips]
    audio = np.stack([c for _, c in clips])

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "clips.raw", Path(tmp) / "mfcc.raw"
        audio.astype("<i2").tofile(src)
        run = subprocess.run([args.exe, str(src), str(dst)], capture_output=True, text=True, check=True)
        c_out = np.fromfile(dst, dtype="<f4").reshape(len(clips), F.N_FRAMES, F.N_MFCC)
    py_out = F.mfcc(audio)

    err = np.abs(c_out - py_out)
    dq = np.abs(quantize(c_out, q) - quantize(py_out, q))
    print(run.stdout.strip())
    print(f"clips: {len(clips)}   values: {err.size}")
    print(f"{'type':20}{'max abs err':>12}{'mean abs err':>14}{'int8 equal':>12}{'max dq':>8}")
    for kind in dict.fromkeys(kinds):
        sel = np.array([k == kind for k in kinds])
        print(f"{kind:20}{err[sel].max():>12.2e}{err[sel].mean():>14.2e}{(dq[sel] == 0).mean():>12.4%}{dq[sel].max():>8}")
    equal = (dq == 0).mean()
    print(f"{'ALL':20}{err.max():>12.2e}{err.mean():>14.2e}{equal:>12.4%}{dq.max():>8}")
    print(f"(one INT8 input step = {q['input']['scale'] * q['norm']['std']:.3f} in raw MFCC units)")

    ok = dq.max() <= 1 and equal >= 0.999
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
