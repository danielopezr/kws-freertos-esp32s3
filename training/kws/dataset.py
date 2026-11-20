"""Speech Commands v2: download, split and feature cache.

Follows the standard TensorFlow keyword-spotting setup (Warden, 2018):
12 classes = "_silence_", "_unknown_" and 10 keywords; the official
validation_list.txt / testing_list.txt define the splits; silence and unknown
each add 10 % of the number of keyword clips in every split.
"""

import json
import math
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np

from . import features as F

DATA_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = ["_silence_", "_unknown_"] + KEYWORDS
SILENCE, UNKNOWN = 0, 1
BACKGROUND_DIR = "_background_noise_"
SILENCE_PCT = 10
UNKNOWN_PCT = 10

# Training-only augmentation (same spirit as the TF speech_commands recipe)
SHIFT_MS = 100
NOISE_PROB = 0.8
NOISE_MAX_VOLUME = 0.1


def download(data_dir):
    """Download and extract the dataset into data_dir (skips if already there)."""
    data_dir = Path(data_dir)
    if (data_dir / "testing_list.txt").exists():
        print(f"dataset already at {data_dir}")
        return data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "speech_commands_v0.02.tar.gz"
    if not archive.exists():
        print(f"downloading {DATA_URL} (~2.3 GB) ...")
        urllib.request.urlretrieve(DATA_URL, archive)
    print("extracting ...")
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")
    archive.unlink()
    return data_dir


def read_wav(path):
    """Mono 16-bit 16 kHz WAV -> float32 in [-1, 1)."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != F.SAMPLE_RATE:
            raise ValueError(f"unexpected format in {path}")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def _read_list(data_dir, name):
    return {line.strip() for line in (Path(data_dir) / name).read_text().splitlines() if line.strip()}


def split_files(data_dir):
    """Return {'train'|'validation'|'test': [(relative_path, word), ...]} using the official lists."""
    data_dir = Path(data_dir)
    val, test = _read_list(data_dir, "validation_list.txt"), _read_list(data_dir, "testing_list.txt")
    splits = {"train": [], "validation": [], "test": []}
    for word_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name != BACKGROUND_DIR):
        for wav in sorted(word_dir.glob("*.wav")):
            rel = f"{word_dir.name}/{wav.name}"
            split = "validation" if rel in val else "test" if rel in test else "train"
            splits[split].append((rel, word_dir.name))
    return splits


def build_items(files, rng):
    """Keyword clips + sampled unknown clips + silence placeholders: [(path_or_None, label)]."""
    known = [(p, LABELS.index(w)) for p, w in files if w in KEYWORDS]
    others = [p for p, w in files if w not in KEYWORDS]
    n_unknown = math.ceil(len(known) * UNKNOWN_PCT / 100)
    n_silence = math.ceil(len(known) * SILENCE_PCT / 100)
    unknown = [(others[i], UNKNOWN) for i in rng.choice(len(others), n_unknown, replace=False)]
    silence = [(None, SILENCE)] * n_silence
    items = known + unknown + silence
    order = rng.permutation(len(items))
    return [items[i] for i in order]


def load_background(data_dir):
    return [read_wav(p) for p in sorted((Path(data_dir) / BACKGROUND_DIR).glob("*.wav"))]


def _noise_crop(background, rng):
    clip = background[rng.integers(len(background))]
    start = rng.integers(0, len(clip) - F.CLIP_SAMPLES)
    return clip[start:start + F.CLIP_SAMPLES]


def _shift(x, samples):
    out = np.zeros_like(x)
    if samples > 0:
        out[samples:] = x[:-samples]
    elif samples < 0:
        out[:samples] = x[-samples:]
    else:
        out[:] = x
    return out


def render(item, data_dir, background, rng, augment):
    """Waveform (float32, 16 000 samples) for one item, with optional augmentation."""
    path, label = item
    if label == SILENCE:
        return (_noise_crop(background, rng) * rng.uniform(0.0, 1.0)).astype(np.float32)
    x = F.fit_length(read_wav(Path(data_dir) / path)).astype(np.float32)
    if augment:
        max_shift = F.SAMPLE_RATE * SHIFT_MS // 1000
        x = _shift(x, int(rng.integers(-max_shift, max_shift + 1)))
        if rng.random() < NOISE_PROB:
            x = x + _noise_crop(background, rng) * rng.uniform(0.0, NOISE_MAX_VOLUME)
        x = np.clip(x, -1.0, 1.0)
    return x.astype(np.float32)


def build_features(data_dir, cache_dir, seed=1234, augment_copies=1, chunk=2048):
    """Compute MFCC features for every split and save them to cache_dir/{split}.npz.

    The training split holds one clean copy of each item plus `augment_copies`
    augmented copies. Features are stored as float16 to halve memory.
    """
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    background = load_background(data_dir)
    splits = split_files(data_dir)
    counts = {}
    for split_index, (split, files) in enumerate(splits.items()):
        out_path = cache_dir / f"{split}.npz"
        rng = np.random.default_rng(seed + split_index)
        items = build_items(files, rng)
        passes = [False] + [True] * (augment_copies if split == "train" else 0)
        total = len(items) * len(passes)
        x = np.empty((total, F.N_FRAMES, F.N_MFCC), dtype=np.float16)
        y = np.empty(total, dtype=np.uint8)
        row = 0
        for augment in passes:
            for i in range(0, len(items), chunk):
                batch = items[i:i + chunk]
                audio = np.stack([render(it, data_dir, background, rng, augment) for it in batch])
                x[row:row + len(batch)] = F.mfcc(audio)
                y[row:row + len(batch)] = [label for _, label in batch]
                row += len(batch)
                print(f"\r{split}: {row}/{total}", end="", flush=True)
        print()
        np.savez(out_path, x=x, y=y)
        counts[split] = {LABELS[k]: int((y == k).sum()) for k in range(len(LABELS))}
    meta = {
        "labels": LABELS, "seed": seed, "augment_copies": augment_copies, "counts": counts,
        "features": {k: getattr(F, k) for k in ("SAMPLE_RATE", "FRAME_LEN", "HOP", "N_FFT", "N_MELS",
                                                 "N_MFCC", "F_MIN", "F_MAX", "LOG_EPS", "N_FRAMES")},
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_split(cache_dir, split):
    data = np.load(Path(cache_dir) / f"{split}.npz")
    return data["x"].astype(np.float32), data["y"].astype(np.int64)


def fit_norm(x_train):
    """Global mean and standard deviation of the training MFCCs (one scalar each)."""
    return {"mean": float(x_train.mean(dtype=np.float64)), "std": float(x_train.std(dtype=np.float64))}


def apply_norm(x, norm):
    return ((x - norm["mean"]) / norm["std"]).astype(np.float32)
