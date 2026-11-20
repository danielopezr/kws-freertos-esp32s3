import json
import math
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kws import dataset as D  # noqa: E402


def write_wav(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def fake_dataset(root, rng, per_word=20, words=("yes", "no", "stop", "bed", "cat", "tree")):
    """Tiny Speech-Commands-shaped tree: word folders, official lists, background noise."""
    val, test = [], []
    for word in words:
        for i in range(per_word):
            n = int(rng.integers(12_000, 16_001))  # some clips shorter than 1 s, like the real set
            write_wav(root / word / f"{i:04d}_nohash_0.wav", rng.integers(-8000, 8000, n))
            rel = f"{word}/{i:04d}_nohash_0.wav"
            (val if i % 10 == 0 else test if i % 10 == 1 else []).append(rel)
    (root / "validation_list.txt").write_text("\n".join(val) + "\n")
    (root / "testing_list.txt").write_text("\n".join(test) + "\n")
    for k in range(2):
        write_wav(root / D.BACKGROUND_DIR / f"noise{k}.wav", rng.integers(-3000, 3000, 48_000))


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "sc"
        cls.cache = Path(cls.tmp.name) / "features"
        fake_dataset(cls.root, np.random.default_rng(0))
        cls.meta = D.build_features(cls.root, cls.cache, seed=7, augment_copies=1, chunk=16)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_labels(self):
        self.assertEqual(len(D.LABELS), 12)
        self.assertEqual(D.LABELS[:2], ["_silence_", "_unknown_"])

    def test_official_lists_define_splits(self):
        splits = D.split_files(self.root)
        self.assertEqual(len(splits["validation"]), 6 * 2)
        self.assertEqual(len(splits["test"]), 6 * 2)
        self.assertEqual(len(splits["train"]), 6 * 16)
        self.assertFalse(any(D.BACKGROUND_DIR in p for s in splits.values() for p, _ in s))

    def test_silence_and_unknown_are_ten_percent(self):
        for split, counts in self.meta["counts"].items():
            known = sum(counts[w] for w in D.KEYWORDS)
            copies = 2 if split == "train" else 1
            expected = copies * math.ceil(known / copies * 0.10)
            self.assertEqual(counts["_silence_"], expected, split)
            self.assertEqual(counts["_unknown_"], expected, split)
            self.assertTrue(all(counts[w] == 0 for w in ("up", "down", "left", "right", "on", "off", "go")))

    def test_train_has_clean_plus_augmented_copy(self):
        x, y = D.load_split(self.cache, "train")
        n = len(y) // 2
        self.assertEqual(len(y), 2 * n)
        np.testing.assert_array_equal(y[:n], y[n:])
        self.assertEqual(x.shape[1:], (49, 40))
        keyword_rows = np.isin(y[:n], [D.LABELS.index(w) for w in D.KEYWORDS])
        self.assertFalse(np.allclose(x[:n][keyword_rows], x[n:][keyword_rows]), "augmented copy must differ")

    def test_features_are_finite(self):
        for split in ("train", "validation", "test"):
            x, _ = D.load_split(self.cache, split)
            self.assertTrue(np.isfinite(x).all(), split)

    def test_deterministic(self):
        other = Path(self.tmp.name) / "features2"
        D.build_features(self.root, other, seed=7, augment_copies=1, chunk=16)
        for split in ("train", "validation", "test"):
            a, ya = D.load_split(self.cache, split)
            b, yb = D.load_split(other, split)
            np.testing.assert_array_equal(ya, yb)
            np.testing.assert_array_equal(a, b)

    def test_meta_written(self):
        meta = json.loads((self.cache / "meta.json").read_text())
        self.assertEqual(meta["features"]["N_FRAMES"], 49)
        self.assertEqual(meta["labels"], D.LABELS)

    def test_norm_roundtrip(self):
        x, _ = D.load_split(self.cache, "train")
        norm = D.fit_norm(x)
        z = D.apply_norm(x, norm)
        self.assertAlmostEqual(float(z.mean()), 0.0, places=3)
        self.assertAlmostEqual(float(z.std()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
