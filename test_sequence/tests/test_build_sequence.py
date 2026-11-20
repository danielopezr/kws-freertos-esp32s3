import csv
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_sequence as B  # noqa: E402
from kws import dataset as D  # noqa: E402

UNKNOWN_WORDS = ("bed", "cat", "tree")


def write_wav(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def fake_dataset(root, rng, per_word=8):
    """Every clip in the test list, one speaker per clip; a burst of 'speech' inside quiet noise."""
    test = []
    for w, word in enumerate(list(D.KEYWORDS) + list(UNKNOWN_WORDS)):
        for i in range(per_word):
            n = int(rng.integers(12_000, 16_001))
            x = rng.normal(0, 20, n)
            x[4000:8000] += 8000 * np.sin(2 * np.pi * 440 * np.arange(4000) / 16_000)
            rel = f"{word}/{w:02d}{i:02d}abcd_nohash_0.wav"
            write_wav(root / rel, np.clip(x, -32768, 32767))
            test.append(rel)
    (root / "validation_list.txt").write_text("\n")
    (root / "testing_list.txt").write_text("\n".join(test) + "\n")


class TestBuildSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "sc"
        fake_dataset(cls.root, np.random.default_rng(0))
        cls.kw = dict(seed=5, per_keyword=3, n_unknown=4, gap_ms=1000)
        cls.pcm, cls.rows = B.build(cls.root, **cls.kw)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_counts_and_length(self):
        self.assertEqual(len(self.rows), 10 * 3 + 4)
        for w in D.KEYWORDS:
            self.assertEqual(sum(r["label"] == w for r in self.rows), 3)
        self.assertEqual(sum(r["label"] == "_unknown_" for r in self.rows), 4)
        self.assertEqual(len(self.pcm), len(self.rows) * 2 * 16_000)

    def test_unknown_spread_over_words(self):
        per_word = [sum(r["word"] == w for r in self.rows) for w in UNKNOWN_WORDS]
        self.assertEqual(sorted(per_word), [1, 1, 2])

    def test_one_speaker_per_utterance(self):
        speakers = [B.speaker(r["source"]) for r in self.rows]
        self.assertEqual(len(speakers), len(set(speakers)))

    def test_only_test_split(self):
        test = set((self.root / "testing_list.txt").read_text().split())
        self.assertTrue(all(r["source"] in test for r in self.rows))

    def test_layout(self):
        for r in self.rows:
            slot = r["index"] * 32_000
            self.assertEqual(r["start"], slot + 16_000)
            self.assertLessEqual(r["end"], slot + 32_000)
            self.assertTrue(r["start"] <= r["voiced_start"] < r["voiced_end"] <= r["end"])

    def test_audio_is_the_clips(self):
        r = self.rows[0]
        clip = (D.read_wav(self.root / r["source"]) * 32768).astype(np.int32)
        seg = self.pcm[r["start"]:r["end"]].astype(np.int32)
        noise = (seg - clip).astype(np.float64)  # only the -60 dBFS floor (~33 LSB RMS) is added
        self.assertAlmostEqual(np.sqrt((noise ** 2).mean()), 32768 * 1e-3, delta=3)

    def test_floor_is_never_digital_silence(self):
        gap = self.pcm[:16_000].astype(np.float64)
        rms_dbfs = 20 * np.log10(np.sqrt((gap ** 2).mean()) / 32768)
        self.assertAlmostEqual(rms_dbfs, -60.0, delta=1.0)

    def test_voiced_range_finds_burst(self):
        x = np.zeros(16_000, dtype=np.float32)
        x[3200:8000] = 0.3 * np.sin(2 * np.pi * 440 * np.arange(4800) / 16_000)
        self.assertEqual(B.voiced_range(x), (3200, 8000))
        self.assertEqual(B.voiced_range(np.zeros(16_000)), (0, 16_000))

    def test_voiced_range_ignores_noise_floor_and_clicks(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.02, 16_000)  # noisy recording: word only ~20 dB above the floor
        x[160:320] += 0.15  # click far from the word, quieter than it
        x[6400:9600] += 0.2 * np.sin(2 * np.pi * 300 * np.arange(3200) / 16_000)
        x[9600:10560] = rng.normal(0, 0.02, 960)  # 60 ms stop closure inside the word
        x[10560:11200] += 0.1 * np.sin(2 * np.pi * 300 * np.arange(640) / 16_000)
        self.assertEqual(B.voiced_range(x), (6400, 11200))

    def test_deterministic_and_seed_dependent(self):
        pcm2, rows2 = B.build(self.root, **self.kw)
        np.testing.assert_array_equal(self.pcm, pcm2)
        self.assertEqual(self.rows, rows2)
        _, rows3 = B.build(self.root, **{**self.kw, "seed": 6})
        self.assertNotEqual([r["source"] for r in self.rows], [r["source"] for r in rows3])

    def test_not_enough_speakers_fails(self):
        with self.assertRaises(ValueError):
            B.build(self.root, seed=1, per_keyword=9, n_unknown=1)

    def test_partition_fits_default_sequence(self):
        size = B.partition_size()
        self.assertEqual(size, 0xA00000)
        self.assertLessEqual(150 * 2 * 16_000 * 2, size)

    def test_outputs(self):
        out = Path(self.tmp.name) / "out"
        m = B.write_outputs(out, self.pcm, self.rows, self.kw, wav=True)
        self.assertEqual((out / "sequence.raw").stat().st_size, 2 * len(self.pcm))
        with wave.open(str(out / "sequence.wav")) as w:
            self.assertEqual((w.getframerate(), w.getnframes()), (16_000, len(self.pcm)))
        with open(out / "labels.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([int(r["voiced_end"]) for r in rows], [r["voiced_end"] for r in self.rows])
        header = (out / "kws_sequence_labels.h").read_text()
        self.assertIn(f"#define KWS_SEQ_N_ITEMS {len(self.rows)}", header)
        self.assertEqual(header.count("u, "), 4 * len(self.rows))
        self.assertEqual(json.loads((out / "manifest.json").read_text())["sha256"], m["sha256"])
        self.assertEqual(m["speakers"], len(self.rows))

    def test_relabel_reproduces_build(self):
        out = Path(self.tmp.name) / "relabel"
        B.write_outputs(out, self.pcm, [dict(r, voiced_start=0, voiced_end=0) for r in self.rows], self.kw, wav=False)
        B.relabel(out)
        with open(out / "labels.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(int(r["voiced_start"]), int(r["voiced_end"])) for r in rows],
                         [(r["voiced_start"], r["voiced_end"]) for r in self.rows])


if __name__ == "__main__":
    unittest.main()
