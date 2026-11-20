import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scoring as S  # noqa: E402

# Three 1 s clips 2 s apart: "yes" (2), unknown (1), "stop" (10)
LABELS = [
    {"label_id": 2, "start": 16_000, "end": 32_000, "voiced_start": 20_000, "voiced_end": 26_000},
    {"label_id": 1, "start": 48_000, "end": 64_000, "voiced_start": 50_000, "voiced_end": 56_000},
    {"label_id": 10, "start": 80_000, "end": 96_000, "voiced_start": 84_000, "voiced_end": 90_000},
]


def det(sample, label):
    return {"sample": sample, "label_id": label, "score": 230}


class TestScoring(unittest.TestCase):
    def test_windows_end_one_second_after_voice_or_at_next_clip(self):
        wins = S.windows(LABELS)
        self.assertEqual(wins[0], (20_000, 42_000, 2))  # 26 000 + 16 000
        self.assertEqual(wins[2], (84_000, 106_000, 10))
        short = S.windows(LABELS, max_latency_ms=5000)
        self.assertEqual(short[0][1], 48_000)  # cut at the next clip

    def test_perfect_run(self):
        m = S.score([det(26_000 + 3_200, 2), det(90_000 + 1_600, 10)], LABELS)
        self.assertEqual((m["hits"], m["misses"], m["false_alarms"]), (2, 0, 0))
        self.assertEqual((m["recall"], m["precision"], m["f1"]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(m["det_latency_ms"], (200 + 100) / 2)

    def test_errors_are_classified(self):
        detections = [
            det(27_600, 2),   # hit, 100 ms
            det(30_000, 2),   # duplicate in the same window
            det(52_000, 5),   # during the unknown word: false alarm
            det(70_000, 4),   # between words: false alarm
            det(91_000, 3),   # wrong word for "stop": substitution + false alarm
        ]
        m = S.score(detections, LABELS)
        self.assertEqual(m["keywords"], 2)
        self.assertEqual((m["hits"], m["substitutions"], m["misses"]), (1, 1, 0))
        self.assertEqual((m["false_alarms"], m["duplicates"]), (3, 1))
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["precision"], 0.25)
        self.assertAlmostEqual(m["f1"], 2 * 0.25 * 0.5 / 0.75)
        self.assertAlmostEqual(m["det_latency_ms"], 100.0)

    def test_early_detection_has_negative_latency(self):
        m = S.score([det(25_200, 2)], LABELS)  # 50 ms before the estimated end of the word
        self.assertAlmostEqual(m["det_latency_ms"], -50.0)

    def test_late_detection_is_a_miss_and_false_alarm(self):
        m = S.score([det(26_000 + 17_000, 2)], LABELS)  # 1.06 s after the word
        self.assertEqual((m["hits"], m["misses"], m["false_alarms"]), (0, 2, 1))

    def test_no_detections(self):
        m = S.score([], LABELS)
        self.assertEqual((m["hits"], m["misses"]), (0, 2))
        self.assertTrue(math.isnan(m["precision"]) and math.isnan(m["det_latency_ms"]))
        self.assertEqual(m["f1"], 0.0)

    def test_real_labels_load(self):
        labels = S.load_labels()
        self.assertEqual(len(labels), 150)
        self.assertEqual(sum(r["label_id"] >= 2 for r in labels), 100)


if __name__ == "__main__":
    unittest.main()
