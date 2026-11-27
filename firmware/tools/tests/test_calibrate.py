import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate import BLOCK, Timeline, hit_latencies, load_labels, run_segment  # noqa: E402

LABELS = [
    {"start": 16000, "voiced_start": 17000, "voiced_end": 26240, "label_id": 5},
    {"start": 48000, "voiced_start": 50000, "voiced_end": 63040, "label_id": 4},
    {"start": 80000, "voiced_start": 81000, "voiced_end": 91840, "label_id": 1},
]


class TimelineTest(unittest.TestCase):
    def test_xorshift_matches_firmware(self):
        # pipeline.cc timeline_build, seed 1: the first xorshift32 output is 270369
        tl = Timeline(LABELS, 1)
        self.assertEqual(tl.len[0], 270369 % 1600)
        self.assertEqual(tl.cum[1], tl.len[0] + tl.len[1])

    def test_total_matches_recorded_runs(self):
        # seq_inserted reported by the board in experiment A (seed -> total silence inserted)
        labels = load_labels()
        for seed, inserted in ((1325260861, 123446), (1867650674, 108265), (491023260, 116521)):
            self.assertEqual(Timeline(labels, seed).cum[-1], inserted)

    def test_stream_seq_round_trip(self):
        tl = Timeline(LABELS, 7)
        for q in (20000, 26240, 63040, 91840):
            j = max(i for i, a in enumerate(tl.at) if a < q)
            s = q + tl.cum[j]
            self.assertEqual(tl.to_seq(s), q)
            self.assertEqual(tl.to_stream(q, s), s)

    def test_inside_silence_maps_to_insertion_point(self):
        tl = Timeline(LABELS, 7)
        s = tl.stream_at[1] + tl.len[1] // 2
        self.assertEqual(tl.to_seq(s), tl.at[1])
        self.assertIsNone(tl.to_stream(tl.at[1], s))


class SegmentTest(unittest.TestCase):
    def test_longest_stretch_skips_probe_test(self):
        rate = 250_000
        probe = np.arange(0, 4.0, 0.02)            # probe test before the reset
        run = 6.0 + np.arange(0, 30.0, 0.02)       # pipeline after boot
        rising = (np.concatenate((probe, run)) * rate).astype(np.int64)
        i0, i1 = run_segment(rising, rate)
        self.assertEqual(rising[i0], int(6.0 * rate))
        self.assertEqual(i1, len(rising) - 1)


class HitLatencyTest(unittest.TestCase):
    def test_first_right_word_in_window(self):
        dets = [{"sample": 27840, "label_id": 5}, {"sample": 28000, "label_id": 5},
                {"sample": 64640, "label_id": 3}, {"sample": 93000, "label_id": 1}]
        hits = hit_latencies(dets, LABELS)
        self.assertEqual(list(hits), [0])
        self.assertAlmostEqual(hits[0][0], (27840 - 26240) / 16)
        self.assertIs(hits[0][1], dets[0])
        self.assertEqual(BLOCK, 320)


if __name__ == "__main__":
    unittest.main()
