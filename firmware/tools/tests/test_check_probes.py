import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_probes as C  # noqa: E402

RATE = 250_000


def square(n, rate, period_s, width_s, phase_s=0.0, jitter_s=0.0, rng=None):
    """0/1 array with pulses of width_s every period_s (optional start-time jitter)."""
    x = np.zeros(n, dtype=np.uint8)
    t = phase_s
    while t < n / rate:
        start = t + (rng.normal(0, jitter_s) if jitter_s else 0.0)
        a, b = int(round(start * rate)), int(round((start + width_s) * rate))
        x[max(a, 0):max(min(b, n), 0)] = 1
        t += period_s
    return x


def write_sr(path, samples, rate, chunk=40_000, rate_text=None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("version", "2")
        probes = "\n".join(f"probe{i + 1}=D{i}" for i in range(8))
        z.writestr("metadata", f"[global]\nsigrok version=0.5.2\n\n[device 1]\ncapturefile=logic-1\n"
                               f"total probes=8\nsamplerate={rate_text or f'{rate // 1000} kHz'}\n"
                               f"total analog=0\n{probes}\nunitsize=1\n")
        for i in range(0, len(samples), chunk):
            z.writestr(f"logic-1-{i // chunk + 1}", samples[i:i + chunk].tobytes())


def capture(seconds=3.6, overrides=None, jitter_s=20e-6, seed=0):
    """Synthetic capture of the probe test as the firmware should produce it."""
    rng = np.random.default_rng(seed)
    exp = C.expectations()
    exp.update(overrides or {})
    n = int(seconds * RATE)
    data = np.zeros(n, dtype=np.uint8)
    for name, ch in C.DEFAULT_MAP.items():
        period, width = exp[name]
        data |= square(n, RATE, period, width, phase_s=0.003 * ch + 0.0007, jitter_s=jitter_s, rng=rng) << ch
    return data


class TestCheckProbes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "probes.sr"

    def tearDown(self):
        self.tmp.cleanup()

    def test_expectations_come_from_probes_h(self):
        exp = C.expectations()
        self.assertEqual(exp["T1"], (0.020, 0.001))
        self.assertEqual(exp["T3"], (0.100, 0.010))
        self.assertEqual(exp["MARK"], (1.0, 0.0001))
        self.assertEqual(exp["LED"], (1.0, 0.5))

    def test_samplerate_text(self):
        self.assertEqual(C.parse_samplerate("250 kHz"), 250e3)
        self.assertEqual(C.parse_samplerate("24 MHz"), 24e6)
        self.assertEqual(C.parse_samplerate("500 Hz"), 500)
        with self.assertRaises(ValueError):
            C.parse_samplerate("fast")

    def test_good_capture_passes(self):
        write_sr(self.path, capture(), RATE)
        rate, total, report = C.check(self.path)
        self.assertEqual((rate, total), (RATE, int(3.6 * RATE)))
        self.assertTrue(all(r["ok"] for r in report), report)
        t1 = next(r for r in report if r["signal"] == "T1")
        self.assertAlmostEqual(t1["period"]["mean"], 0.020, delta=2 / RATE)
        self.assertAlmostEqual(t1["width"]["mean"], 0.001, delta=2 / RATE)

    def test_edges_across_chunk_boundaries(self):
        data = capture()
        write_sr(self.path, data, RATE, chunk=40_000)
        a = C.check(self.path)[2]
        write_sr(self.path, data, RATE, chunk=7_919)  # boundaries fall inside pulses
        b = C.check(self.path)[2]
        self.assertEqual([r["period"] for r in a], [r["period"] for r in b])
        self.assertEqual([r["width"] for r in a], [r["width"] for r in b])

    def test_wrong_width_fails_only_that_channel(self):
        write_sr(self.path, capture(overrides={"T2": (0.020, 0.0025)}), RATE)
        report = {r["signal"]: r["ok"] for r in C.check(self.path)[2]}
        self.assertFalse(report["T2"])
        self.assertTrue(all(ok for name, ok in report.items() if name != "T2"))

    def test_wrong_period_fails(self):
        write_sr(self.path, capture(overrides={"T3": (0.105, 0.010)}), RATE)
        self.assertFalse({r["signal"]: r["ok"] for r in C.check(self.path)[2]}["T3"])

    def test_disconnected_channel_fails(self):
        data = capture() & ~np.uint8(1 << 3)  # T4 probe not connected
        write_sr(self.path, data, RATE)
        t4 = next(r for r in C.check(self.path)[2] if r["signal"] == "T4")
        self.assertFalse(t4["ok"])
        self.assertIsNone(t4["period"])

    def test_too_short_capture_fails_slow_channels(self):
        write_sr(self.path, capture(seconds=1.2), RATE)
        report = {r["signal"]: r["ok"] for r in C.check(self.path)[2]}
        self.assertTrue(report["T1"])
        self.assertFalse(report["MARK"])  # needs 3 periods = 3 s

    def test_pulses_ignore_partial_edges(self):
        rising = np.array([10, 30, 50])
        falling = np.array([5, 15, 35])  # starts high, ends high
        periods, widths = C.pulses(rising, falling)
        np.testing.assert_array_equal(periods, [20, 20])
        np.testing.assert_array_equal(widths, [5, 5])


if __name__ == "__main__":
    unittest.main()
