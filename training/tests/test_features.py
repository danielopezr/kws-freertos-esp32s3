import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kws import features as F  # noqa: E402


class TestFeatures(unittest.TestCase):
    def test_frame_count(self):
        self.assertEqual(F.N_FRAMES, 49)

    def test_shapes(self):
        rng = np.random.default_rng(0)
        one = rng.integers(-3000, 3000, 16_000).astype(np.int16)
        self.assertEqual(F.mfcc(one).shape, (49, 40))
        batch = rng.integers(-3000, 3000, (3, 16_000)).astype(np.int16)
        self.assertEqual(F.mfcc(batch).shape, (3, 49, 40))
        self.assertEqual(F.mfcc_batched(batch, batch=2).shape, (3, 49, 40))

    def test_int16_and_float_inputs_match(self):
        pcm = np.random.default_rng(1).integers(-20000, 20000, 16_000).astype(np.int16)
        np.testing.assert_allclose(F.mfcc(pcm), F.mfcc(pcm.astype(np.float32) / 32768.0), rtol=1e-5, atol=1e-4)

    def test_batch_matches_single(self):
        batch = np.random.default_rng(2).integers(-5000, 5000, (4, 16_000)).astype(np.int16)
        out = F.mfcc(batch)
        for i in range(4):
            np.testing.assert_allclose(out[i], F.mfcc(batch[i]), rtol=1e-5, atol=1e-4)

    def test_short_clip_is_zero_padded(self):
        short = np.ones(8_000, dtype=np.float32) * 0.1
        padded = np.concatenate([short, np.zeros(8_000, dtype=np.float32)])
        np.testing.assert_allclose(F.mfcc(short), F.mfcc(padded))

    def test_window_is_periodic_hann(self):
        w = F.hann_window(8)
        np.testing.assert_allclose(w, [0, 0.1464466, 0.5, 0.8535534, 1, 0.8535534, 0.5, 0.1464466], atol=1e-6)

    def test_filterbank(self):
        fb = F.mel_filterbank()
        self.assertEqual(fb.shape, (40, 513))
        self.assertTrue(np.all(fb.max(axis=1) > 0), "every filter must cover at least one bin")
        self.assertLessEqual(fb.max(), 1.0 + 1e-6)
        peaks = fb.argmax(axis=1)
        self.assertTrue(np.all(np.diff(peaks) >= 0), "filter centres must increase")
        top_bin = int(np.ceil(F.F_MAX * F.N_FFT / F.SAMPLE_RATE))
        self.assertEqual(fb[:, top_bin + 1:].sum(), 0.0, "nothing above f_max")

    def test_dct_is_orthonormal(self):
        d = F.dct_matrix()
        np.testing.assert_allclose(d @ d.T, np.eye(40), atol=1e-5)

    def test_tone_lands_in_the_right_mel_band(self):
        t = np.arange(16_000) / 16_000
        tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        x = tone[: F.FRAME_LEN] * F.hann_window()
        power = np.abs(np.fft.rfft(x, n=F.N_FFT)) ** 2
        band = int(np.argmax(F.mel_filterbank() @ power))
        centres = F.mel_to_hz(np.linspace(F.hz_to_mel(F.F_MIN), F.hz_to_mel(F.F_MAX), 42))[1:-1]
        self.assertEqual(band, int(np.argmin(np.abs(centres - 1000))))


if __name__ == "__main__":
    unittest.main()
