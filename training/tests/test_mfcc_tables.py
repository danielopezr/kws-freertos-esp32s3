import re
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_mfcc_tables as E  # noqa: E402
from kws import features as F  # noqa: E402


def parse_array(text, name):
    body = re.search(rf"{name}\[\d+\] = \{{(.*?)\}};", text, re.S).group(1)
    return np.array([float(v.strip().rstrip("f")) for v in body.replace("\n", " ").split(",") if v.strip()], dtype=np.float32)


class TestMfccTables(unittest.TestCase):
    def test_header_is_up_to_date(self):
        self.assertTrue(E.OUT.exists(), "run training/export_mfcc_tables.py")
        self.assertEqual(E.OUT.read_text(), E.render(),
                         "kws_mfcc_tables.h is stale: run training/export_mfcc_tables.py")

    def test_values_round_trip_to_float32(self):
        text = E.render()
        np.testing.assert_array_equal(parse_array(text, "kws_window"), F.hann_window())
        np.testing.assert_array_equal(parse_array(text, "kws_dct"), F.dct_matrix().reshape(-1))

    def test_sparse_filterbank_rebuilds_dense(self):
        text = E.render()
        start = parse_array(text, "kws_mel_start").astype(int)
        length = parse_array(text, "kws_mel_len").astype(int)
        offset = parse_array(text, "kws_mel_offset").astype(int)
        w = parse_array(text, "kws_mel_w")
        dense = np.zeros((F.N_MELS, F.N_FFT // 2 + 1), dtype=np.float32)
        for m in range(F.N_MELS):
            dense[m, start[m]:start[m] + length[m]] = w[offset[m]:offset[m] + length[m]]
        np.testing.assert_array_equal(dense, F.mel_filterbank())


if __name__ == "__main__":
    unittest.main()
