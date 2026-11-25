import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pilot  # noqa: E402


class ReplicatesNeededTest(unittest.TestCase):
    def test_effect_of_one_sd_needs_17(self):
        # d = 1: two-sided t test, alpha 0.05, power 0.8 -> 16.7 per group
        self.assertEqual(pilot.replicates_needed(100.0, 10.0, 0.10, 0.8, 0.05), 17)

    def test_large_effect_needs_few(self):
        self.assertLessEqual(pilot.replicates_needed(100.0, 2.0, 0.10, 0.8, 0.05), 3)

    def test_no_spread(self):
        self.assertIsNone(pilot.replicates_needed(100.0, 0.0, 0.10, 0.8, 0.05))


if __name__ == "__main__":
    unittest.main()
