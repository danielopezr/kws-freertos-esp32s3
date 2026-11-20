import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import design as D  # noqa: E402

DESIGNS = Path(__file__).resolve().parents[1] / "designs"


class TestDesigns(unittest.TestCase):
    def test_sizes_match_the_proposal(self):
        for name, trts, runs in (("A", 18, 90), ("B", 12, 60), ("C", 9, 9)):
            d = D.load(DESIGNS / f"{name}.yaml")
            self.assertEqual(len(D.treatments(d)), trts, name)
            self.assertEqual(len(D.plan(d)), runs, name)

    def test_every_treatment_gets_every_replicate(self):
        d = D.load(DESIGNS / "A.yaml")
        p = D.plan(d)
        self.assertEqual(Counter(r["treatment"] for r in p), {t: 5 for t in range(1, 19)})
        self.assertEqual(len({r["run_id"] for r in p}), len(p))
        self.assertEqual(sorted(r["order"] for r in p), list(range(1, 91)))

    def test_fixed_settings_and_yaml_booleans(self):
        d = D.load(DESIGNS / "A.yaml")
        for _, factors, cfg in D.treatments(d):
            self.assertEqual((cfg["model"], cfg["mem"], cfg["alloc"]), ("cnn", "sram", "static"))
            self.assertIn(cfg["load"], ("on", "off"))
            self.assertEqual(cfg["load"], factors["load"])

    def test_order_is_random_but_reproducible(self):
        d = D.load(DESIGNS / "B.yaml")
        a, b = D.plan(d), D.plan(d)
        self.assertEqual(a, b)
        self.assertNotEqual([r["treatment"] for r in a], sorted(r["treatment"] for r in a))
        d2 = dict(d, seed=d["seed"] + 1)
        self.assertNotEqual([r["run_id"] for r in a], [r["run_id"] for r in D.plan(d2)])
        self.assertNotEqual(D.fingerprint(d), D.fingerprint(d2))

    def test_seeds_differ_between_runs(self):
        p = D.plan(D.load(DESIGNS / "A.yaml"))
        self.assertEqual(len({r["seed"] for r in p}), len(p))

    def test_d_refuses_undefined_strategies(self):
        with self.assertRaisesRegex(D.DesignError, "best"):
            D.load(DESIGNS / "D.yaml")

    def test_presets_expand(self):
        text = (DESIGNS / "D.yaml").read_text().replace(
            "best: null", 'best: {affinity: split, prio: lower, load: "off", mem: sram, alloc: static}').replace(
            "alternative: null", 'alternative: {affinity: none, prio: lower, load: "off", mem: psram, alloc: static}')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "D.yaml"
            path.write_text(text)
            d = D.load(path)
        trts = D.treatments(d)
        self.assertEqual(len(trts), 9)
        best = [cfg for _, f, cfg in trts if f["strategy"] == "best"]
        self.assertTrue(all(c["affinity"] == "split" and c["prio"] == "lower" for c in best))
        self.assertEqual({c["model"] for c in best}, {"dnn", "cnn", "dscnn"})

    def test_invalid_level_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "X.yaml"
            path.write_text("name: X\nreplicates: 1\nseed: 1\nfixed: {model: cnn, mem: sram, alloc: static, "
                            "frag: 0, seq: full, load: 'off', prio: equal}\nfactors:\n  affinity: [same, both]\n")
            with self.assertRaisesRegex(D.DesignError, "both"):
                D.treatments(D.load(path))


if __name__ == "__main__":
    unittest.main()
