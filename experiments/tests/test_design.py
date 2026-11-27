import re
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
        text = re.sub(r"(?m)^(\s+)(best|alternative): \{.*\}$", r"\1\2: null", (DESIGNS / "D.yaml").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "D.yaml"
            path.write_text(text)
            with self.assertRaisesRegex(D.DesignError, "best"):
                D.load(path)

    def test_presets_expand(self):
        trts = D.treatments(D.load(DESIGNS / "D.yaml"))
        self.assertEqual(len(trts), 9)
        by = {s: [cfg for _, f, cfg in trts if f["strategy"] == s] for s in ("base", "best", "alternative")}
        self.assertTrue(all(c["affinity"] == "split" and c["prio"] == "equal" for c in by["best"]))
        self.assertTrue(all(c["affinity"] == "none" and c["prio"] == "lower" for c in by["alternative"]))
        self.assertEqual({c["model"] for c in by["best"]}, {"dnn", "cnn", "dscnn"})
        # best and alternative each differ from base in one factor only (affinity, priority)
        for s, factor in (("best", "affinity"), ("alternative", "prio")):
            for b, c in zip(sorted(by["base"], key=lambda c: c["model"]), sorted(by[s], key=lambda c: c["model"])):
                self.assertEqual({k for k in c if c[k] != b[k]}, {factor})

    def test_demo_is_refused_in_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "X.yaml"
            text = (DESIGNS / "A.yaml").read_text().replace("seq: full", "seq: full\n  demo: 'on'")
            path.write_text(text)
            with self.assertRaisesRegex(D.DesignError, "demo"):
                D.treatments(D.load(path))
        for name in "ABC":
            self.assertTrue(all(c["demo"] == "off" for _, _, c in D.treatments(D.load(DESIGNS / f"{name}.yaml"))))

    def test_invalid_level_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "X.yaml"
            path.write_text("name: X\nreplicates: 1\nseed: 1\nfixed: {model: cnn, mem: sram, alloc: static, "
                            "frag: 0, seq: full, load: 'off', prio: equal}\nfactors:\n  affinity: [same, both]\n")
            with self.assertRaisesRegex(D.DesignError, "both"):
                D.treatments(D.load(path))


if __name__ == "__main__":
    unittest.main()
