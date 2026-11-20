"""End to end: simulator -> runner -> analysis, on a small design."""
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))
import analyze as A  # noqa: E402
import data  # noqa: E402
import runner as R  # noqa: E402
import sim_board  # noqa: E402

DESIGN = """name: A
replicates: 3
seed: 9
fixed: {model: cnn, mem: sram, alloc: static, frag: "0", seq: full}
factors:
  affinity: [same, split]
  prio: [lower, equal, higher]
  load: ["off", "on"]
"""


class TestAnalyzeSimulated(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        (tmp / "A.yaml").write_text(DESIGN)
        sim = sim_board.Simulator(seed=2).start()
        try:
            R.run_design(tmp / "A.yaml", f"socket://127.0.0.1:{sim.port}", tmp / "results",
                         timeouts={"boot": 2, "ok": 1, "done": 2}, echo=lambda m: None)
        finally:
            sim.stop()
        cls.out, cls.df, cls.factors = A.analyse(tmp / "results", tmp / "report")
        cls.report = (cls.out / "report.md").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_row_per_run_with_scores(self):
        self.assertEqual(len(self.df), 36)
        self.assertEqual(self.factors, ["affinity", "prio", "load"])
        for col in ("det_latency_ms", "f1", "loss_pct", "hits", "false_alarms"):
            self.assertTrue(self.df[col].notna().all(), col)
        self.assertEqual(list(self.df["prio"].cat.categories), ["lower", "equal", "higher"])

    def test_report_sections_and_files(self):
        for text in ("# Experiment A", "Factorial ANOVA", "Assumptions", "Tukey HSD", "logistic regression"):
            self.assertIn(text, self.report)
        for name in ("runs.csv", "det_latency_ms_anova.csv", "loss_pct_box.png", "f1_interaction.png"):
            self.assertTrue((self.out / name).exists(), name)

    def test_simulated_effects_are_found(self):
        anova = __import__("pandas").read_csv(self.out / "loss_pct_anova.csv", index_col=0)
        self.assertTrue(anova.loc["affinity", "significant"])
        lat = __import__("pandas").read_csv(self.out / "det_latency_ms_anova.csv", index_col=0)
        self.assertTrue(lat.loc["load", "significant"])

    def test_failed_runs_are_excluded(self):
        self.assertEqual(len(data.ok_runs(self.df)), 36)
        self.assertIn("36 ok", self.report)


if __name__ == "__main__":
    unittest.main()
