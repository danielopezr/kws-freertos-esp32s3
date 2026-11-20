import csv
import io
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import design as D  # noqa: E402
import runner as R  # noqa: E402
import sim_board as S  # noqa: E402

DESIGN = """name: T
replicates: 2
seed: 7
fixed: {model: cnn, mem: sram, alloc: static, frag: "0", seq: full, load: "off"}
factors:
  affinity: [same, split]
  prio: [lower, higher]
"""
FAST = {"boot": 2.0, "ok": 1.0, "done": 1.5}


def read_index(out):
    with open(Path(out) / "index.csv", newline="") as f:
        return list(csv.DictReader(f))


class TestRunnerWithSimulator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.design = self.dir / "T.yaml"
        self.design.write_text(DESIGN)
        self.out = self.dir / "results"
        self.sims = []

    def tearDown(self):
        for sim in self.sims:
            sim.stop()
        self.tmp.cleanup()

    def sim(self, **kw):
        sim = S.Simulator(**kw).start()
        self.sims.append(sim)
        return f"socket://127.0.0.1:{sim.port}"

    def run_design(self, port, **kw):
        log = io.StringIO()
        rows = R.run_design(self.design, port, self.out, timeouts=FAST, echo=lambda m: log.write(m + "\n"), **kw)
        return rows, log.getvalue()

    def test_all_runs_ok(self):
        rows, _ = self.run_design(self.sim())
        self.assertEqual(len(rows), 8)
        index = read_index(self.out)
        self.assertEqual([r["status"] for r in index], ["ok"] * 8)
        self.assertEqual(sorted(int(r["order"]) for r in index), list(range(1, 9)))
        for r in index:
            self.assertGreater(int(r["detections"]), 50)
            self.assertGreater(int(r["inf_mean_us"]), 0)
            self.assertNotEqual(r["T3_cpu_pct"], "")
        rec = json.loads((self.out / "runs" / f"{index[0]['run_id']}.json").read_text())
        self.assertEqual(rec["cfg"]["model"], "cnn")
        self.assertEqual(rec["seed"], int(index[0]["seed"]))
        log = (self.out / "runs" / f"{index[0]['run_id']}.log").read_text()
        self.assertIn("> CFG model=cnn", log)
        self.assertIn("< DONE", log)

    def test_same_seed_same_results(self):
        self.run_design(self.sim())
        first = {r["run_id"]: r["inf_mean_us"] for r in read_index(self.out)}
        for p in (self.out / "runs").glob("*.json"):
            p.unlink()
        self.run_design(self.sim())
        self.assertEqual({r["run_id"]: r["inf_mean_us"] for r in read_index(self.out)}, first)

    def test_failures_are_recorded_and_retried(self):
        self.run_design(self.sim(fault_rate=0.6, seed=3), max_consecutive_failures=99)
        index = read_index(self.out)
        failed = {r["run_id"] for r in index if r["status"] != "ok"}
        self.assertEqual(len(index), 8)
        self.assertTrue(failed, "fault injection should break some runs")
        self.assertTrue({r["status"] for r in index} - {"ok"} <= {"timeout", "reset"})

        _, log = self.run_design(self.sim(), retry_failed=True)
        self.assertIn(f"{8 - len(failed)} already recorded", log)
        index = read_index(self.out)
        self.assertEqual([r["status"] for r in index], ["ok"] * 8)
        self.assertEqual({r["run_id"] for r in index if r["attempts"] == "2"}, failed)

    def test_resume_skips_recorded_runs(self):
        port = self.sim()
        self.run_design(port, limit=3)
        self.assertEqual(len(read_index(self.out)), 3)
        _, log = self.run_design(port)
        self.assertIn("3 already recorded, 5 to run", log)
        self.assertEqual(len(read_index(self.out)), 8)

    def test_edited_design_is_refused(self):
        port = self.sim()
        self.run_design(port, limit=1)
        self.design.write_text(DESIGN.replace("replicates: 2", "replicates: 3"))
        with self.assertRaises(D.DesignError):
            self.run_design(port)

    def test_stops_when_board_is_missing(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]  # nothing listens here once closed
        rows, log = self.run_design(f"socket://127.0.0.1:{port}", max_consecutive_failures=2)
        self.assertEqual([r["status"] for r in rows], ["no_port"] * 2)
        self.assertIn("stopping", log)

    def test_dry_run_writes_nothing(self):
        rows, log = self.run_design(None, dry_run=True)
        self.assertEqual(rows, [])
        self.assertIn("4 treatments x 2 replicates = 8 runs", log)
        self.assertFalse(self.out.exists())


class TestSimulatorModel(unittest.TestCase):
    def test_effects_point_the_right_way(self):
        labels = S.load_labels()
        base = {"model": "cnn", "affinity": "same", "prio": "equal", "load": "off", "mem": "sram",
                "alloc": "static", "frag": "0", "seq": "full"}

        def res(**kw):
            return S.simulate(dict(base, **kw), 1, labels)[2]

        self.assertLess(res(model="dnn")["inf_mean_us"], res()["inf_mean_us"])
        self.assertLess(res()["inf_mean_us"], res(model="dscnn")["inf_mean_us"])
        self.assertGreater(res(mem="psram")["inf_mean_us"], res()["inf_mean_us"])
        self.assertGreater(res(prio="higher")["blocks_lost"], res(affinity="split", prio="higher")["blocks_lost"])
        c = S.simulate(dict(base, model="dscnn", frag="2", seq="none"), 1, labels)
        self.assertEqual((c[0], c[2]["alloc_fail"]), ([], 1))


if __name__ == "__main__":
    unittest.main()
