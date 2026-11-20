import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protocol as P  # noqa: E402

CFG = {"model": "cnn", "affinity": "split", "prio": "higher", "load": "on", "mem": "sram",
       "alloc": "static", "frag": "0", "seq": "full"}


class TestProtocol(unittest.TestCase):
    def test_cfg_round_trip(self):
        line = P.cfg_line(CFG, 42)
        self.assertEqual(line, "CFG model=cnn affinity=split prio=higher load=on mem=sram alloc=static "
                               "frag=0 seq=full demo=off seed=42")
        self.assertEqual(P.parse_cfg(line), (dict(CFG, demo="off"), 42))

    def test_demo_defaults_to_off(self):
        self.assertEqual(P.validate(CFG)["demo"], "off")
        self.assertEqual(P.validate(dict(CFG, demo="on"))["demo"], "on")
        with self.assertRaises(P.ProtocolError):
            P.validate(dict(CFG, demo="maybe"))

    def test_yaml_values_are_normalized(self):
        cfg = dict(CFG, load=True, frag=0)
        self.assertEqual(P.validate(cfg)["load"], "on")
        self.assertEqual(P.validate(dict(CFG, load=False))["load"], "off")
        self.assertEqual(P.validate(cfg)["frag"], "0")

    def test_invalid_cfg(self):
        with self.assertRaises(P.ProtocolError):
            P.cfg_line(dict(CFG, model="resnet"), 1)
        with self.assertRaises(P.ProtocolError):
            P.cfg_line({k: v for k, v in CFG.items() if k != "mem"}, 1)
        with self.assertRaises(P.ProtocolError):
            P.parse_cfg("CFG model=cnn")
        with self.assertRaises(P.ProtocolError):
            P.parse_cfg("HELLO")

    def test_parse_lines(self):
        self.assertEqual(P.parse_line("READY\r"), ("READY", None))
        self.assertEqual(P.parse_line("ERR arena too large"), ("ERR", "arena too large"))
        self.assertEqual(P.parse_line("DET,123456,4,230"), ("DET", {"sample": 123456, "label_id": 4, "score": 230}))
        self.assertEqual(P.parse_line("TASK,T3,stack_free_min=1200,cpu_pct=7.5"),
                         ("TASK", {"name": "T3", "stack_free_min": 1200, "cpu_pct": 7.5}))
        self.assertEqual(P.parse_line("RESULT,inf_mean_us=7512,blocks_lost=0"),
                         ("RESULT", {"inf_mean_us": 7512, "blocks_lost": 0}))
        self.assertEqual(P.parse_line("I (312) cpu_start: boot"), ("LOG", "I (312) cpu_start: boot"))

    def test_bad_lines_raise(self):
        for line in ("DET,1,2", "DET,a,b,c", "RESULT,novalue", "TASK"):
            with self.assertRaises(P.ProtocolError, msg=line):
                P.parse_line(line)


if __name__ == "__main__":
    unittest.main()
