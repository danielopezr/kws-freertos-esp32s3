"""Board simulator: speaks the UART protocol over TCP so runner.py can be tested without hardware.

    python sim_board.py --port 5555 [--fault-rate 0.1] [--seconds 0]
    python runner.py designs/A.yaml --port socket://localhost:5555 --out sim_results/A

Each TCP connection is one boot (the runner reconnects to "reset" it). The numbers
are invented but follow the effects the experiments look for (larger models are
slower, PSRAM slows inference, T3 above T1 on the same core loses audio blocks,
background load hurts most on a shared core), with run-to-run noise drawn from
the run seed, so the analysis scripts can be developed on realistic-looking data.
They are NOT measurements and must never be mixed with real results.
"""

import argparse
import csv
import socket
import threading
import time
from pathlib import Path

import numpy as np

from protocol import ProtocolError, parse_cfg

LABELS_CSV = Path(__file__).resolve().parents[1] / "test_sequence" / "labels.csv"
SAMPLE_RATE = 16_000
SEQ_SECONDS = 300
BLOCKS = SEQ_SECONDS * 50          # 20 ms capture blocks
INFERENCES = SEQ_SECONDS * 10      # one every 100 ms

INF_US = {"dnn": 1_800, "cnn": 7_500, "dscnn": 32_000}
PSRAM_FACTOR = {"dnn": 1.6, "cnn": 1.35, "dscnn": 1.3}   # weight-bound DNN suffers most
ACCURACY = {"dnn": 0.79, "cnn": 0.915, "dscnn": 0.952}  # INT8 test accuracy of the first run
ARENA = {"dnn": 22_000, "cnn": 64_000, "dscnn": 118_000}
LARGEST_BLOCK = {"0": 290_000, "1": 150_000, "2": 70_000}
LOSS = {("same", "higher"): 0.02, ("same", "equal"): 0.004, ("same", "lower"): 0.0005,
        ("none", "higher"): 0.006, ("none", "equal"): 0.0015, ("none", "lower"): 0.0004}


def load_labels(path=LABELS_CSV):
    with open(path, newline="") as f:
        return [{"label_id": int(r["label_id"]), "voiced_end": int(r["voiced_end"])} for r in csv.DictReader(f)]


def simulate(cfg, seed, labels):
    """(detections, tasks, result) for one run."""
    rng = np.random.default_rng(seed)
    model, shared = cfg["model"], cfg["affinity"] != "split"
    run_noise = rng.normal(1.0, 0.04)
    inf = INF_US[model] * run_noise
    if cfg["mem"] == "psram":
        inf *= PSRAM_FACTOR[model]
    if cfg["alloc"] == "dynamic":
        inf *= 1.03
    if cfg["load"] == "on":
        inf *= 1.25 if shared else 1.08
    arena = ARENA[model]
    largest = LARGEST_BLOCK[cfg["frag"]] if cfg["mem"] == "sram" else 4_000_000
    alloc_fail = int(arena > largest)
    result = {"heap_free_min": int(largest * 1.4 - arena * (1 - alloc_fail)), "heap_largest_block": largest,
              "arena_bytes": arena, "alloc_fail": alloc_fail}
    if cfg["seq"] == "none" or alloc_fail:
        return [], {}, result

    loss = LOSS.get((cfg["affinity"], cfg["prio"]), 0.0002) * inf / INF_US["cnn"]
    if cfg["load"] == "on" and shared:
        loss *= 2
    loss = min(0.5, loss * rng.lognormal(0, 0.2))
    lost = int(rng.binomial(BLOCKS, loss))
    samples = rng.normal(inf, inf * 0.05, INFERENCES).clip(inf * 0.8)
    result.update({"inf_mean_us": int(samples.mean()), "inf_sd_us": int(samples.std()),
                   "inf_p95_us": int(np.percentile(samples, 95)), "inf_max_us": int(samples.max()),
                   "inferences": INFERENCES, "blocks_total": BLOCKS, "blocks_lost": lost})

    p_detect = ACCURACY[model] * (1 - 5 * loss)
    latency_ms = 160 + 3 * inf / 1000 + (40 if cfg["load"] == "on" and shared else 0)
    detections = []
    for item in labels:
        if item["label_id"] >= 2 and rng.random() < p_detect:
            label = item["label_id"] if rng.random() > 0.03 else int(rng.integers(2, 12))
        elif item["label_id"] < 2 and rng.random() < (1 - ACCURACY[model]) * 0.3:
            label = int(rng.integers(2, 12))  # false alarm on an unknown word
        else:
            continue
        delay = max(20.0, rng.normal(latency_ms, 30)) * SAMPLE_RATE / 1000
        detections.append((int(item["voiced_end"] + delay), label, int(rng.integers(205, 256))))

    busy = {"T1": 2.0, "T2": 9.0, "T3": inf / 1000, "T4": 0.3}  # % of one core; T3 runs every 100 ms
    tasks = {name: {"stack_free_min": int(rng.integers(900, 1400)), "cpu_pct": round(pct * rng.normal(1, 0.02), 2)}
             for name, pct in busy.items()}
    return sorted(detections), tasks, result


class Simulator:
    def __init__(self, host="127.0.0.1", port=0, fault_rate=0.0, seconds=0.0, seed=0, labels=None):
        self.labels = labels if labels is not None else load_labels()
        self.fault_rate, self.seconds = fault_rate, seconds
        self.rng = np.random.default_rng(seed)  # faults are per boot, so a retried run can succeed
        self.server = socket.create_server((host, port))
        self.port = self.server.getsockname()[1]
        self.boots = 0
        self._stop = threading.Event()

    def serve_forever(self):
        self.server.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self.server.accept()
            except (socket.timeout, OSError):
                continue
            conn.settimeout(None)  # do not inherit the accept() polling timeout
            with conn:
                try:
                    self.boot(conn)
                except (ConnectionError, OSError):
                    pass

    def start(self):
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        self.server.close()

    def boot(self, conn):
        self.boots += 1
        f = conn.makefile("rwb", buffering=0)

        def send(line):
            f.write((line + "\r\n").encode())

        for line in ("ESP-ROM:esp32s3-20210327", "I (312) cpu_start: simulated board", "READY"):
            send(line)
        line = f.readline().decode().strip()
        try:
            cfg, seed = parse_cfg(line)
        except ProtocolError as e:
            send(f"ERR {e}")
            return
        send("OK")
        fault = self.rng.random() < self.fault_rate and self.rng.choice(["hang", "reboot"])
        detections, tasks, result = simulate(cfg, seed, self.labels)
        for i, (sample, label, score) in enumerate(detections):
            if self.seconds:
                time.sleep(self.seconds / max(1, len(detections)))
            send(f"DET,{sample},{label},{score}")
            if fault and i == len(detections) // 2:
                break
        if fault == "hang":
            conn.settimeout(None)
            f.read()  # wait until the runner gives up and disconnects
            return
        if fault == "reboot":
            send("rst:0xc (RTC_SW_CPU_RST),boot:0x8 (SPI_FAST_FLASH_BOOT)")
            send("READY")
            f.read()
            return
        for name, fields in tasks.items():
            send(f"TASK,{name}," + ",".join(f"{k}={v}" for k, v in fields.items()))
        send("RESULT," + ",".join(f"{k}={v}" for k, v in result.items()))
        send("DONE")
        f.read()  # until the runner disconnects (its next reset)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--fault-rate", type=float, default=0.0, help="fraction of boots that hang or reboot")
    ap.add_argument("--seconds", type=float, default=0.0, help="time each run takes (0 = instant)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sim = Simulator(port=args.port, fault_rate=args.fault_rate, seconds=args.seconds, seed=args.seed)
    print(f"simulated board on socket://localhost:{sim.port} (Ctrl+C to stop)")
    try:
        sim.serve_forever()
    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
