"""Run a factorial design on the board, one run per treatment x replicate, in random order.

    python runner.py designs/A.yaml --port COM5               # real board (UART port)
    python runner.py designs/A.yaml --dry-run                 # show the plan and the estimated time
    python runner.py designs/A.yaml --port socket://localhost:5555 --out /tmp/sim   # simulator

For each run: reset the board (RTS), wait for READY, send CFG, wait for OK, collect
DET / TASK / RESULT lines until DONE (7 min limit). The outcome of every run, good
or bad, is saved as results/<design>/runs/<run_id>.json plus the raw .log, and
results/<design>/index.csv (one row per run: factors, status and summary fields,
the table the analysis reads) is rebuilt after each run.

Interrupting is safe: running the same command again skips runs that are already
recorded (use --retry-failed to repeat the failed ones). The plan is frozen in
plan.json; resuming with an edited design file is refused.
"""

import argparse
import csv
import json
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import design as DS
from protocol import RESULT_FIELDS, ProtocolError, cfg_line, parse_line

HERE = Path(__file__).resolve().parent
RUN_MINUTES = {"full": 5.5, "none": 0.5}  # board time per run, for the estimate
BOOT_MARKERS = ("ESP-ROM:", "rst:")       # a reboot in the middle of a run


class _Socket:
    """socket://host:port link to sim_board.py with the small part of the pyserial API used here
    (pyserial's own socket:// handler drops data on Windows)."""

    in_waiting = 0

    def __init__(self, url):
        host, port = url[len("socket://"):].rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=5)
        self.sock.settimeout(0.1)

    def read(self, n):
        try:
            data = self.sock.recv(4096)
        except socket.timeout:
            return b""
        if not data:
            raise OSError("connection closed")
        return data

    def write(self, data):
        self.sock.sendall(data)

    def flush(self):
        pass

    def close(self):
        self.sock.close()


class Board:
    """Line-based link to the board: a serial port (COM5, /dev/ttyUSB0) or socket://host:port (simulator)."""

    def __init__(self, port, baud=115200):
        import serial  # imported here so --dry-run works without pyserial
        self._serial = serial
        self.port, self.baud = port, baud
        self.ser = None
        self.buf = b""

    def _open(self):
        if self.port.startswith("socket://"):
            self.ser = _Socket(self.port)
        else:
            self.ser = self._serial.serial_for_url(self.port, baudrate=self.baud, timeout=0.1)
        self.buf = b""

    def reset(self):
        """Hardware reset through the USB-UART bridge (RTS -> EN); a socket is reconnected instead.

        Returns False when the port cannot be opened (board unplugged, wrong port).
        """
        try:
            if self.port.startswith("socket://"):
                self.close()
                self._open()
                return True
            if self.ser is None:
                self._open()
            self.ser.dtr = False   # keep GPIO0 high: boot the app, not the ROM downloader
            self.ser.rts = True    # EN low
            time.sleep(0.1)
            self.ser.rts = False
            self.ser.reset_input_buffer()
        except (OSError, self._serial.SerialException):
            self.close()
            return False
        self.buf = b""
        return True

    def send(self, line):
        self.ser.write((line + "\n").encode())
        self.ser.flush()

    def readline(self, deadline):
        """Next line without the newline, or None when the deadline (time.monotonic) passes."""
        while True:
            if b"\n" in self.buf:
                raw, self.buf = self.buf.split(b"\n", 1)
                return raw.decode("utf-8", "replace").rstrip("\r")
            if time.monotonic() >= deadline:
                return None
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
            except (OSError, self._serial.SerialException):
                return None  # port vanished or the peer closed it
            self.buf += chunk

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except OSError:
                pass
            self.ser = None


def execute(board, run, timeouts, log, capture=None):
    """Play one run on the board. Returns the run record (status ok or the failure reason)."""
    rec = {"status": "ok", "result": {}, "detections": [], "tasks": {}, "parse_errors": 0}
    t0 = time.monotonic()

    def line_until(seconds):
        line = board.readline(time.monotonic() + seconds)
        if line is not None:
            log.write(f"{time.monotonic() - t0:9.3f} < {line}\n")
        return line

    proc = None
    if capture:
        cmd = capture.format(path=str(Path(log.name).with_suffix(".sr")))
        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not board.reset():
            rec["status"] = "no_port"
            return rec
        deadline = time.monotonic() + timeouts["boot"]
        while True:
            line = line_until(max(0.0, deadline - time.monotonic()))
            if line is None:
                rec["status"] = "no_ready"
                return rec
            if line.strip() == "READY":
                break

        cfg = cfg_line(run["cfg"], run["seed"])
        log.write(f"{time.monotonic() - t0:9.3f} > {cfg}\n")
        board.send(cfg)
        line = line_until(timeouts["ok"])
        kind = None
        while line is not None:
            kind, payload = parse_line(line)
            if kind in ("OK", "ERR"):
                break
            line = line_until(timeouts["ok"])
        if kind != "OK":
            rec["status"] = "rejected" if kind == "ERR" else "no_ok"
            rec["error"] = payload if kind == "ERR" else None
            return rec

        deadline = time.monotonic() + timeouts["done"]
        while True:
            line = line_until(max(0.0, deadline - time.monotonic()))
            if line is None:
                rec["status"] = "timeout"
                return rec
            try:
                kind, payload = parse_line(line)
            except ProtocolError:
                rec["parse_errors"] += 1
                continue
            if kind == "DET":
                rec["detections"].append(payload)
            elif kind == "TASK":
                rec["tasks"][payload.pop("name")] = payload
            elif kind == "RESULT":
                rec["result"] = payload
            elif kind == "DONE":
                if not rec["result"]:
                    rec["status"] = "no_result"
                return rec
            elif kind == "READY" or (kind == "LOG" and line.startswith(BOOT_MARKERS)):
                rec["status"] = "reset"
                return rec
    finally:
        rec["duration_s"] = round(time.monotonic() - t0, 3)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def index_rows(plan, runs_dir):
    factor_names = list(plan[0]["factors"]) if plan else []
    rows, extra = [], set()
    for run in plan:
        path = runs_dir / f"{run['run_id']}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        row = {"order": run["order"], "run_id": run["run_id"], "treatment": run["treatment"],
               "replicate": run["replicate"], "seed": run["seed"], **run["factors"],
               "status": rec["status"], "attempts": rec["attempts"], "started": rec["started"],
               "duration_s": rec["duration_s"], "detections": len(rec["detections"]),
               "parse_errors": rec["parse_errors"]}
        row.update(rec["result"])
        for task, fields in rec["tasks"].items():
            row.update({f"{task}_{k}": v for k, v in fields.items()})
        extra |= set(row) - set(RESULT_FIELDS)
        rows.append(row)
    fixed = ["order", "run_id", "treatment", "replicate", "seed", *factor_names, "status", "attempts",
             "started", "duration_s", "detections", "parse_errors"]
    columns = fixed + list(RESULT_FIELDS) + sorted(extra - set(fixed))
    return columns, rows


def write_index(plan, out):
    columns, rows = index_rows(plan, out / "runs")
    tmp = out / "index.csv.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out / "index.csv")
    return rows


def freeze_plan(d, design_path, out):
    """Write plan.json the first time; afterwards check that the design did not change."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.json"
    fp = DS.fingerprint(d)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["fingerprint"] != fp:
            raise DS.DesignError(f"{design_path} changed since {path} was written; "
                                 "use a new --out directory or restore the design")
        return saved["runs"]
    runs = DS.plan(d)
    path.write_text(json.dumps({"design": d["name"], "fingerprint": fp, "seed": d["seed"],
                                "replicates": d["replicates"], "runs": runs}, indent=1))
    return runs


def estimate_minutes(runs):
    return sum(RUN_MINUTES[r["cfg"]["seq"]] for r in runs)


def run_design(design_path, port=None, out=None, baud=115200, retry_failed=False, limit=None,
               timeouts=None, capture=None, max_consecutive_failures=3, dry_run=False, echo=print):
    d = DS.load(design_path)
    out = Path(out) if out else HERE / "results" / d["name"]
    timeouts = {"boot": 20.0, "ok": 5.0, "done": 420.0, **(timeouts or {})}

    if dry_run:
        runs = DS.plan(d)
        echo(f"design {d['name']}: {len(DS.treatments(d))} treatments x {d['replicates']} replicates = "
             f"{len(runs)} runs, ~{estimate_minutes(runs) / 60:.1f} h on the board")
        for r in runs[:10]:
            echo(f"  {r['order']:3d} {r['run_id']:12s} {cfg_line(r['cfg'], r['seed'])}")
        if len(runs) > 10:
            echo(f"  ... {len(runs) - 10} more")
        return []

    plan = freeze_plan(d, design_path, out)
    runs_dir = out / "runs"
    runs_dir.mkdir(exist_ok=True)
    todo = []
    for run in plan:
        path = runs_dir / f"{run['run_id']}.json"
        prev = json.loads(path.read_text()) if path.exists() else None
        if prev is None or (retry_failed and prev["status"] != "ok"):
            todo.append((run, prev["attempts"] if prev else 0))
    if limit is not None:
        todo = todo[:limit]
    echo(f"design {d['name']}: {len(plan)} runs, {len(plan) - len(todo)} already recorded, "
         f"{len(todo)} to run (~{estimate_minutes([r for r, _ in todo]) / 60:.1f} h)")

    board = Board(port, baud)
    failures_in_a_row = 0
    try:
        for n, (run, attempts) in enumerate(todo, start=1):
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with open(runs_dir / f"{run['run_id']}.log", "w", encoding="utf-8") as log:
                rec = execute(board, run, timeouts, log, capture)
            rec.update({"run_id": run["run_id"], "order": run["order"], "seed": run["seed"],
                        "factors": run["factors"], "cfg": run["cfg"], "attempts": attempts + 1,
                        "started": started})
            (runs_dir / f"{run['run_id']}.json").write_text(json.dumps(rec, indent=1))
            write_index(plan, out)
            echo(f"[{n}/{len(todo)}] #{run['order']} {run['run_id']}: {rec['status']} "
                 f"({rec['duration_s']:.0f} s, {len(rec['detections'])} detections)")
            failures_in_a_row = 0 if rec["status"] == "ok" else failures_in_a_row + 1
            if failures_in_a_row >= max_consecutive_failures:
                echo(f"stopping: {failures_in_a_row} failed runs in a row (is the board connected?)")
                break
    finally:
        board.close()
    rows = write_index(plan, out)
    ok = sum(r["status"] == "ok" for r in rows)
    echo(f"{ok}/{len(plan)} runs ok, {len(rows) - ok} failed, {len(plan) - len(rows)} pending -> {out / 'index.csv'}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", help="designs/A.yaml ...")
    ap.add_argument("--port", help="serial port (COM5) or pyserial URL (socket://host:port)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", help="results directory (default: experiments/results/<design>)")
    ap.add_argument("--retry-failed", action="store_true", help="repeat runs whose status is not ok")
    ap.add_argument("--limit", type=int, help="run at most this many runs now")
    ap.add_argument("--done-timeout", type=float, default=420.0, help="seconds from OK to DONE")
    ap.add_argument("--capture", help='logic analyzer command started for each run, e.g. '
                    '"sigrok-cli -d fx2lafw --config samplerate=4m --continuous -o {path}"')
    ap.add_argument("--dry-run", action="store_true", help="only print the plan")
    args = ap.parse_args()
    if not args.dry_run and not args.port:
        ap.error("--port is required (or use --dry-run)")
    try:
        run_design(args.design, args.port, args.out, args.baud, args.retry_failed, args.limit,
                   {"done": args.done_timeout}, args.capture, dry_run=args.dry_run)
    except (DS.DesignError, ProtocolError) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit("interrupted: run the same command again to continue")


if __name__ == "__main__":
    main()
