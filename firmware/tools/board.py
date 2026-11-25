"""Send one command to the board and print what it answers.

    python board.py COM7 ARENA
    python board.py COM7 "EVAL model=cnn"
    python board.py COM7 "CFG model=cnn affinity=none prio=equal load=off mem=sram alloc=static frag=0 seq=full demo=off seed=1"
    python board.py COM7 PROBE --seconds 30

Resets the board (RTS -> EN, as runner.py does), waits for READY, sends the command and
prints every line until the answer is complete (DONE, EVAL_SUMMARY, the third ARENA line,
an ERR) or --seconds pass. --log saves the whole exchange.
"""

import argparse
import sys
import time

import serial

END = ("DONE", "EVAL_SUMMARY", "ERR")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port")
    ap.add_argument("command")
    ap.add_argument("--seconds", type=float, default=420.0, help="give up after this long (default 420)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--log", help="also write every line to this file")
    ap.add_argument("--quiet", action="store_true", help="do not echo DET/EVAL lines (still logged)")
    args = ap.parse_args()

    log = open(args.log, "w", encoding="utf-8") if args.log else None
    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = args.port, args.baud, 0.1
    ser.dtr = ser.rts = False
    ser.open()
    ser.rts = True  # EN low: reset, GPIO0 high (DTR false) so the app boots
    time.sleep(0.1)
    ser.rts = False
    ser.reset_input_buffer()

    buf, sent, arena = b"", False, 0
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        buf += ser.read(ser.in_waiting or 1)
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode("utf-8", "replace").rstrip("\r")
            if log:
                log.write(line + "\n")
            if not (args.quiet and line.startswith(("DET,", "EVAL,"))):
                print(line, flush=True)
            if not sent:
                if line == "READY":
                    ser.write((args.command + "\n").encode())
                    sent = True
                continue
            arena += line.startswith("ARENA,")
            if line.startswith(END) or arena == 3:
                ser.close()
                return 0
    print(f"-- no complete answer after {args.seconds:.0f} s", file=sys.stderr)
    ser.close()
    return 1


if __name__ == "__main__":
    sys.exit(main())
