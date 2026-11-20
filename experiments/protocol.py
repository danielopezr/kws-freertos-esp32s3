"""UART protocol between the PC and the board (one text line per message).

    board -> PC   READY
    PC -> board   CFG model=cnn affinity=split prio=higher load=on mem=sram alloc=static frag=0 seq=full demo=off seed=42
    board -> PC   OK                      (or: ERR <reason>)
                  ... the run plays the test sequence (~5 min) ...
    board -> PC   DET,<sample>,<label_id>,<score>        one per detection
    board -> PC   TASK,<name>,<key>=<value>,...          one per task (T1..T4)
    board -> PC   RESULT,<key>=<value>,...               run summary
    board -> PC   DONE

Any other line (ESP-IDF boot log, debug prints) is kept in the run log and ignored.
`sample` is the index into the test sequence at which the detection fired, so
detection latency = (sample - voiced_end) / 16 kHz, measured on the board's own clock.
"""

# Configuration keys, in the order they are sent, and their allowed values
PARAMS = {
    "model": ("dnn", "cnn", "dscnn"),
    "affinity": ("same", "split", "none"),  # same: T1-T3 on one core; split: T1-T2 core 0, T3 core 1
    "prio": ("lower", "equal", "higher"),   # T3 priority relative to T1
    "load": ("off", "on"),                  # synthetic background task with a fixed duty cycle
    "mem": ("sram", "psram"),               # buffers and tensor arena
    "alloc": ("static", "dynamic"),
    "frag": ("0", "1", "2"),                # heap fragmentation pattern before loading the model
    "seq": ("full", "none"),                # none: only load the model (experiment C)
    "demo": ("off", "on"),                  # on: OLED display for the presentation; never in experiments
}

# Keys that may be left out of a design; they take these values
DEFAULTS = {"demo": "off"}

# Summary fields the firmware sends in RESULT (missing ones are left empty in index.csv)
RESULT_FIELDS = (
    "inf_mean_us", "inf_sd_us", "inf_p95_us", "inf_max_us", "inferences",
    "blocks_total", "blocks_lost", "heap_free_min", "heap_largest_block",
    "arena_bytes", "alloc_fail",
)


class ProtocolError(ValueError):
    pass


def normalize(value):
    """YAML turns on/off into booleans and 0 into int; the protocol wants text."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    return str(value)


def validate(cfg):
    cfg = {k: normalize(v) for k, v in {**DEFAULTS, **cfg}.items()}
    unknown = set(cfg) - set(PARAMS)
    missing = set(PARAMS) - set(cfg)
    if unknown or missing:
        raise ProtocolError(f"unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    for k, v in cfg.items():
        if v not in PARAMS[k]:
            raise ProtocolError(f"{k}={v} not in {PARAMS[k]}")
    return cfg


def cfg_line(cfg, seed):
    cfg = validate(cfg)
    return "CFG " + " ".join(f"{k}={cfg[k]}" for k in PARAMS) + f" seed={int(seed)}"


def parse_cfg(line):
    """'CFG k=v ... seed=N' -> (cfg dict, seed). Raises ProtocolError."""
    parts = line.split()
    if not parts or parts[0] != "CFG":
        raise ProtocolError(f"not a CFG line: {line!r}")
    try:
        kv = dict(p.split("=", 1) for p in parts[1:])
        seed = int(kv.pop("seed"))
    except (ValueError, KeyError) as e:
        raise ProtocolError(f"bad CFG line {line!r}: {e}") from None
    return validate(kv), seed


def _number(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _kv(fields):
    out = {}
    for f in fields:
        if "=" not in f:
            raise ProtocolError(f"expected key=value, got {f!r}")
        k, v = f.split("=", 1)
        out[k] = _number(v)
    return out


def parse_line(line):
    """Return (kind, payload). kind is READY, OK, ERR, DET, TASK, RESULT, DONE or LOG."""
    line = line.strip()
    if line in ("READY", "OK", "DONE"):
        return line, None
    if line.startswith("ERR"):
        return "ERR", line[3:].strip()
    head, _, rest = line.partition(",")
    fields = rest.split(",") if rest else []
    try:
        if head == "DET":
            sample, label, score = fields
            return "DET", {"sample": int(sample), "label_id": int(label), "score": _number(score)}
        if head == "TASK":
            return "TASK", {"name": fields[0], **_kv(fields[1:])}
        if head == "RESULT":
            return "RESULT", _kv(fields)
    except (ValueError, IndexError) as e:
        raise ProtocolError(f"bad {head} line {line!r}: {e}") from None
    return "LOG", line
