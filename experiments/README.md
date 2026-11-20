# experiments

| Experiment | Factors | Treatments | Replicates | Runs |
| --- | --- | --- | --- | --- |
| [A](designs/A.yaml). Scheduling | affinity (3) × T3 priority relative to T1 (3) × background load (2), CNN | 18 | 5 | 90 |
| [B](designs/B.yaml). Memory | model (3) × SRAM/PSRAM placement (2) × static/dynamic allocation (2) | 12 | 5 | 60 |
| [C](designs/C.yaml). Fragmentation | fragmentation level (3) × model (3), no test sequence | 9 | 1 | 9 |
| [D](designs/D.yaml). Strategies | base/best/alternative strategy (3) × model (3) | 9 | 5 | 45 |

5 replicates is the minimum from the proposal; the final number is set from the pilot run. `best` and `alternative` in D are filled in after analysing A and B (the runner refuses D until then).

## Files

| File | Role |
| --- | --- |
| [`protocol.py`](protocol.py) | UART protocol: CFG keys and values, line parser |
| [`design.py`](design.py) | Loads a design and expands it into a randomized run plan |
| [`runner.py`](runner.py) | Runs a design on the board and records every run |
| [`sim_board.py`](sim_board.py) | Board simulator over TCP, to test the runner and the analysis without hardware |
| `designs/*.yaml` | Fixed settings, factors, levels, replicates and seed of each experiment |
| `results/<design>/` | Real results (versioned) |

## UART protocol

```
board -> PC   READY
PC -> board   CFG model=cnn affinity=split prio=higher load=on mem=sram alloc=static frag=0 seq=full seed=42
board -> PC   OK                              (or ERR <reason>)
board -> PC   DET,<sample>,<label_id>,<score>  one per detection
board -> PC   TASK,T3,stack_free_min=1200,cpu_pct=7.5
board -> PC   RESULT,inf_mean_us=7512,inf_sd_us=210,...,blocks_lost=0,alloc_fail=0
board -> PC   DONE
```

`sample` is the position in the test sequence where the detection fired; latency, missed words and false alarms are computed on the PC against [`test_sequence/labels.csv`](../test_sequence/labels.csv). Any other line (boot log) is only logged. `seq=none` loads the model without playing the sequence (experiment C). `RESULT` fields are listed in `protocol.RESULT_FIELDS`.

## Running

```bash
pip install -r requirements.txt
python runner.py designs/A.yaml --dry-run          # plan and estimated time (~8 h for A)
python runner.py designs/A.yaml --port COM5        # the board's UART port (not the native USB one)
```

Per run the runner resets the board (RTS → EN), waits for `READY` (20 s), sends `CFG`, waits for `OK` (5 s) and then for `DONE` (7 min). It saves:

- `results/<design>/plan.json`: the frozen randomized order with the seed of every run.
- `results/<design>/runs/<run_id>.json` and `.log`: parsed outcome and raw serial log of each run.
- `results/<design>/index.csv`: one row per run with factors, status and summary fields, rebuilt after each run. **This is the table the analysis reads.**

A failed run is recorded with its reason (`no_port`, `no_ready`, `no_ok`, `rejected`, `timeout`, `reset`, `no_result`) and the runner moves on; after 3 failures in a row it stops (board unplugged?). Interrupting is safe: the same command continues where it stopped, `--retry-failed` repeats the failed runs, and resuming with an edited design file is refused. `--capture "sigrok-cli ... -o {path}"` starts a logic analyzer capture for each run, saved next to its log.

## Without the board

```bash
python sim_board.py --port 5555 --fault-rate 0.05
python runner.py designs/A.yaml --port socket://localhost:5555 --out sim_results/A --done-timeout 5
python -m unittest discover -s tests
```

The simulator invents plausible numbers (factor effects in the expected direction plus run-to-run noise) so the analysis can be written before real data exists. **Simulated results are not measurements**: they go to `sim_results/` (not versioned), never to `results/`.
