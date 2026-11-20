# experiments

| Experiment | Factors | Treatments |
| --- | --- | --- |
| A. Scheduling | affinity (3) × T3 priority relative to T1 (3) × background load (2) | 18 |
| B. Memory | model (3) × SRAM/PSRAM placement (2) × static/dynamic allocation (2) | 12 |
| C. Fragmentation | fragmentation level (3) × model (3) | 9 |
| D. Strategies | base/best/alternative strategy (3) × model (3) | 9 |

- `designs/`: one file per experiment with its factors and levels.
- `runner.py` (to do): randomizes the run order with a recorded seed, sends each configuration over UART, resets the board and saves one CSV per run in `results/`.

At least 5 replicates per treatment; the final number is set from the Phase 1 pilot run.
