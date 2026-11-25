# firmware

ESP-IDF project for the ESP32-S3 N16R8: the real-time keyword spotting pipeline whose scheduling and memory management the experiments vary ([`main/pipeline.cc`](main/pipeline.cc)).

```
I2S DMA (20 ms blocks) -> T1 capture -> queue (8) -> T2 features: 1 MFCC frame per block -> ring of 49 frames (1 s)
                                                     -> T3 every 5 frames (100 ms): model + mean of last 3 outputs -> detection
T4 logger (low priority): heap watermark, detection LED        LOAD (load=on): synthetic background task
```

| Task | Role | Activation | Priority | Core (affinity same / split / none) |
| --- | --- | --- | --- | --- |
| T1 Capture | Waits for each 20 ms I2S DMA block; in playback replaces it with the test sequence; counts lost blocks | I2S DMA, 20 ms | 10 | 0 / 0 / any |
| T2 Features | One MFCC frame (previous + current block) per block ([`components/kws_mfcc`](components/kws_mfcc)) | each block | 9 | 0 / 0 / any |
| T3 Inference | TFLM on the last 49 frames, softmax, mean of the last 3 outputs, threshold, 1 s refractory | every 5 frames | `prio`: lower 8, equal 10, higher 11 | 0 / 1 / any |
| T4 Logger | Largest free block of the run's heap, LED off 200 ms after a detection; metrics go out at the end | 100 ms | 3 | any |
| LOAD | Busy 10 ms out of every 20 ms (50 % of a core) | 20 ms | 9 | 0 / 0 / any |

**Playback (`seq=full`)**: the I2S peripheral runs exactly as with the microphone (clock, DMA, interrupts, 20 ms pacing), but T1 replaces every block with the matching 20 ms of the fixed test sequence, memory-mapped from the `sequence` flash partition. A block the DMA drops because T1 read too late (`on_recv_q_ovf`) is skipped in the sequence too, as it would be lost from a real microphone. So every run hears the same 5 minutes of audio and the microphone is only needed for the demo.

**Measurement on the board**: inference time with `esp_timer` (1 µs), CPU use per task from the FreeRTOS run-time stats, stack watermarks, heap minimum (`heap_caps_monitor_local_minimum_free_size_*`), and each detection as the sequence sample at the instant of the decision (`analysis/scoring.py` turns it into the detection latency from `voiced_end`). The GPIO probes ([`main/probes.h`](main/probes.h)) mirror T1–T4, the end-of-word marker and the LED for the logic analyzer, which calibrates these internal measurements.

## Toolchain (fixed for the whole project)

| Item | Version |
| --- | --- |
| ESP-IDF | **v6.0.3** |
| `espressif/esp-tflite-micro` | 1.4.3 (officially supported up to ESP-IDF v6.0; v6.1 is not in its support table) |

Install on Windows with the ESP-IDF Installation Manager:

```powershell
winget install Espressif.EIM-CLI
eim install -i v6.0.3 -t esp32s3 -n true
```

## Build and flash

From an ESP-IDF terminal (the activation script created by EIM):

```bash
cd firmware
idf.py set-target esp32s3     # first time only
idf.py build
idf.py -p COMx flash monitor
```

## Fixed configuration

- [`sdkconfig.defaults`](sdkconfig.defaults): 240 MHz, 16 MB flash, octal PSRAM reachable **only** through `heap_caps_malloc(..., MALLOC_CAP_SPIRAM)` (so memory placement stays an explicit experiment factor), 1 kHz FreeRTOS tick, run-time statistics enabled, `-O2`.
- [`partitions.csv`](partitions.csv): 3 MB app + 10 MB `sequence` data partition for the fixed test sequence.

`sdkconfig` itself is generated and not versioned. The task watchdog is turned off at boot (`esp_task_wdt_deinit`): some treatments keep a core 100 % busy on purpose and it must not print during a run.

The test sequence goes into its partition once (it survives app reflashes; `esptool` checks the hash):

```bash
python -m esptool --chip esp32s3 -p COMx -b 921600 write-flash 0x310000 ../test_sequence/out/sequence.raw
```

## Fixed design parameters

Same for every run ([`main/pipeline.cc`](main/pipeline.cc)); only the CFG factors change.

| Parameter | Value |
| --- | --- |
| Audio block / DMA | 320 samples (20 ms); 4 DMA buffers of one block (80 ms of slack for T1) |
| T1 → T2 queue | 8 blocks (160 ms); a block that finds it full is lost (`ring_ovf`) |
| Features | 49 × 40 MFCC (1 s), normalised with the training mean/std, quantised with the model's input scale |
| Decision | softmax, mean of the last 3 outputs, keyword if ≥ 0.80 (`thr=` in CFG to change it), 1 s refractory |
| Replicates | before each utterance, 0–99.9 ms of silence (drawn from the run seed) is inserted 100 ms before its slot, so every word meets the 100 ms inference grid at its own phase, as real speech would (`seq_inserted` in RESULT, ~7.5 s per run). One common delay per run is not enough: every `voiced_end` lies on the 20 ms frame grid, so it would move all words at once and make the run mean depend on the delay. Detections are reported in sequence samples, so a word's latency is unchanged |
| `alloc=static` | tasks with `xTaskCreateStatic` (stacks reserved at link time), block pool and staging buffer allocated once |
| `alloc=dynamic` | `xTaskCreate`; T1 mallocs each block and T2 frees it, T3 mallocs its input buffer each inference |
| `mem` | buffers (blocks, feature ring, staging) and the tensor arena in internal SRAM or PSRAM; stacks are always internal, model weights stay in flash |
| `frag` | before loading the model, internal SRAM is filled with 8 KB blocks and freed in runs, keeping 1 of every 13 (1, holes ≈ 96 KB) or 5 (2, holes ≈ 32 KB) |

## Models on the board

| Model | Tensor arena (used) | Inference, arena in SRAM | Accuracy on the 150 sequence clips (`EVAL`) |
| --- | --- | --- | --- |
| DNN | 8.7 KB | 22.4 ms (340 KB of weights read from flash) | 73.3 % (110/150) |
| CNN | 52.5 KB | 18.3 ms | 89.3 % (134/150) |
| DS-CNN | 458.9 KB: **does not fit internal SRAM** (largest free block ≈ 262 KB), PSRAM only | ≈ 1 s in PSRAM | pending |

## Commands

After boot the board prints `READY` and waits for one line. [`tools/board.py`](tools/board.py) resets it, sends the line and prints the answer:

```bash
python tools/board.py COMx "CFG model=cnn affinity=none prio=equal load=off mem=sram alloc=static frag=0 seq=full demo=off seed=1"
python tools/board.py COMx "EVAL model=cnn"     # offline accuracy on the 150 clips (no real time)
python tools/board.py COMx ARENA                # tensor arena each model needs
python tools/board.py COMx PROBE --seconds 60   # probe test for the logic analyzer
```

`CFG` is the experiment protocol ([`experiments/protocol.py`](../experiments/protocol.py)); `runner.py` sends it. `demo=on` uses the live microphone, never ends and prints each detection as it happens.

## Probe test

[`main/probes.c`](main/probes.c), plan step 5, started with the `PROBE` command: FreeRTOS tasks that only raise their pin, busy-wait a known time and sleep until the next period, so the logic analyzer must see exactly these signals:

| Signal | GPIO | Analyzer | Period | High for | Task (priority, core) |
| --- | --- | --- | --- | --- | --- |
| T1 capture | 4 | D0 | 20 ms | 1 ms | 5, core 0 |
| T2 features | 5 | D1 | 20 ms | 2 ms | 4, core 0 |
| T3 inference | 6 | D2 | 100 ms | 10 ms | 3, core 1 |
| T4 logger | 7 | D3 | 500 ms | 0.5 ms | 1, core 1 |
| End-of-word marker | 15 | D4 | 1 s | 100 µs | 2, core 1 |
| LED (220 Ω to GND) | 16 | D5 | 1 s | 500 ms | same task as the marker |

All values live in [`main/probes.h`](main/probes.h); the checker reads them from there. Every 5 s the board also prints what the tasks measured on their own clock (`PROBES T1 ...` lines: period and width mean/min/max, overruns), to compare with the analyzer.

## Bring-up with the board

### 1. Drivers and tools (Windows, once)

- **USB-UART of the board:** plug the USB-C port labelled **UART/COM** and check that a `COMx` port appears in Device Manager. If it does not, install the WCH driver for the bridge chip written on the board (CH340 → CH341SER, CH343 → CH343SER).
- **PulseView** (logic analyzer GUI): `winget install Sigrok.PulseView`.
- **Analyzer driver:** the 24 MHz / 8-channel analyzer is an `fx2lafw` device. Plug it in, open **Zadig** (installed with PulseView), select the analyzer and install the **WinUSB** driver. PulseView then lists it as *fx2lafw (generic driver for FX2 based LAs)*.
- **sigrok-cli** (command-line captures, needed for the 5-minute test and `runner.py --capture`): use the **nightly** build (`sigrok-cli-NIGHTLY-x86_64-release-installer.exe` from [sigrok.org/download/binary/sigrok-cli](https://sigrok.org/download/binary/sigrok-cli/); tested with 0.8.0-git-f44dd91). The 0.7.2 release on Windows cancels every fx2lafw capture after about 2 s at any sample rate (`Device only sent N samples`).

### 2. Wiring

Board straddling the breadboard with its **header pins soldered** (pins only pressed into the holes make no contact: every channel reads a constant 1). Analyzer channels D0–D5 to GPIO 4, 5, 6, 7, 15, 16 and **analyzer GND to board GND** (without a common ground the capture is garbage). LED anode to GPIO 16, cathode through 220 Ω to GND. The microphone and the OLED are not needed for this step.

### 3. Flash and start the probe test

```bash
idf.py -p COMx flash
python tools/board.py COMx PROBE --seconds 60
```

Expected: the ESP-IDF version with `free PSRAM` ≈ 8 MB, `sequence partition: mapped`, `READY`, `probe test running ...`, then a `PROBES` table every 5 s with periods ≈ 20 / 20 / 100 / 500 ms, widths ≈ 1000 / 2000 / 10000 / 500 µs and `overruns` 0. The LED blinks once per second. The probe test keeps running after `board.py` exits, until the next reset.

### 4. Capture and check

In PulseView: 1 MHz, 5 s (≥ 3 periods of the 1 s signals), *Save as* `probes.sr`. Or:

```bash
sigrok-cli -d fx2lafw --config samplerate=1m --time 5000 -C D0,D1,D2,D3,D4,D5 -o probes.sr
python tools/check_probes.py probes.sr
```

`check_probes.py` passes when every mean period and width is within 1 % of nominal. If it fails, the problem is in the measurement chain (wiring, ground, sample rate), not in the ML. Its tests run without hardware: `python -m unittest discover -s tools/tests`.

### 5. Long capture (decides how latencies are measured)

```bash
sigrok-cli -d fx2lafw --config samplerate=250k --time 300000 -C D0,D1,D2,D3,D4,D5 -o long.sr
python tools/check_probes.py long.sr
```

If the PC sustains 5 minutes at 250 kHz without dropping samples, the experiments measure latencies with the analyzer, as the proposal says. If not, latencies are measured on the board with `esp_timer` and the analyzer only calibrates them (plan step 5.4). With sigrok-cli nightly the PC sustained the 5 minutes (75 M samples, 2026-11-25).
