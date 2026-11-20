# firmware

ESP-IDF project for the ESP32-S3 N16R8.

| Task | Role | Period |
| --- | --- | --- |
| T1 Capture | I2S/DMA in 20 ms blocks into a ring buffer; counts overflows | 20 ms |
| T2 Features | MFCC for each block ([`components/kws_mfcc`](components/kws_mfcc)) | 20 ms |
| T3 Inference | TFLM + decision rule | 100 ms |
| T4 Logger | Metrics kept in RAM; sent over UART at the end of the run | low priority |

Each task toggles a GPIO pin when it starts and finishes, so timings are measured with the logic analyzer.
The experiment configuration (affinity, priorities, load, model, memory) is received over UART at boot.

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

`sdkconfig` itself is generated and not versioned.

## Current state: skeleton + probe test

`main/app_main.cc` checks that everything builds, links and runs, then starts the probe test:

1. Prints the ESP-IDF version, cores and free internal / PSRAM heap.
2. Builds a TFLM op resolver with every op the three models use.
3. Runs the MFCC on a 1 kHz tone and times one frame. Expected: `c0 ≈ -65.83`, `c1 ≈ 4.86` (Python reference).
4. **Probe test** ([`main/probes.c`](main/probes.c), plan step 5): FreeRTOS tasks that only raise their pin, busy-wait a known time and sleep until the next period, so the logic analyzer must see exactly these signals:

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
- **sigrok-cli** (command-line captures, needed for the 5-minute test and `runner.py --capture`): installer from [sigrok.org/wiki/Downloads](https://sigrok.org/wiki/Downloads).

### 2. Wiring

Board straddling the breadboard; analyzer channels D0–D5 to GPIO 4, 5, 6, 7, 15, 16 and **analyzer GND to board GND** (without a common ground the capture is garbage). LED anode to GPIO 16, cathode through 220 Ω to GND. The microphone and the OLED are not needed for this step.

### 3. Flash and watch the log

```bash
idf.py -p COMx flash monitor      # Ctrl+] leaves the monitor
```

Expected: the ESP-IDF version with `free PSRAM` ≈ 8 MB, the MFCC line, `skeleton OK`, `probe test running ...`, then a `PROBES` table every 5 s with periods ≈ 20 / 20 / 100 / 500 ms, widths ≈ 1000 / 2000 / 10000 / 500 µs and `overruns` 0. The LED blinks once per second.

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

If the PC sustains 5 minutes at 250 kHz without dropping samples, the experiments measure latencies with the analyzer, as the proposal says. If not, latencies are measured on the board with `esp_timer` and the analyzer only calibrates them (plan step 5.4).

Next: tasks T1–T4 doing real work (I2S capture, MFCC, inference, logging) instead of busy waits.
