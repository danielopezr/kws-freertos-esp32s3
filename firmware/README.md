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

## Current state: skeleton

`main/app_main.cc` only checks that everything builds, links and runs:

1. Prints the ESP-IDF version, cores and free internal / PSRAM heap.
2. Builds a TFLM op resolver with every op the three models use.
3. Runs the MFCC on a 1 kHz tone and times one frame. Expected: `c0 ≈ -65.83`, `c1 ≈ 4.86` (Python reference).

Next: tasks T1–T4 that only toggle their GPIO probes, to validate the measurement chain with the logic analyzer before adding audio and the model.
