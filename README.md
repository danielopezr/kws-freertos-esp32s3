# kws-freertos-esp32s3

Evaluating how FreeRTOS scheduling and memory management affect real-time keyword spotting (TinyML) on an ESP32-S3, using factorial experiments analysed with ANOVA.

Final project for an **Operating Systems** course. It measures how FreeRTOS scheduling decisions (core affinity, task priorities) and memory management decisions (internal SRAM vs. PSRAM, static vs. dynamic allocation, heap fragmentation) affect a TinyML **keyword spotting** application that listens to audio continuously at 16 kHz on an ESP32-S3.

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repository and structure | ✅ |
| 1 | Models (DNN, CNN, DS-CNN), base firmware and baseline characterization | ⏳ |
| 2 | Experiment A: scheduling | ⏳ |
| 3 | Experiments B and C: memory and fragmentation | ⏳ |
| 4 | Experiment D: alternative strategy and comparison | ⏳ |

## Hardware

| Item | Model |
| --- | --- |
| Microcontroller | ESP32-S3 N16R8 (2 × Xtensa LX7 at 240 MHz, 512 KB SRAM, 8 MB PSRAM, 16 MB Flash) |
| Microphone | INMP441 (I2S) |
| Measurement | 8-channel 24 MHz USB logic analyzer (PulseView / sigrok) |

## Layout

```
docs/           Final report and diagrams
training/       Model training, INT8 quantization and export (Python)
test_sequence/  Fixed 5-minute test sequence and its labels
firmware/       ESP-IDF project: FreeRTOS tasks T1–T4, TFLM and GPIO probes
experiments/    Factorial designs (A–D), UART experiment runner and results
analysis/       Descriptive statistics, ANOVA, Tukey HSD and plots
```

## Workflow

- `main` always builds; every change goes on its own branch (`feat/…`, `fix/…`) and is merged through a pull request.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/), e.g. `feat: add i2s capture task`.
- The dataset (Speech Commands v2) is **not** committed; it is downloaded by a script in `training/`.
- Experiment settings are sent over UART at runtime, so every run uses the same firmware binary.

## Authors

- Andrés Felipe Flórez Correa
- Daniel López Restrepo
