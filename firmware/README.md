# firmware

ESP-IDF project for the ESP32-S3 N16R8.

| Task | Role | Period |
| --- | --- | --- |
| T1 Capture | I2S/DMA in 20 ms blocks into a ring buffer; counts overflows | 20 ms |
| T2 Features | MFCC for each block | 20 ms |
| T3 Inference | TFLM + decision rule | 100 ms |
| T4 Logger | Metrics kept in RAM; sent over UART at the end of the run | low priority |

Each task toggles a GPIO pin when it starts and finishes, so timings are measured with the logic analyzer.
The experiment configuration (affinity, priorities, load, model, memory) is received over UART at boot.

First step: validate the measurement chain with empty tasks that only toggle their GPIO pins, before integrating the model.

Still to confirm: `esp-tflite-micro` compatibility with ESP-IDF v6.1; if it is not compatible, pin the latest working 5.x release.
