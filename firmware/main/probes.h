// GPIO probes read by the logic analyzer, and the probe test of plan step 5.
//
// The probe test validates the measurement chain before any audio or ML runs:
// each task raises its pin, busy-waits a known time, lowers it and sleeps until
// its next period, so the analyzer must see exactly these periods and widths.
// firmware/tools/check_probes.py reads the expected values from this file.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pin plan (avoid 0/3/45/46 strapping, 19/20 USB, 33-37 octal PSRAM, 43/44 UART0)
#define PROBE_T1_GPIO 4     // analyzer CH0: T1 capture
#define PROBE_T2_GPIO 5     // analyzer CH1: T2 features
#define PROBE_T3_GPIO 6     // analyzer CH2: T3 inference
#define PROBE_T4_GPIO 7     // analyzer CH3: T4 logger
#define PROBE_MARK_GPIO 15  // analyzer CH4: end-of-word marker (playback)
#define PROBE_LED_GPIO 16   // analyzer CH5: detection LED (220 ohm to GND)

// Probe test: distinct widths so each channel is recognisable at a glance
#define PROBE_T1_PERIOD_MS 20
#define PROBE_T1_BUSY_US 1000
#define PROBE_T2_PERIOD_MS 20
#define PROBE_T2_BUSY_US 2000
#define PROBE_T3_PERIOD_MS 100
#define PROBE_T3_BUSY_US 10000
#define PROBE_T4_PERIOD_MS 500
#define PROBE_T4_BUSY_US 500
#define PROBE_MARK_PERIOD_MS 1000  // marker pulse once per second
#define PROBE_MARK_BUSY_US 100
#define PROBE_LED_PERIOD_MS 1000   // LED toggles every half period (50 % duty)
#define PROBE_LED_BUSY_US 500000

// Configure all probe pins as outputs, driven low.
void probes_init(void);

// Start the probe test tasks (T1, T2 on core 0; T3, T4 and the marker on core 1).
void probes_start_test(void);

// Print what the tasks measured on their own clock (esp_timer) since the last
// call: start-to-start period (mean / min / max), pulse width and overruns.
// Comparing this with the analyzer calibrates the internal measurement.
void probes_report(void);

#ifdef __cplusplus
}
#endif
