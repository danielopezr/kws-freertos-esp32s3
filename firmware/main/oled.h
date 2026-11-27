// SSD1306 128x64 OLED on I2C (SDA GPIO 8, SCL GPIO 9), used only with demo=on (plan step 16).
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OLED_SDA_GPIO 8
#define OLED_SCL_GPIO 9
#define OLED_W 128
#define OLED_H 64

// Wiring test (`OLED` command): scans the I2C bus, initialises the display and shows three test
// screens (all on, checkerboard, frame with an X). Prints the devices found and the time of one
// full-screen update. Returns false when no SSD1306 answers at 0x3C or 0x3D.
bool oled_test(int sda, int scl);  // the display wiring: OLED_SDA_GPIO, OLED_SCL_GPIO

#ifdef __cplusplus
}
#endif
