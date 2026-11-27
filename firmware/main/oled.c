#include "oled.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define OLED_HZ 400000  // fast mode: a full screen is 1024 bytes, ~23 ms on the bus

static i2c_master_bus_handle_t s_bus;
static i2c_master_dev_handle_t s_dev;
static uint8_t s_fb[1 + OLED_W * OLED_H / 8];  // control byte 0x40 + one bit per pixel, page by page

static esp_err_t cmds(const uint8_t *c, size_t n)
{
    uint8_t buf[32];
    buf[0] = 0x00;  // control byte: the rest are commands
    memcpy(buf + 1, c, n);
    return i2c_master_transmit(s_dev, buf, n + 1, 100);
}

static esp_err_t flush(void)
{
    static const uint8_t window[] = {0x21, 0, OLED_W - 1, 0x22, 0, OLED_H / 8 - 1};  // columns, pages
    esp_err_t err = cmds(window, sizeof window);
    if (err == ESP_OK) {
        s_fb[0] = 0x40;  // control byte: the rest is display RAM
        err = i2c_master_transmit(s_dev, s_fb, sizeof s_fb, 200);
    }
    return err;
}

static void pixel(int x, int y)
{
    if (x >= 0 && x < OLED_W && y >= 0 && y < OLED_H) {
        s_fb[1 + x + (y / 8) * OLED_W] |= (uint8_t)(1u << (y % 8));
    }
}

static bool bus_init(int sda, int scl)
{
    if (s_bus) {
        return true;
    }
    i2c_master_bus_config_t cfg = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = sda,
        .scl_io_num = scl,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,  // the module has its own pull-ups; these only help
    };
    return i2c_new_master_bus(&cfg, &s_bus) == ESP_OK;
}

// Devices answering on the bus; returns the SSD1306 address (0x3C/0x3D) or 0
static uint16_t scan(int sda, int scl)
{
    if (s_bus) {
        i2c_del_master_bus(s_bus);
        s_bus = NULL;
    }
    if (!bus_init(sda, scl)) {
        printf("ERR i2c bus init failed\n");
        return 0;
    }
    printf("I2C scan (SDA GPIO %d, SCL GPIO %d):", sda, scl);
    uint16_t addr = 0;
    int found = 0;
    for (uint16_t a = 0x08; a < 0x78; a++) {
        if (i2c_master_probe(s_bus, a, 20) == ESP_OK) {
            printf(" 0x%02x", a);
            found++;
            if (a == 0x3C || a == 0x3D) {
                addr = a;
            }
        }
    }
    printf("%s\n", found ? "" : " nothing");
    return addr;
}

// With the ESP32's ~45 kOhm pull-downs, a line reads 1 only if it reaches the module's own pull-ups
// (a few kOhm to the module's VCC): checks each jumper and the module's supply without the protocol.
static void line_check(void)
{
    if (s_bus) {
        i2c_del_master_bus(s_bus);
        s_bus = NULL;
    }
    const gpio_config_t io = {
        .pin_bit_mask = (1ULL << OLED_SDA_GPIO) | (1ULL << OLED_SCL_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
    };
    gpio_config(&io);
    vTaskDelay(pdMS_TO_TICKS(10));
    const int sda = gpio_get_level(OLED_SDA_GPIO), scl = gpio_get_level(OLED_SCL_GPIO);
    printf("I2C lines with pull-downs: SDA=%d SCL=%d -> %s\n", sda, scl,
           sda && scl  ? "both reach a powered module"
           : sda || scl ? "only one line reaches the module: check the other jumper"
                        : "neither line reaches a powered module: check VCC/GND first, then the jumpers");
    gpio_reset_pin(OLED_SDA_GPIO);
    gpio_reset_pin(OLED_SCL_GPIO);
}

bool oled_test(int sda_pin, int scl_pin)
{
    line_check();
    uint16_t addr = scan(sda_pin, scl_pin);
    if (!addr) {
        // Diagnostics only: the display stays on GPIO 8/9, so a hit here means the wires are crossed
        const uint16_t swapped = scan(scl_pin, sda_pin);
        printf("OLED,found=no%s\n", swapped ? ",answers_with_sda_scl_swapped=yes (swap the SDA and SCL wires)"
                                            : " (check VCC, GND and the SDA/SCL jumpers)");
        return false;
    }

    i2c_device_config_t dev = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = addr,
        .scl_speed_hz = OLED_HZ,
    };
    if (!s_dev && i2c_master_bus_add_device(s_bus, &dev, &s_dev) != ESP_OK) {
        printf("ERR i2c add device failed\n");
        return false;
    }
    static const uint8_t init[] = {
        0xAE,        // display off
        0xD5, 0x80,  // clock divide
        0xA8, 0x3F,  // multiplex: 64 rows
        0xD3, 0x00,  // no display offset
        0x40,        // start line 0
        0x8D, 0x14,  // charge pump on (module powered from 3.3 V)
        0x20, 0x00,  // horizontal addressing
        0xA1, 0xC8,  // segment remap and COM scan direction: pins on top, image upright
        0xDA, 0x12,  // COM pins for 128x64
        0x81, 0xCF,  // contrast
        0xD9, 0xF1,  // pre-charge
        0xDB, 0x40,  // VCOMH
        0xA4, 0xA6,  // show RAM, not inverted
        0xAF,        // display on
    };
    esp_err_t err = cmds(init, sizeof init);

    // 1) all pixels on, timed
    memset(s_fb + 1, 0xFF, sizeof s_fb - 1);
    const int64_t t0 = esp_timer_get_time();
    if (err == ESP_OK) {
        err = flush();
    }
    const int64_t flush_us = esp_timer_get_time() - t0;
    vTaskDelay(pdMS_TO_TICKS(800));
    // 2) checkerboard of 8x8 squares
    for (int i = 0; i < OLED_W * OLED_H / 8; i++) {
        s_fb[1 + i] = (((i % OLED_W) / 8 + i / OLED_W) % 2) ? 0xFF : 0x00;
    }
    if (err == ESP_OK) {
        err = flush();
    }
    vTaskDelay(pdMS_TO_TICKS(800));
    // 3) frame and both diagonals (stays on screen)
    memset(s_fb + 1, 0, sizeof s_fb - 1);
    for (int x = 0; x < OLED_W; x++) {
        pixel(x, 0);
        pixel(x, OLED_H - 1);
        pixel(x, x * (OLED_H - 1) / (OLED_W - 1));
        pixel(x, OLED_H - 1 - x * (OLED_H - 1) / (OLED_W - 1));
    }
    for (int y = 0; y < OLED_H; y++) {
        pixel(0, y);
        pixel(OLED_W - 1, y);
    }
    if (err == ESP_OK) {
        err = flush();
    }
    printf("OLED,addr=0x%02x,ok=%d,flush_us=%" PRId64 ",i2c_hz=%d\n", addr, err == ESP_OK, flush_us, OLED_HZ);
    return err == ESP_OK;
}
