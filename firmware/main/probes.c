#include "probes.h"

#include <inttypes.h>
#include <stdio.h>

#include "driver/gpio.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

typedef struct {
    int64_t period_sum, period_min, period_max;
    int64_t width_sum, width_min, width_max;
    uint32_t periods, pulses, overruns;
} probe_stats_t;

typedef struct {
    const char *name;
    gpio_num_t pin;
    uint32_t period_ms, busy_us;
    UBaseType_t priority;
    BaseType_t core;
    int64_t last_start;
    probe_stats_t stats;
} probe_task_t;

// Priorities follow the real design: capture above features above inference above logger.
static probe_task_t s_tasks[] = {
#define PROBE_TASK(n, p, c)     {.name = #n, .pin = PROBE_##n##_GPIO, .period_ms = PROBE_##n##_PERIOD_MS, .busy_us = PROBE_##n##_BUSY_US,      .priority = (p), .core = (c)}
    PROBE_TASK(T1, 5, 0),
    PROBE_TASK(T2, 4, 0),
    PROBE_TASK(T3, 3, 1),
    PROBE_TASK(T4, 1, 1),
};
#define N_TASKS (sizeof(s_tasks) / sizeof(s_tasks[0]))

static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;

static void stats_reset(probe_stats_t *s)
{
    *s = (probe_stats_t){.period_min = INT64_MAX, .width_min = INT64_MAX};
}

static void record(probe_task_t *t, int64_t start, int64_t width, bool overrun)
{
    portENTER_CRITICAL(&s_mux);
    probe_stats_t *s = &t->stats;
    if (t->last_start) {
        const int64_t p = start - t->last_start;
        s->period_sum += p;
        s->period_min = p < s->period_min ? p : s->period_min;
        s->period_max = p > s->period_max ? p : s->period_max;
        s->periods++;
    }
    s->width_sum += width;
    s->width_min = width < s->width_min ? width : s->width_min;
    s->width_max = width > s->width_max ? width : s->width_max;
    s->pulses++;
    s->overruns += overrun;
    t->last_start = start;
    portEXIT_CRITICAL(&s_mux);
}

static void probe_task(void *arg)
{
    probe_task_t *t = (probe_task_t *)arg;
    TickType_t wake = xTaskGetTickCount();
    for (;;) {
        const int64_t start = esp_timer_get_time();
        gpio_set_level(t->pin, 1);
        esp_rom_delay_us(t->busy_us);  // busy wait: stands in for real work of known length
        gpio_set_level(t->pin, 0);
        const int64_t width = esp_timer_get_time() - start;
        const bool overrun = xTaskDelayUntil(&wake, pdMS_TO_TICKS(t->period_ms)) == pdFALSE;
        record(t, start, width, overrun);
    }
}

// Marker pulse once per PROBE_MARK_PERIOD_MS and LED toggle every half LED period.
static void mark_task(void *arg)
{
    (void)arg;
    const uint32_t half_led = PROBE_LED_PERIOD_MS / 2;
    TickType_t wake = xTaskGetTickCount();
    uint32_t elapsed = 0;
    int led = 0;
    for (;;) {
        if (elapsed % PROBE_MARK_PERIOD_MS == 0) {
            gpio_set_level(PROBE_MARK_GPIO, 1);
            esp_rom_delay_us(PROBE_MARK_BUSY_US);
            gpio_set_level(PROBE_MARK_GPIO, 0);
        }
        led = !led;
        gpio_set_level(PROBE_LED_GPIO, led);
        xTaskDelayUntil(&wake, pdMS_TO_TICKS(half_led));
        elapsed += half_led;
    }
}

void probes_init(void)
{
    const gpio_config_t cfg = {
        .pin_bit_mask = (1ULL << PROBE_T1_GPIO) | (1ULL << PROBE_T2_GPIO) | (1ULL << PROBE_T3_GPIO) |
                        (1ULL << PROBE_T4_GPIO) | (1ULL << PROBE_MARK_GPIO) | (1ULL << PROBE_LED_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));
    for (int pin = 0; pin < 64; pin++) {
        if (cfg.pin_bit_mask & (1ULL << pin)) {
            gpio_set_level((gpio_num_t)pin, 0);
        }
    }
}

void probes_start_test(void)
{
    for (size_t i = 0; i < N_TASKS; i++) {
        stats_reset(&s_tasks[i].stats);
        xTaskCreatePinnedToCore(probe_task, s_tasks[i].name, 3072, &s_tasks[i], s_tasks[i].priority, NULL,
                                s_tasks[i].core);
    }
    // Priority 2 on core 1, below T3: created on the same tick as T3, so its 100 us pulse
    // normally runs right after T3's and is not stretched by it
    xTaskCreatePinnedToCore(mark_task, "MARK", 2048, NULL, 2, NULL, 1);
}

void probes_report(void)
{
    printf("PROBES task  period_ms(mean/min/max)        width_us(mean/min/max)  pulses overruns\n");
    for (size_t i = 0; i < N_TASKS; i++) {
        probe_stats_t s;
        portENTER_CRITICAL(&s_mux);
        s = s_tasks[i].stats;
        stats_reset(&s_tasks[i].stats);
        portEXIT_CRITICAL(&s_mux);
        if (!s.periods || !s.pulses) {
            printf("PROBES %-4s no data yet\n", s_tasks[i].name);
            continue;
        }
        printf("PROBES %-4s %8.3f %8.3f %8.3f   %8.1f %6" PRId64 " %6" PRId64 "   %6" PRIu32 " %8" PRIu32 "\n",
               s_tasks[i].name, s.period_sum / 1000.0 / s.periods, s.period_min / 1000.0, s.period_max / 1000.0,
               (double)s.width_sum / s.pulses, s.width_min, s.width_max, s.pulses, s.overruns);
    }
}
