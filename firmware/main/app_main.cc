// Board side of the experiments. After boot it prints READY and waits for one command line:
//
//   CFG ...          one experiment run (experiments/protocol.py): OK, DET/TASK/RESULT lines, DONE
//   EVAL model=cnn [mem=psram]   offline accuracy check on the 150 clips of the test sequence
//   ARENA            tensor arena each model really needs
//   PROBE            GPIO probe test for the logic analyzer (plan step 5), until reset
//
// runner.py resets the board before every run, so each run starts from a fresh boot.

#include <cstdio>
#include <cstring>

#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "kws_mfcc.h"
#include "kws_models.h"
#include "pipeline.h"
#include "probes.h"
#include "run_config.h"

namespace {

constexpr uart_port_t kUart = UART_NUM_0;  // console, wired to the USB-UART bridge (CH343)

void report_system()
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    printf("ESP-IDF %s | cores=%d rev=%d\n", esp_get_idf_version(), chip.cores, chip.revision);
    printf("free internal=%u B (largest block %u B) | free PSRAM=%u B\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

// Blocking read of one line (without the line ending)
void read_line(char *buf, size_t len)
{
    size_t n = 0;
    for (;;) {
        uint8_t c;
        if (uart_read_bytes(kUart, &c, 1, portMAX_DELAY) != 1) {
            continue;
        }
        if (c == '\n' || c == '\r') {
            if (n == 0) {
                continue;  // empty line or the second half of CRLF
            }
            buf[n] = '\0';
            return;
        }
        if (n + 1 < len) {
            buf[n++] = (char)c;
        }
    }
}

// Value of key=... in a command line, or nullptr
const char *arg(const char *line, const char *key, char *out, size_t len)
{
    const size_t k = strlen(key);
    for (const char *p = strstr(line, key); p; p = strstr(p + 1, key)) {
        if ((p == line || p[-1] == ' ') && p[k] == '=') {
            size_t n = strcspn(p + k + 1, " \t");
            n = n < len - 1 ? n : len - 1;
            memcpy(out, p + k + 1, n);
            out[n] = '\0';
            return out;
        }
    }
    return nullptr;
}

}  // namespace

extern "C" void app_main(void)
{
    ESP_ERROR_CHECK(uart_driver_install(kUart, 2048, 4096, 0, nullptr, 0));
    uart_vfs_dev_use_driver(kUart);
    // Some treatments keep a core 100 % busy on purpose (e.g. T3 above T1 plus the load on core 0):
    // turn the task watchdog off (it only watches the idle tasks) so it never prints during a run
    esp_task_wdt_deinit();
    probes_init();
    kws_mfcc_init();
    report_system();
    const bool seq_ok = pipeline_init();
    printf("sequence partition: %s\n", seq_ok ? "mapped" : "MISSING (flash test_sequence/out/sequence.raw)");

    char line[256], err[96], val[16];
    for (;;) {
        printf("READY\n");
        fflush(stdout);
        read_line(line, sizeof line);

        if (strncmp(line, "CFG", 3) == 0) {
            kws_cfg_t cfg;
            if (!kws_parse_cfg(line, &cfg, err, sizeof err)) {
                printf("ERR %s\n", err);
                continue;
            }
            pipeline_run(&cfg);
        } else if (strncmp(line, "EVAL", 4) == 0) {
            const int model = kws_model_by_name(arg(line, "model", val, sizeof val) ? val : "cnn");
            const bool psram = arg(line, "mem", val, sizeof val) && strcmp(val, "psram") == 0;
            if (model < 0) {
                printf("ERR unknown model\n");
                continue;
            }
            pipeline_eval(model, psram);
        } else if (strcmp(line, "ARENA") == 0) {
            pipeline_arena();
        } else if (strcmp(line, "PROBE") == 0) {
            probes_start_test();
            printf("probe test running: T1 %d ms/%d us, T2 %d ms/%d us, T3 %d ms/%d us, T4 %d ms/%d us\n",
                   PROBE_T1_PERIOD_MS, PROBE_T1_BUSY_US, PROBE_T2_PERIOD_MS, PROBE_T2_BUSY_US, PROBE_T3_PERIOD_MS,
                   PROBE_T3_BUSY_US, PROBE_T4_PERIOD_MS, PROBE_T4_BUSY_US);
            for (;;) {
                vTaskDelay(pdMS_TO_TICKS(5000));
                probes_report();
            }
        } else {
            printf("ERR unknown command: %.32s\n", line);
        }
    }
}
