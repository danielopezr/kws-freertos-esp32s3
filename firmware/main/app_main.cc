// Phase 1 skeleton: proves that the toolchain, PSRAM, the MFCC component and
// TensorFlow Lite Micro (esp-tflite-micro) build, link and run on the ESP32-S3,
// then runs the GPIO probe test (plan step 5) to validate the measurement chain
// with the logic analyzer. The real tasks T1-T4 replace the probe test later.

#include <cmath>
#include <cstdio>

#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "kws_mfcc.h"
#include "probes.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace {

// Every op used by the DNN, CNN and DS-CNN models
tflite::MicroMutableOpResolver<8> g_resolver;

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

void check_tflm()
{
    g_resolver.AddConv2D();
    g_resolver.AddDepthwiseConv2D();
    g_resolver.AddFullyConnected();
    g_resolver.AddMean();
    g_resolver.AddAveragePool2D();
    g_resolver.AddReshape();
    g_resolver.AddQuantize();
    g_resolver.AddDequantize();
    printf("TFLM op resolver ready\n");
}

void check_mfcc()
{
    static int16_t tone[KWS_MFCC_FRAME_LEN];
    static float coeffs[KWS_MFCC_N_COEFFS];
    for (int i = 0; i < KWS_MFCC_FRAME_LEN; i++) {
        tone[i] = (int16_t)(10000.0f * sinf(2.0f * 3.14159265f * 1000.0f * i / KWS_MFCC_SAMPLE_RATE));
    }
    kws_mfcc_init();
    const int64_t t0 = esp_timer_get_time();
    kws_mfcc_frame(tone, coeffs);
    const int64_t t1 = esp_timer_get_time();
    printf("MFCC 1 kHz tone: c0=%.4f c1=%.4f | %lld us per frame\n", coeffs[0], coeffs[1], (long long)(t1 - t0));
}

}  // namespace

extern "C" void app_main(void)
{
    report_system();
    check_tflm();
    check_mfcc();
    printf("skeleton OK\n");

    probes_init();
    probes_start_test();
    printf("probe test running: T1 %d ms/%d us, T2 %d ms/%d us, T3 %d ms/%d us, T4 %d ms/%d us\n",
           PROBE_T1_PERIOD_MS, PROBE_T1_BUSY_US, PROBE_T2_PERIOD_MS, PROBE_T2_BUSY_US,
           PROBE_T3_PERIOD_MS, PROBE_T3_BUSY_US, PROBE_T4_PERIOD_MS, PROBE_T4_BUSY_US);
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        probes_report();
    }
}
