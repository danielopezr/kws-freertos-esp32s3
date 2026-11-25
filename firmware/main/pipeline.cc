// Real-time keyword spotting pipeline:
//
//   I2S DMA, 20 ms blocks --> T1 capture --queue--> T2 features: one MFCC frame per block
//   --> ring of the last 49 frames (1 s) --> T3 inference every 5 frames (100 ms) + decision
//   T4 logger (low priority): heap watermark and the detection LED.  LOAD: synthetic load.
//
// Playback (seq=full): the I2S peripheral runs exactly as with the microphone, so the DMA, its
// interrupts and the 20 ms pacing are real, but T1 replaces every block with the matching 20 ms
// of the test sequence (memory-mapped flash). A block the DMA drops (T1 too late) is skipped in
// the sequence too, as it would be lost from a real microphone. Every run plays the same audio.
//
// Timing is measured on the board: esp_timer (1 us) for inference and work times, FreeRTOS
// run-time stats for CPU use, and the detection instant expressed as a sample index of the
// sequence (analysis/scoring.py turns it into the detection latency). The GPIO probes of
// probes.h mirror the same events for the logic analyzer.

#include "pipeline.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "kws_infer.h"
#include "kws_mfcc.h"
#include "kws_models.h"
#include "kws_sequence_labels.h"
#include "probes.h"

namespace {

// ---- fixed design parameters (the same for every run) ----
constexpr int kBlock = KWS_MFCC_HOP;                 // 320 samples = 20 ms
constexpr int kFrames = KWS_MFCC_N_FRAMES;           // 49 frames = 1 s window
constexpr int kCoeffs = KWS_MFCC_N_COEFFS;           // 40
constexpr int kFeat = kFrames * kCoeffs;
constexpr int kInferEvery = 5;                       // T3 every 5 frames = 100 ms
constexpr int kQueueDepth = 8;                       // T1 -> T2: up to 160 ms of audio in flight
constexpr int kPoolBlocks = kQueueDepth + 3;         // static pool: queue + T2's two blocks + T1's
constexpr int kDmaDesc = 4;                          // I2S DMA: 4 buffers of one block = 80 ms
constexpr int kAvg = 3;                              // decision: mean of the last 3 outputs (300 ms)
constexpr int64_t kRefractoryUs = 1000 * 1000;       // no second detection within 1 s
constexpr uint32_t kLedOnMs = 200;
constexpr int kMaxInferences = 3200;                 // 300 s x 10 per second
constexpr int kMaxDets = 600;
constexpr int64_t kLoadBusyUs = 10 * 1000;           // synthetic load: 10 ms busy every 20 ms
constexpr uint32_t kLoadPeriodMs = 20;
constexpr uint32_t kT4PeriodMs = 100;

// Priorities (configMAX_PRIORITIES = 25; esp_timer runs at 22 and IPC at 24, both above these)
constexpr UBaseType_t kPrioT1 = 10, kPrioT2 = 9, kPrioLoad = 9, kPrioT4 = 3;
UBaseType_t prio_t3(int prio)
{
    return prio == PRIO_LOWER ? kPrioT1 - 2 : prio == PRIO_EQUAL ? kPrioT1 : kPrioT1 + 1;
}

constexpr uint32_t kStackT1 = 4096, kStackT2 = 4096, kStackT3 = 10240, kStackT4 = 3072, kStackLoad = 2048;

// INMP441 (plan step 5); unused pins are harmless in playback mode
constexpr gpio_num_t kI2sBclk = GPIO_NUM_12, kI2sWs = GPIO_NUM_11, kI2sDin = GPIO_NUM_10;

enum : EventBits_t { BIT_T1 = 1, BIT_T2 = 2, BIT_T3 = 4, BIT_T4 = 8, BIT_LOAD = 16 };

struct det_t {
    uint32_t sample;
    uint8_t label, score;
};

// Response time of each activation of a task (start of its work to end, preemption included)
struct work_t {
    int64_t sum_us, max_us;
    uint32_t n;
    void add(int64_t us)
    {
        sum_us += us;
        n++;
        max_us = us > max_us ? us : max_us;
    }
};

struct run_t {
    kws_cfg_t cfg;
    uint32_t caps;          // heap caps of the buffers and the arena (mem factor)
    uint32_t total_blocks;  // blocks to play; 0 = endless (demo)
    // T1
    i2s_chan_handle_t rx;
    QueueHandle_t q;
    int16_t *pool;
    uint32_t pool_next;
    std::atomic<uint32_t> dma_ovf;  // DMA buffers dropped because T1 did not read in time
    uint32_t blocks_rx, ring_ovf, run_alloc_fail;
    int64_t t_first, t1_gap_max;
    // T2
    float *feat;  // ring of kFrames MFCC frames
    uint32_t frames;
    SemaphoreHandle_t feat_mx;
    // T3
    kws_net_t net;
    float *staging;
    uint32_t n_inf, inf_skipped, inf_fail;
    float hist[kAvg][KWS_N_CLASSES];
    uint32_t hist_n, n_det;
    int64_t last_det_us;
    volatile uint32_t led_on_ms;
    // T4
    size_t largest_min;
    // control
    TaskHandle_t t1, t2, t3, t4, load;
    volatile bool stop, stop_aux;
    EventGroupHandle_t done;
    work_t w1, w2, w3, w4, wl;
};

alignas(run_t) uint8_t g_run_mem[sizeof(run_t)];
uint32_t g_inf_us[kMaxInferences];  // metrics live in internal RAM whatever the mem factor
det_t g_dets[kMaxDets];
volatile uint32_t g_sink;

const int16_t *g_seq = nullptr;  // memory-mapped test sequence

// Task stacks and TCBs for alloc=static (internal RAM, reserved at link time)
StackType_t g_stack_t1[kStackT1], g_stack_t2[kStackT2], g_stack_t3[kStackT3], g_stack_t4[kStackT4],
    g_stack_load[kStackLoad];
StaticTask_t g_tcb[5];

inline int64_t now_us()
{
    return esp_timer_get_time();
}

// ---------------------------------------------------------------- heap fragmentation (experiment C)
// Fill internal SRAM with 8 KB blocks, then free them in runs, keeping one block out of every
// `period`: the free memory stays large but its largest contiguous piece shrinks.
//   frag=1 (medium): runs of 12 blocks -> holes of ~96 KB      frag=2 (high): runs of 4 -> ~32 KB
constexpr size_t kFragBlock = 8 * 1024;
constexpr int kFragMax = 96;
void *g_frag[kFragMax];
int g_frag_n = 0;

void frag_apply(int level)
{
    g_frag_n = 0;
    if (level == 0) {
        return;
    }
    const int period = level == 1 ? 13 : 5;
    while (g_frag_n < kFragMax) {
        void *p = heap_caps_malloc(kFragBlock, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!p) {
            break;
        }
        g_frag[g_frag_n++] = p;
    }
    for (int i = 0; i < g_frag_n; i++) {
        if (i % period != period - 1) {
            heap_caps_free(g_frag[i]);
            g_frag[i] = nullptr;
        }
    }
}

void frag_release()
{
    for (int i = 0; i < g_frag_n; i++) {
        heap_caps_free(g_frag[i]);
        g_frag[i] = nullptr;
    }
    g_frag_n = 0;
}

// ---------------------------------------------------------------- tasks
bool IRAM_ATTR on_rx_overflow(i2s_chan_handle_t, i2s_event_data_t *, void *ctx)
{
    static_cast<run_t *>(ctx)->dma_ovf.fetch_add(1, std::memory_order_relaxed);
    return false;
}

void finish(run_t &r, EventBits_t bit)
{
    xEventGroupSetBits(r.done, bit);
    vTaskSuspend(nullptr);  // the run collects its stack watermark, then deletes it
}

// ---------------------------------------------------------------- playback timeline
// Every word of the sequence ends on the 20 ms frame grid, so one common delay per run would move
// all of them against the 100 ms inference grid at once. Instead, d_i samples of silence (0..1599,
// drawn from the run seed) are inserted 100 ms before the slot of each utterance i: every word
// meets the inference grid at its own phase, as real speech would, the words and the audio around
// them stay intact, and the latency of a word is the same in stream and in sequence samples.
constexpr int kItems = KWS_SEQ_N_ITEMS;
constexpr uint32_t kJitterMax = kInferEvery * kBlock;   // 1600 samples = 100 ms
constexpr uint32_t kJitterLead = kInferEvery * kBlock;  // insertion point: 100 ms before the slot

struct timeline_t {
    uint32_t at[kItems];   // insertion point, in sequence samples
    uint32_t len[kItems];  // inserted silence, samples
    uint32_t cum[kItems];  // silence inserted up to and including insertion i
    uint32_t stream_at(int i) const { return at[i] + (i ? cum[i - 1] : 0); }
};
timeline_t g_tl;

void timeline_build(uint32_t seed)
{
    uint32_t x = seed ? seed : 0x9e3779b9u, sum = 0;
    for (int i = 0; i < kItems; i++) {
        x ^= x << 13;  // xorshift32
        x ^= x >> 17;
        x ^= x << 5;
        g_tl.at[i] = kws_seq_items[i].start - kJitterLead;
        g_tl.len[i] = x % kJitterMax;
        sum += g_tl.len[i];
        g_tl.cum[i] = sum;
    }
}

// Last insertion that starts at or before stream sample s, or -1
int insertion_before(uint32_t s)
{
    int lo = 0, hi = kItems - 1, j = -1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (g_tl.stream_at(mid) <= s) {
            j = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return j;
}

// Stream sample -> sequence sample; inside an inserted silence, its insertion point
uint32_t stream_to_seq(uint32_t s)
{
    const int j = insertion_before(s);
    if (j < 0) {
        return s;
    }
    if (s < g_tl.stream_at(j) + g_tl.len[j]) {
        return g_tl.at[j];
    }
    return s - g_tl.cum[j];
}

// Stream block [s0, s0 + kBlock): the sequence with the inserted silences; silence after its end
void copy_stream(int16_t *dst, uint32_t s0)
{
    int i = 0;
    while (i < kBlock) {
        const uint32_t s = s0 + i;
        const int j = insertion_before(s);
        int n;
        if (j >= 0 && s < g_tl.stream_at(j) + g_tl.len[j]) {
            n = (int)std::min<uint32_t>(kBlock - i, g_tl.stream_at(j) + g_tl.len[j] - s);
            memset(dst + i, 0, n * sizeof(int16_t));
        } else {
            const uint32_t next = j + 1 < kItems ? g_tl.stream_at(j + 1) : UINT32_MAX;
            n = (int)std::min<uint32_t>(kBlock - i, next - s);
            const uint32_t q = s - (j >= 0 ? g_tl.cum[j] : 0);
            const int avail = q < KWS_SEQ_TOTAL_SAMPLES ? (int)std::min<uint32_t>(n, KWS_SEQ_TOTAL_SAMPLES - q) : 0;
            memcpy(dst + i, g_seq + q, avail * sizeof(int16_t));
            memset(dst + i + avail, 0, (n - avail) * sizeof(int16_t));
        }
        i += n;
    }
}

// T1: wait for each 20 ms DMA block, put the audio (or the sequence) in the queue to T2
void t1_capture(void *arg)
{
    run_t &r = *static_cast<run_t *>(arg);
    static int32_t dma[kBlock];  // one block of 32-bit I2S slots
    uint32_t next_item = 0;
    int64_t prev = 0;
    for (;;) {
        size_t got = 0;
        if (i2s_channel_read(r.rx, dma, sizeof dma, &got, 1000) != ESP_OK || got != sizeof dma) {
            continue;
        }
        const int64_t t0 = now_us();
        gpio_set_level((gpio_num_t)PROBE_T1_GPIO, 1);
        if (!r.t_first) {
            r.t_first = t0;
        }
        if (prev && t0 - prev > r.t1_gap_max) {
            r.t1_gap_max = t0 - prev;
        }
        prev = t0;
        const uint32_t idx = r.blocks_rx + r.dma_ovf.load(std::memory_order_relaxed);  // stream position
        r.blocks_rx++;
        if (r.total_blocks && idx >= r.total_blocks) {
            gpio_set_level((gpio_num_t)PROBE_T1_GPIO, 0);
            break;
        }
        // End-of-word marker: high while T1 handles the block that holds a voiced_end
        bool mark = false;
        if (r.total_blocks) {
            const uint32_t lo = idx * kBlock, hi = lo + kBlock;  // stream samples
            // utterance i is played g_tl.cum[i] samples late (its own silence was inserted before it)
            while (next_item < kItems && kws_seq_items[next_item].voiced_end + g_tl.cum[next_item] < lo) {
                next_item++;
            }
            mark = next_item < kItems && kws_seq_items[next_item].voiced_end + g_tl.cum[next_item] < hi;
            if (mark) {
                gpio_set_level((gpio_num_t)PROBE_MARK_GPIO, 1);
            }
        }
        if (uxQueueSpacesAvailable(r.q) == 0) {
            r.ring_ovf++;  // T2 is behind: this block is lost
        } else {
            int16_t *dst = r.cfg.dynamic_alloc
                               ? static_cast<int16_t *>(heap_caps_malloc(kBlock * sizeof(int16_t), r.caps))
                               : r.pool + (r.pool_next++ % kPoolBlocks) * kBlock;
            if (!dst) {
                r.run_alloc_fail++;
                r.ring_ovf++;
            } else {
                if (r.total_blocks) {
                    copy_stream(dst, idx * kBlock);
                } else {
                    for (int i = 0; i < kBlock; i++) {
                        dst[i] = (int16_t)(dma[i] >> 16);  // INMP441: 24-bit sample, MSB aligned in 32
                    }
                }
                xQueueSend(r.q, &dst, 0);
            }
        }
        if (mark) {
            gpio_set_level((gpio_num_t)PROBE_MARK_GPIO, 0);
        }
        gpio_set_level((gpio_num_t)PROBE_T1_GPIO, 0);
        r.w1.add(now_us() - t0);
    }
    int16_t *end = nullptr;
    xQueueSend(r.q, &end, portMAX_DELAY);
    finish(r, BIT_T1);
}

// T2: one MFCC frame (previous + current block = 40 ms) per block into the feature ring
void t2_features(void *arg)
{
    run_t &r = *static_cast<run_t *>(arg);
    static int16_t frame[KWS_MFCC_FRAME_LEN];
    static float coeffs[kCoeffs];
    int16_t *prev = nullptr;
    for (;;) {
        int16_t *cur = nullptr;
        xQueueReceive(r.q, &cur, portMAX_DELAY);
        if (!cur) {
            break;
        }
        const int64_t t0 = now_us();
        gpio_set_level((gpio_num_t)PROBE_T2_GPIO, 1);
        if (prev) {
            memcpy(frame, prev, kBlock * sizeof(int16_t));
            memcpy(frame + kBlock, cur, kBlock * sizeof(int16_t));
            kws_mfcc_frame(frame, coeffs);
            xSemaphoreTake(r.feat_mx, portMAX_DELAY);
            memcpy(r.feat + (r.frames % kFrames) * kCoeffs, coeffs, sizeof coeffs);
            const uint32_t f = ++r.frames;
            xSemaphoreGive(r.feat_mx);
            if (f >= (uint32_t)kFrames && f % kInferEvery == 0) {
                xTaskNotifyGive(r.t3);
            }
            if (r.cfg.dynamic_alloc) {
                heap_caps_free(prev);
            }
        }
        prev = cur;
        gpio_set_level((gpio_num_t)PROBE_T2_GPIO, 0);
        r.w2.add(now_us() - t0);
    }
    if (r.cfg.dynamic_alloc && prev) {
        heap_caps_free(prev);
    }
    r.stop = true;
    xTaskNotifyGive(r.t3);
    finish(r, BIT_T2);
}

// T3: every 100 ms, model on the last 1 s of features; mean of the last 3 outputs; detection
void t3_inference(void *arg)
{
    run_t &r = *static_cast<run_t *>(arg);
    for (;;) {
        const uint32_t pending = ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (r.stop) {
            break;
        }
        if (pending > 1) {
            r.inf_skipped += pending - 1;  // T3 fell behind: those windows are never classified
        }
        const int64_t t0 = now_us();
        gpio_set_level((gpio_num_t)PROBE_T3_GPIO, 1);
        float *stg = r.cfg.dynamic_alloc ? static_cast<float *>(heap_caps_malloc(kFeat * sizeof(float), r.caps))
                                         : r.staging;
        if (!stg) {
            r.run_alloc_fail++;
            gpio_set_level((gpio_num_t)PROBE_T3_GPIO, 0);
            continue;
        }
        xSemaphoreTake(r.feat_mx, portMAX_DELAY);
        const uint32_t oldest = r.frames % kFrames;  // ring slot of the oldest frame
        const size_t tail = (size_t)(kFrames - oldest) * kCoeffs;
        memcpy(stg, r.feat + oldest * kCoeffs, tail * sizeof(float));
        memcpy(stg + tail, r.feat, (size_t)oldest * kCoeffs * sizeof(float));
        xSemaphoreGive(r.feat_mx);
        kws_net_set_input(&r.net, stg);
        if (r.cfg.dynamic_alloc) {
            heap_caps_free(stg);
        }

        const int64_t ti = now_us();
        const bool ok = kws_net_invoke(&r.net);
        const int64_t inf = now_us() - ti;
        if (!ok) {
            r.inf_fail++;
        } else {
            if (r.n_inf < (uint32_t)kMaxInferences) {
                g_inf_us[r.n_inf] = (uint32_t)inf;
            }
            r.n_inf++;
            kws_net_probs(&r.net, r.hist[r.hist_n % kAvg]);
            r.hist_n++;
            const uint32_t m = r.hist_n < (uint32_t)kAvg ? r.hist_n : kAvg;
            float avg[KWS_N_CLASSES] = {};
            for (uint32_t k = 0; k < m; k++) {
                for (int c = 0; c < KWS_N_CLASSES; c++) {
                    avg[c] += r.hist[k][c] / m;
                }
            }
            int top = 0;
            for (int c = 1; c < KWS_N_CLASSES; c++) {
                top = avg[c] > avg[top] ? c : top;
            }
            const int64_t t_det = now_us();
            if (top >= KWS_FIRST_KEYWORD && avg[top] >= r.cfg.threshold &&
                (!r.last_det_us || t_det - r.last_det_us >= kRefractoryUs)) {
                r.last_det_us = t_det;
                // Instant of the decision as a sample of the sequence: stream block 0 ended at t_first
                const int64_t stream = kBlock + (t_det - r.t_first) * KWS_SEQ_SAMPLE_RATE / 1000000;
                const uint32_t sample = r.total_blocks ? stream_to_seq((uint32_t)stream) : (uint32_t)stream;
                const uint8_t score = (uint8_t)lrintf(avg[top] * 255.0f);
                gpio_set_level((gpio_num_t)PROBE_LED_GPIO, 1);
                r.led_on_ms = (uint32_t)(t_det / 1000) | 1u;
                if (r.cfg.demo) {
                    printf("DET,%" PRIu32 ",%d,%u  %s %.2f\n", sample, top, score, kws_seq_label_names[top], avg[top]);
                } else if (r.n_det < (uint32_t)kMaxDets) {
                    g_dets[r.n_det++] = det_t{sample, (uint8_t)top, score};
                }
            }
        }
        gpio_set_level((gpio_num_t)PROBE_T3_GPIO, 0);
        r.w3.add(now_us() - t0);
    }
    finish(r, BIT_T3);
}

// T4: low-priority bookkeeping every 100 ms
void t4_logger(void *arg)
{
    run_t &r = *static_cast<run_t *>(arg);
    TickType_t wake = xTaskGetTickCount();
    while (!r.stop_aux) {
        const int64_t t0 = now_us();
        gpio_set_level((gpio_num_t)PROBE_T4_GPIO, 1);
        const size_t largest = heap_caps_get_largest_free_block(r.caps);
        r.largest_min = largest < r.largest_min ? largest : r.largest_min;
        const uint32_t on = r.led_on_ms;
        if (on && (uint32_t)(t0 / 1000) - on >= kLedOnMs) {
            gpio_set_level((gpio_num_t)PROBE_LED_GPIO, 0);
            r.led_on_ms = 0;
        }
        gpio_set_level((gpio_num_t)PROBE_T4_GPIO, 0);
        r.w4.add(now_us() - t0);
        xTaskDelayUntil(&wake, pdMS_TO_TICKS(kT4PeriodMs));
    }
    finish(r, BIT_T4);
}

// LOAD (load=on): busy 10 ms out of every 20 ms
void load_task(void *arg)
{
    run_t &r = *static_cast<run_t *>(arg);
    TickType_t wake = xTaskGetTickCount();
    uint32_t x = 0x12345678u;
    while (!r.stop_aux) {
        const int64_t t0 = now_us();
        while (now_us() - t0 < kLoadBusyUs) {
            for (int i = 0; i < 64; i++) {
                x = x * 1664525u + 1013904223u;
            }
        }
        g_sink = x;
        r.wl.add(now_us() - t0);
        xTaskDelayUntil(&wake, pdMS_TO_TICKS(kLoadPeriodMs));
    }
    finish(r, BIT_LOAD);
}

TaskHandle_t spawn(const run_t &r, TaskFunction_t fn, const char *name, uint32_t stack, StackType_t *buf,
                   StaticTask_t *tcb, UBaseType_t prio, BaseType_t core)
{
    TaskHandle_t h = nullptr;
    if (r.cfg.dynamic_alloc) {
        xTaskCreatePinnedToCore(fn, name, stack, (void *)&r, prio, &h, core);
    } else {
        h = xTaskCreateStaticPinnedToCore(fn, name, stack, (void *)&r, prio, buf, tcb, core);
    }
    return h;
}

bool i2s_setup(run_t &r)
{
    i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    cc.dma_desc_num = kDmaDesc;
    cc.dma_frame_num = kBlock;
    if (i2s_new_channel(&cc, nullptr, &r.rx) != ESP_OK) {
        return false;
    }
    const i2s_std_clk_config_t clk = I2S_STD_CLK_DEFAULT_CONFIG(KWS_SEQ_SAMPLE_RATE);
    const i2s_std_slot_config_t slot = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
    i2s_std_config_t sc = {};
    sc.clk_cfg = clk;
    sc.slot_cfg = slot;
    sc.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;  // INMP441 with L/R tied to GND
    sc.gpio_cfg.mclk = I2S_GPIO_UNUSED;
    sc.gpio_cfg.bclk = kI2sBclk;
    sc.gpio_cfg.ws = kI2sWs;
    sc.gpio_cfg.dout = I2S_GPIO_UNUSED;
    sc.gpio_cfg.din = kI2sDin;
    if (i2s_channel_init_std_mode(r.rx, &sc) != ESP_OK) {
        return false;
    }
    i2s_event_callbacks_t cbs = {};
    cbs.on_recv_q_ovf = on_rx_overflow;
    return i2s_channel_register_event_callback(r.rx, &cbs, &r) == ESP_OK;
}

// ---------------------------------------------------------------- reporting
void print_inference_stats(uint32_t n)
{
    n = n < (uint32_t)kMaxInferences ? n : kMaxInferences;
    if (!n) {
        return;
    }
    double sum = 0, sq = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum += g_inf_us[i];
    }
    const double mean = sum / n;
    for (uint32_t i = 0; i < n; i++) {
        sq += (g_inf_us[i] - mean) * (g_inf_us[i] - mean);
    }
    std::sort(g_inf_us, g_inf_us + n);
    const uint32_t p95 = g_inf_us[std::min<uint32_t>(n - 1, (uint32_t)ceil(0.95 * n) - 1)];
    printf(",inf_mean_us=%.0f,inf_sd_us=%.0f,inf_p95_us=%" PRIu32 ",inf_max_us=%" PRIu32, mean, sqrt(sq / n), p95,
           g_inf_us[n - 1]);
}

void print_task(const char *name, TaskHandle_t h, const work_t &w, int64_t run_us)
{
    if (!h) {
        return;
    }
    TaskStatus_t st;
    vTaskGetInfo(h, &st, pdTRUE, eInvalid);
    const double cpu = run_us > 0 ? 100.0 * st.ulRunTimeCounter / run_us : 0.0;
    printf("TASK,%s,stack_free_min=%u,cpu_pct=%.2f,activations=%" PRIu32 ",resp_mean_us=%.0f,resp_max_us=%lld\n", name,
           (unsigned)st.usStackHighWaterMark, cpu, w.n, w.n ? (double)w.sum_us / w.n : 0.0, (long long)w.max_us);
}

void print_heap_fields(size_t largest_before)
{
    printf(",heap_free_min=%u,heap_largest_block=%u,psram_free_min=%u",
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL), (unsigned)largest_before,
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
}

void release(run_t &r)
{
    TaskHandle_t *tasks[] = {&r.t1, &r.t2, &r.t3, &r.t4, &r.load};
    for (TaskHandle_t *t : tasks) {
        if (*t) {
            vTaskDelete(*t);
            *t = nullptr;
        }
    }
    if (r.rx) {
        i2s_del_channel(r.rx);
    }
    heap_caps_free(r.feat);
    heap_caps_free(r.pool);
    heap_caps_free(r.staging);
    if (r.q) {
        // dynamic alloc: blocks still queued when the run stopped
        int16_t *b;
        while (xQueueReceive(r.q, &b, 0) == pdTRUE) {
            if (r.cfg.dynamic_alloc && b) {
                heap_caps_free(b);
            }
        }
        vQueueDelete(r.q);
    }
    if (r.feat_mx) {
        vSemaphoreDelete(r.feat_mx);
    }
    if (r.done) {
        vEventGroupDelete(r.done);
    }
    kws_net_free(&r.net);
    frag_release();
    heap_caps_monitor_local_minimum_free_size_stop();
    gpio_set_level((gpio_num_t)PROBE_LED_GPIO, 0);
}

}  // namespace

// ---------------------------------------------------------------- public
bool pipeline_init(void)
{
    const esp_partition_t *p =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "sequence");
    const size_t bytes = (size_t)KWS_SEQ_TOTAL_SAMPLES * sizeof(int16_t);
    if (!p || p->size < bytes) {
        return false;
    }
    const void *ptr = nullptr;
    esp_partition_mmap_handle_t handle;
    if (esp_partition_mmap(p, 0, bytes, ESP_PARTITION_MMAP_DATA, &ptr, &handle) != ESP_OK) {
        return false;
    }
    g_seq = static_cast<const int16_t *>(ptr);
    return true;
}

void pipeline_run(const kws_cfg_t *cfg)
{
    const bool play = cfg->seq_full && !cfg->demo;
    if (play && !g_seq) {
        printf("ERR no sequence partition\n");
        return;
    }
    printf("OK\n");
    fflush(stdout);

    run_t &r = *new (g_run_mem) run_t();
    r.cfg = *cfg;
    r.caps = (cfg->psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL) | MALLOC_CAP_8BIT;
    r.largest_min = SIZE_MAX;
    // Replicates differ in where each word falls relative to the 100 ms inference grid (seed)
    if (play) {
        timeline_build(cfg->seed);
        r.total_blocks = (KWS_SEQ_TOTAL_SAMPLES + g_tl.cum[kItems - 1] + kBlock - 1) / kBlock;
    }

    // 1. Fragmentation pattern, then load the model (the arena is the largest allocation)
    frag_apply(cfg->frag);
    heap_caps_monitor_local_minimum_free_size_start();
    const size_t largest_before = heap_caps_get_largest_free_block(r.caps);
    const size_t arena_req = kws_models[cfg->model].arena_bytes;
    const int64_t tl = now_us();
    int err = kws_net_init(&r.net, cfg->model, r.caps, arena_req);
    const int64_t load_us = now_us() - tl;

    // 2. Buffers (alloc=static: everything now; alloc=dynamic: blocks and staging per use)
    if (err == KWS_NET_OK && (play || cfg->demo)) {
        r.feat = static_cast<float *>(heap_caps_calloc(kFeat, sizeof(float), r.caps));
        if (!cfg->dynamic_alloc) {
            r.pool = static_cast<int16_t *>(heap_caps_malloc(kPoolBlocks * kBlock * sizeof(int16_t), r.caps));
            r.staging = static_cast<float *>(heap_caps_malloc(kFeat * sizeof(float), r.caps));
        }
        r.q = xQueueCreate(kQueueDepth, sizeof(int16_t *));
        r.feat_mx = xSemaphoreCreateMutex();
        r.done = xEventGroupCreate();
        if (!r.feat || (!cfg->dynamic_alloc && (!r.pool || !r.staging)) || !r.q || !r.feat_mx || !r.done) {
            err = 4;  // run buffers
        }
    }
    const size_t arena_used = r.net.interp ? r.net.interp->arena_used_bytes() : arena_req;

    if (err != KWS_NET_OK || !(play || cfg->demo)) {
        // Experiment C (seq=none) or a failed allocation: report memory only
        printf("RESULT,arena_bytes=%u,alloc_fail=%d,alloc_err=%d,load_us=%lld", (unsigned)arena_used,
               err != KWS_NET_OK, err, (long long)load_us);
        print_heap_fields(largest_before);
        printf(",frag_blocks=%d\n", g_frag_n);
        printf("DONE\n");
        fflush(stdout);
        release(r);
        return;
    }

    // 3. Tasks, then start the I2S clock: from here on the run is paced by the DMA
    ESP_ERROR_CHECK(i2s_setup(r) ? ESP_OK : ESP_FAIL);
    const BaseType_t core_t12 = cfg->affinity == AFF_NONE ? tskNO_AFFINITY : 0;
    const BaseType_t core_t3 = cfg->affinity == AFF_SAME ? 0 : cfg->affinity == AFF_SPLIT ? 1 : tskNO_AFFINITY;
    const BaseType_t core_load = cfg->affinity == AFF_NONE ? tskNO_AFFINITY : 0;
    r.t3 = spawn(r, t3_inference, "T3", kStackT3, g_stack_t3, &g_tcb[2], prio_t3(cfg->prio), core_t3);
    r.t2 = spawn(r, t2_features, "T2", kStackT2, g_stack_t2, &g_tcb[1], kPrioT2, core_t12);
    r.t1 = spawn(r, t1_capture, "T1", kStackT1, g_stack_t1, &g_tcb[0], kPrioT1, core_t12);
    r.t4 = spawn(r, t4_logger, "T4", kStackT4, g_stack_t4, &g_tcb[3], kPrioT4, tskNO_AFFINITY);
    if (cfg->load) {
        r.load = spawn(r, load_task, "LOAD", kStackLoad, g_stack_load, &g_tcb[4], kPrioLoad, core_load);
    }
    if (!r.t1 || !r.t2 || !r.t3 || !r.t4 || (cfg->load && !r.load)) {
        printf("RESULT,arena_bytes=%u,alloc_fail=1,alloc_err=5\nDONE\n", (unsigned)arena_used);
        fflush(stdout);
        release(r);
        return;
    }
    const int64_t t_start = now_us();
    ESP_ERROR_CHECK(i2s_channel_enable(r.rx));

    if (cfg->demo) {
        for (;;) {
            vTaskDelay(portMAX_DELAY);  // live until reset; T3 prints the detections
        }
    }
    xEventGroupWaitBits(r.done, BIT_T1 | BIT_T2 | BIT_T3, pdFALSE, pdTRUE, portMAX_DELAY);
    const int64_t run_us = now_us() - t_start;
    r.stop_aux = true;
    xEventGroupWaitBits(r.done, BIT_T4 | (cfg->load ? (EventBits_t)BIT_LOAD : (EventBits_t)0), pdFALSE, pdTRUE, pdMS_TO_TICKS(1000));
    i2s_channel_disable(r.rx);

    // 4. Report (protocol.py): detections, one line per task, the run summary
    for (uint32_t i = 0; i < r.n_det; i++) {
        printf("DET,%" PRIu32 ",%u,%u\n", g_dets[i].sample, g_dets[i].label, g_dets[i].score);
    }
    print_task("T1", r.t1, r.w1, run_us);
    print_task("T2", r.t2, r.w2, run_us);
    print_task("T3", r.t3, r.w3, run_us);
    print_task("T4", r.t4, r.w4, run_us);
    print_task("LOAD", r.load, r.wl, run_us);
    const uint32_t dma_ovf = r.dma_ovf.load();
    printf("RESULT,inferences=%" PRIu32, r.n_inf);
    print_inference_stats(r.n_inf);
    printf(",blocks_total=%" PRIu32 ",blocks_lost=%" PRIu32 ",dma_ovf=%" PRIu32 ",ring_ovf=%" PRIu32,
           r.total_blocks, dma_ovf + r.ring_ovf, dma_ovf, r.ring_ovf);
    print_heap_fields(largest_before);
    printf(",arena_bytes=%u,alloc_fail=0,load_us=%lld,largest_block_min=%u,run_alloc_fail=%" PRIu32
           ",inf_skipped=%" PRIu32 ",inf_fail=%" PRIu32 ",detections=%" PRIu32 ",threshold=%.2f"
           ",t1_gap_max_us=%lld,run_ms=%lld,seq_inserted=%" PRIu32 "\n",
           (unsigned)arena_used, (long long)load_us, (unsigned)r.largest_min, r.run_alloc_fail, r.inf_skipped,
           r.inf_fail, r.n_det, cfg->threshold, (long long)r.t1_gap_max, (long long)(run_us / 1000),
           g_tl.cum[kItems - 1]);
    printf("DONE\n");
    fflush(stdout);
    release(r);
}

void pipeline_eval(int model, bool psram)
{
    if (!g_seq) {
        printf("ERR no sequence partition\n");
        return;
    }
    kws_net_t net;
    const uint32_t caps = (psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL) | MALLOC_CAP_8BIT;
    const int err = kws_net_init(&net, model, caps, kws_models[model].arena_bytes);
    if (err != KWS_NET_OK) {
        printf("ERR model load failed (%d)\n", err);
        return;
    }
    static float feat[kFeat];
    int correct = 0;
    int64_t inf_sum = 0, mfcc_sum = 0;
    for (int i = 0; i < KWS_SEQ_N_ITEMS; i++) {
        const kws_seq_item_t &it = kws_seq_items[i];
        const int64_t t0 = now_us();
        kws_mfcc_clip(g_seq + it.start, feat);
        const int64_t t1 = now_us();
        kws_net_set_input(&net, feat);
        kws_net_invoke(&net);
        const int64_t t2 = now_us();
        mfcc_sum += t1 - t0;
        inf_sum += t2 - t1;
        const int pred = kws_net_argmax(&net);
        correct += pred == it.label;
        printf("EVAL,%d,%u,%d\n", i, it.label, pred);
    }
    printf("EVAL_SUMMARY,model=%s,mem=%s,correct=%d,total=%d,acc=%.4f,mfcc_clip_us=%lld,inf_mean_us=%lld,"
           "arena_used=%u\n",
           kws_models[model].name, psram ? "psram" : "sram", correct, KWS_SEQ_N_ITEMS, (double)correct / KWS_SEQ_N_ITEMS,
           (long long)(mfcc_sum / KWS_SEQ_N_ITEMS), (long long)(inf_sum / KWS_SEQ_N_ITEMS),
           (unsigned)net.interp->arena_used_bytes());
    fflush(stdout);
    kws_net_free(&net);
}

void pipeline_arena(void)
{
    for (int m = 0; m < KWS_N_MODELS; m++) {
        kws_net_t net;
        const int err = kws_net_init(&net, m, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT, 1024 * 1024);
        printf("ARENA,%s,err=%d,used=%u,reserved=%u\n", kws_models[m].name, err,
               err == KWS_NET_OK ? (unsigned)net.interp->arena_used_bytes() : 0u,
               (unsigned)kws_models[m].arena_bytes);
        kws_net_free(&net);
    }
    fflush(stdout);
}
