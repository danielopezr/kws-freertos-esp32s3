// Experiment configuration received over UART (see experiments/protocol.py):
//
//   CFG model=cnn affinity=split prio=higher load=on mem=sram alloc=static frag=0 seq=full demo=off seed=42
//
// Every key is required. Optional, never sent by runner.py: `thr=<0..1>` (decision threshold, default
// KWS_DEFAULT_THRESHOLD), `secs=<n>` (play only the first n seconds: smoke tests of a configuration) and
// `gain=<1..128>` (live microphone only: digital gain on the 24-bit INMP441 sample, default 1 = its top
// 16 bits; a quiet voice at 30 cm needs ~8 to reach the -22 dBFS of the training clips) and
// `slot=left|right` (live microphone only: which I2S slot to read; right = L/R wiring check).
#pragma once

#include <cstddef>
#include <cstdint>

enum { AFF_SAME, AFF_SPLIT, AFF_NONE };        // same: T1-T3 on core 0; split: T1-T2 core 0, T3 core 1
enum { PRIO_LOWER, PRIO_EQUAL, PRIO_HIGHER };  // T3 priority relative to T1

#define KWS_DEFAULT_THRESHOLD 0.80f

struct kws_cfg_t {
    int model;           // KWS_MODEL_*
    int affinity;        // AFF_*
    int prio;            // PRIO_*
    bool load;           // synthetic background task
    bool psram;          // buffers and tensor arena in PSRAM (else internal SRAM)
    bool dynamic_alloc;  // per-block / per-inference malloc+free (else preallocated pools)
    int frag;            // 0, 1, 2: heap fragmentation pattern before loading the model
    bool seq_full;       // play the test sequence (else only load the model)
    bool demo;           // live microphone, endless, detections printed as they happen
    uint32_t seed;
    float threshold;
    uint32_t max_seconds;  // 0 = the whole sequence
    uint32_t gain;         // live microphone: 1..KWS_MAX_GAIN
    bool mic_right;        // live microphone: read the right I2S slot (`slot=right`, L/R wiring check)
};

#define KWS_MAX_GAIN 128u  // (24-bit sample) x gain stays inside int32

// Parse a CFG line. On error returns false and writes the reason into err.
bool kws_parse_cfg(const char *line, kws_cfg_t *cfg, char *err, size_t errlen);
