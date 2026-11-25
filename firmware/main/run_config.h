// Experiment configuration received over UART (see experiments/protocol.py):
//
//   CFG model=cnn affinity=split prio=higher load=on mem=sram alloc=static frag=0 seq=full demo=off seed=42
//
// Every key is required; `thr=<0..1>` is optional (decision threshold, default KWS_DEFAULT_THRESHOLD).
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
};

// Parse a CFG line. On error returns false and writes the reason into err.
bool kws_parse_cfg(const char *line, kws_cfg_t *cfg, char *err, size_t errlen);
