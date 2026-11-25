// TensorFlow Lite Micro wrapper for the keyword spotting models (task T3).
//
// Only one network can be loaded at a time (the interpreter lives in a static buffer).
// The tensor arena is allocated with heap_caps_aligned_alloc in the requested memory,
// so where it lives (internal SRAM or PSRAM) is an explicit experiment factor.
#pragma once

#include <cstddef>
#include <cstdint>

#include "kws_models.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

enum {
    KWS_NET_OK = 0,
    KWS_NET_NO_ARENA = 1,       // heap_caps_aligned_alloc of the arena failed (no large enough free block)
    KWS_NET_ALLOC_TENSORS = 2,  // AllocateTensors failed (arena too small)
    KWS_NET_BAD_MODEL = 3,      // schema version, tensor types or shapes not as expected
};

struct kws_net_t {
    tflite::MicroInterpreter *interp;
    TfLiteTensor *in, *out;
    uint8_t *arena;
    size_t arena_size;
    float in_k;  // q = round((x - mean) * in_k) + in_zp, with in_k = 1 / (std * input scale)
    int in_zp;
    float out_scale;
    int out_zp;
};

// Load a model with an arena of arena_size bytes in memory with the given heap caps.
int kws_net_init(kws_net_t *net, int model, uint32_t caps, size_t arena_size);
void kws_net_free(kws_net_t *net);

// Normalise and quantise 49 x 40 MFCCs (oldest frame first) into the input tensor.
void kws_net_set_input(kws_net_t *net, const float *feat);
bool kws_net_invoke(kws_net_t *net);
int kws_net_argmax(const kws_net_t *net);
// Softmax of the dequantised logits (KWS_N_CLASSES values).
void kws_net_probs(const kws_net_t *net, float *probs);
