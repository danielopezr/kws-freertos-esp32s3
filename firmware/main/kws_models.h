// The three INT8 keyword spotting models (training/, first training run) and how they are fed.
#pragma once

#include <cstddef>
#include <cstdint>

enum { KWS_MODEL_DNN, KWS_MODEL_CNN, KWS_MODEL_DSCNN, KWS_N_MODELS };

#define KWS_N_CLASSES 12     // _silence_, _unknown_, then the 10 keywords
#define KWS_FIRST_KEYWORD 2

// Feature normalisation from training (norm.json; identical for the three models).
// The INT8 input scale and zero point are read from each model at run time.
#define KWS_NORM_MEAN (-0.6560660446818979f)
#define KWS_NORM_STD (6.152801217507387f)

struct kws_model_info_t {
    const char *name;
    const unsigned char *data;
    size_t arena_bytes;  // tensor arena reserved for the model: measured with the ARENA command, plus a margin
};

extern const kws_model_info_t kws_models[KWS_N_MODELS];

// Index of a model by name ("dnn", "cnn", "dscnn"), or -1
int kws_model_by_name(const char *name);
