#include "kws_models.h"

#include <cstring>

#include "cnn_model_data.h"
#include "dnn_model_data.h"
#include "dscnn_model_data.h"

// Arena sizes: arena_used_bytes() measured on the board with the ARENA command (2026-11-25),
// rounded up with a small margin. DS-CNN (172 channels) needs more than the largest free block
// of internal SRAM (~262 KB), so it only runs with mem=psram; with mem=sram it is an allocation
// failure, which the experiments record as a result.
const kws_model_info_t kws_models[KWS_N_MODELS] = {
    {"dnn", g_dnn_model_data, 10 * 1024},     // used 8716 B
    {"cnn", g_cnn_model_data, 54 * 1024},     // used 52524 B
    {"dscnn", g_dscnn_model_data, 450 * 1024},  // used 458876 B
};

int kws_model_by_name(const char *name)
{
    for (int i = 0; i < KWS_N_MODELS; i++) {
        if (strcmp(name, kws_models[i].name) == 0) {
            return i;
        }
    }
    return -1;
}
