#include "kws_infer.h"

#include <cmath>
#include <new>

#include "esp_heap_caps.h"
#include "kws_mfcc.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace {

// Every op used by the three models (read from the .tflite files):
//   dnn:   SHAPE, STRIDED_SLICE, PACK, RESHAPE (Keras Flatten), FULLY_CONNECTED
//   cnn:   CONV_2D, MEAN (global average pooling), FULLY_CONNECTED
//   dscnn: CONV_2D, DEPTHWISE_CONV_2D, MEAN, FULLY_CONNECTED
tflite::MicroMutableOpResolver<8> g_ops;
bool g_ops_ready = false;

alignas(16) uint8_t g_interp_mem[sizeof(tflite::MicroInterpreter)];

void ops_init()
{
    if (g_ops_ready) {
        return;
    }
    g_ops.AddShape();
    g_ops.AddStridedSlice();
    g_ops.AddPack();
    g_ops.AddReshape();
    g_ops.AddFullyConnected();
    g_ops.AddConv2D();
    g_ops.AddDepthwiseConv2D();
    g_ops.AddMean();
    g_ops_ready = true;
}

}  // namespace

int kws_net_init(kws_net_t *net, int model, uint32_t caps, size_t arena_size)
{
    ops_init();
    *net = kws_net_t{};
    const tflite::Model *m = tflite::GetModel(kws_models[model].data);
    if (m->version() != TFLITE_SCHEMA_VERSION) {
        return KWS_NET_BAD_MODEL;
    }
    net->arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, arena_size, caps));
    if (!net->arena) {
        return KWS_NET_NO_ARENA;
    }
    net->arena_size = arena_size;
    net->interp = new (g_interp_mem) tflite::MicroInterpreter(m, g_ops, net->arena, arena_size);
    if (net->interp->AllocateTensors() != kTfLiteOk) {
        // The destructor of a half-initialised interpreter crashes (FreeSubgraphs); everything it
        // allocated lives in the arena, so dropping the arena is enough
        net->interp = nullptr;
        kws_net_free(net);
        return KWS_NET_ALLOC_TENSORS;
    }
    net->in = net->interp->input(0);
    net->out = net->interp->output(0);
    const int n_feat = KWS_MFCC_N_FRAMES * KWS_MFCC_N_COEFFS;
    if (net->in->type != kTfLiteInt8 || net->out->type != kTfLiteInt8 || (int)net->in->bytes != n_feat ||
        (int)net->out->bytes != KWS_N_CLASSES) {
        kws_net_free(net);
        return KWS_NET_BAD_MODEL;
    }
    net->in_k = 1.0f / (KWS_NORM_STD * net->in->params.scale);
    net->in_zp = net->in->params.zero_point;
    net->out_scale = net->out->params.scale;
    net->out_zp = net->out->params.zero_point;
    return KWS_NET_OK;
}

void kws_net_free(kws_net_t *net)
{
    if (net->interp) {
        net->interp->~MicroInterpreter();
    }
    if (net->arena) {
        heap_caps_free(net->arena);
    }
    *net = kws_net_t{};
}

void kws_net_set_input(kws_net_t *net, const float *feat)
{
    int8_t *q = net->in->data.int8;
    const int n = KWS_MFCC_N_FRAMES * KWS_MFCC_N_COEFFS;
    for (int i = 0; i < n; i++) {
        int v = (int)lrintf((feat[i] - KWS_NORM_MEAN) * net->in_k) + net->in_zp;
        q[i] = (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
    }
}

bool kws_net_invoke(kws_net_t *net)
{
    return net->interp->Invoke() == kTfLiteOk;
}

int kws_net_argmax(const kws_net_t *net)
{
    const int8_t *o = net->out->data.int8;
    int best = 0;
    for (int i = 1; i < KWS_N_CLASSES; i++) {
        if (o[i] > o[best]) {
            best = i;
        }
    }
    return best;
}

void kws_net_probs(const kws_net_t *net, float *probs)
{
    const int8_t *o = net->out->data.int8;
    float logits[KWS_N_CLASSES];
    float max = -INFINITY;
    for (int i = 0; i < KWS_N_CLASSES; i++) {
        logits[i] = (o[i] - net->out_zp) * net->out_scale;
        max = logits[i] > max ? logits[i] : max;
    }
    float sum = 0.0f;
    for (int i = 0; i < KWS_N_CLASSES; i++) {
        probs[i] = expf(logits[i] - max);
        sum += probs[i];
    }
    for (int i = 0; i < KWS_N_CLASSES; i++) {
        probs[i] /= sum;
    }
}
