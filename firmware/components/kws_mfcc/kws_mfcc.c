#include "kws_mfcc.h"

#include <math.h>
#include <string.h>

#include "kws_mfcc_tables.h"

#define N_FFT   KWS_MFCC_TABLES_N_FFT        /* 1024 real points          */
#define N_HALF  (N_FFT / 2)                  /* 512-point complex FFT     */
#define N_BINS  (N_FFT / 2 + 1)              /* 513 power-spectrum bins   */
#define N_MELS  KWS_MFCC_TABLES_N_MELS

#if KWS_MFCC_TABLES_FRAME_LEN != KWS_MFCC_FRAME_LEN || KWS_MFCC_TABLES_N_MFCC != KWS_MFCC_N_COEFFS
#error "kws_mfcc_tables.h does not match kws_mfcc.h: regenerate the tables"
#endif

/* Work buffers (static: the front end runs in a single task) */
static float s_re[N_HALF];
static float s_im[N_HALF];
static float s_power[N_BINS];
static float s_logmel[N_MELS];
static uint16_t s_bitrev[N_HALF];

void kws_mfcc_init(void)
{
    int bits = 0;
    while ((1 << bits) < N_HALF) {
        bits++;
    }
    for (int i = 0; i < N_HALF; i++) {
        unsigned r = 0;
        for (int b = 0; b < bits; b++) {
            r |= ((i >> b) & 1u) << (bits - 1 - b);
        }
        s_bitrev[i] = (uint16_t)r;
    }
}

/* In-place iterative radix-2 complex FFT of size N_HALF (forward, e^{-i...}).
 * Twiddle for this size at index m is table entry 2*m (tables are for N_FFT). */
static void fft_half(float *re, float *im)
{
    for (int i = 0; i < N_HALF; i++) {
        int j = s_bitrev[i];
        if (j > i) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= N_HALF; len <<= 1) {
        int half = len >> 1;
        int step = (N_FFT / len);  /* twiddle stride in the N_FFT table */
        for (int start = 0; start < N_HALF; start += len) {
            for (int k = 0; k < half; k++) {
                float wr = kws_tw_cos[k * step];
                float wi = -kws_tw_sin[k * step];
                int a = start + k, b = a + half;
                float tr = re[b] * wr - im[b] * wi;
                float ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
    }
}

/* Power spectrum of the windowed frame (zero-padded to N_FFT) into s_power.
 * Real FFT of length N via a complex FFT of length N/2:
 *   z[n] = x[2n] + i x[2n+1],  Z = FFT_{N/2}(z)
 *   X[k] = E[k] + W^k O[k],  E = (Z[k] + conj Z[N/2-k]) / 2,  O = (Z[k] - conj Z[N/2-k]) / 2i
 */
static void power_spectrum(const int16_t *samples)
{
    const float scale = 1.0f / 32768.0f;
    for (int n = 0; n < N_HALF; n++) {
        int i0 = 2 * n, i1 = 2 * n + 1;
        s_re[n] = (i0 < KWS_MFCC_FRAME_LEN) ? (samples[i0] * scale) * kws_window[i0] : 0.0f;
        s_im[n] = (i1 < KWS_MFCC_FRAME_LEN) ? (samples[i1] * scale) * kws_window[i1] : 0.0f;
    }
    fft_half(s_re, s_im);

    /* k = 0 and k = N/2 are real */
    float x0 = s_re[0] + s_im[0];
    float xn = s_re[0] - s_im[0];
    s_power[0] = x0 * x0;
    s_power[N_HALF] = xn * xn;

    for (int k = 1; k < N_HALF; k++) {
        float zr = s_re[k], zi = s_im[k];
        float cr = s_re[N_HALF - k], ci = -s_im[N_HALF - k];  /* conj Z[N/2-k] */
        float er = 0.5f * (zr + cr), ei = 0.5f * (zi + ci);
        /* O = (Z - conjZ') / 2i  ->  (a + ib) / 2i = (b - ia) / 2 */
        float ar = zr - cr, ai = zi - ci;
        float or_ = 0.5f * ai, oi = -0.5f * ar;
        float wr = kws_tw_cos[k], wi = -kws_tw_sin[k];
        float xr = er + (or_ * wr - oi * wi);
        float xi = ei + (or_ * wi + oi * wr);
        s_power[k] = xr * xr + xi * xi;
    }
}

void kws_mfcc_frame(const int16_t *samples, float *coeffs)
{
    power_spectrum(samples);

    for (int m = 0; m < N_MELS; m++) {
        const float *w = &kws_mel_w[kws_mel_offset[m]];
        const float *p = &s_power[kws_mel_start[m]];
        float acc = 0.0f;
        for (int j = 0; j < kws_mel_len[m]; j++) {
            acc += p[j] * w[j];
        }
        s_logmel[m] = logf(acc + KWS_MFCC_TABLES_LOG_EPS);
    }

    for (int c = 0; c < KWS_MFCC_N_COEFFS; c++) {
        const float *row = &kws_dct[c * N_MELS];
        float acc = 0.0f;
        for (int m = 0; m < N_MELS; m++) {
            acc += row[m] * s_logmel[m];
        }
        coeffs[c] = acc;
    }
}

void kws_mfcc_clip(const int16_t *clip, float *out)
{
    for (int f = 0; f < KWS_MFCC_N_FRAMES; f++) {
        kws_mfcc_frame(&clip[f * KWS_MFCC_HOP], &out[f * KWS_MFCC_N_COEFFS]);
    }
}
