/*
 * PC harness for kws_mfcc: reads N clips of 16 000 int16 samples (raw, little endian)
 * and writes N x 49 x 40 float32 (raw). Also reports the average time per frame.
 *
 *   host_test <clips.raw> <mfcc_out.raw>
 *
 * Driven by training/validate_mfcc_c.py, which compares the output with features.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kws_mfcc.h"

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <clips.raw> <mfcc_out.raw>\n", argv[0]);
        return 2;
    }
    FILE *in = fopen(argv[1], "rb");
    FILE *out = fopen(argv[2], "wb");
    if (!in || !out) {
        fprintf(stderr, "cannot open files\n");
        return 1;
    }
    kws_mfcc_init();

    static int16_t clip[KWS_MFCC_CLIP_LEN];
    static float mfcc[KWS_MFCC_N_FRAMES * KWS_MFCC_N_COEFFS];
    long clips = 0;
    clock_t t0 = clock();
    while (fread(clip, sizeof(int16_t), KWS_MFCC_CLIP_LEN, in) == KWS_MFCC_CLIP_LEN) {
        kws_mfcc_clip(clip, mfcc);
        fwrite(mfcc, sizeof(float), KWS_MFCC_N_FRAMES * KWS_MFCC_N_COEFFS, out);
        clips++;
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
    fclose(in);
    fclose(out);
    printf("clips=%ld frames=%ld us_per_frame_host=%.2f\n", clips, clips * KWS_MFCC_N_FRAMES,
           clips ? 1e6 * secs / (clips * KWS_MFCC_N_FRAMES) : 0.0);
    return 0;
}
