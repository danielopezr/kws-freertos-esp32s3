# kws_mfcc

C99 MFCC front end used by task T2. It reproduces [`training/kws/features.py`](../../../training/kws/features.py) step by step:

1. `int16 / 32768`, 640-sample frame (40 ms) × periodic Hann window
2. 1024-point real FFT (computed as a 512-point complex FFT + split), power spectrum, 513 bins
3. 40 HTK mel filters (20–4000 Hz), stored sparse
4. `ln(x + 1e-6)`
5. Orthonormal DCT-II → 40 coefficients

```c
kws_mfcc_init();                     // once
kws_mfcc_frame(samples_640, out_40); // one 20 ms hop in streaming mode
kws_mfcc_clip(samples_16000, out_49x40);
```

Not re-entrant (static work buffers): only T2 calls it.

## Tables

`kws_mfcc_tables.h` is **generated**. Window, mel filterbank, DCT matrix and FFT twiddles are the exact float32 values from `features.py`, so the two implementations cannot drift apart. After any change to `features.py`:

```bash
cd training
python export_mfcc_tables.py          # regenerate
python export_mfcc_tables.py --check  # CI-style check
```

`training/tests/test_mfcc_tables.py` fails if the header is stale.

## Validation on the PC

`host_test/host_test.c` runs the same code on the PC. Build it with any C compiler, for example:

```bash
cc -O2 -std=c11 -Iinclude -I. kws_mfcc.c host_test/host_test.c -lm -o host_test
```

Then compare against Python:

```bash
cd training
python validate_mfcc_c.py --exe ../firmware/components/kws_mfcc/host_test/host_test
```

The check that matters is the **INT8 model input** (after normalisation and quantisation): it must be identical for ≥ 99.9 % of the values and never differ by more than one step.
