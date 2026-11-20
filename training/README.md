# training

Trains the three keyword spotting models on Speech Commands v2 and exports them as INT8 TFLite models plus C arrays for the firmware.

12 classes: `_silence_`, `_unknown_` and the keywords *yes, no, up, down, left, right, on, off, stop, go*.

| Model | Architecture | Role in the study |
| --- | --- | --- |
| `dnn` | 3 × FC-144 | little compute, many weights |
| `cnn` | 3 strided convolutions + global pooling | main model for the scheduling experiments |
| `dscnn` | conv + 5 depthwise-separable blocks, 172 channels | largest activations, stresses internal SRAM |

## Run it (Google Colab)

Open [`colab_train.ipynb`](colab_train.ipynb) in Colab, select a **T4 GPU** runtime and *Run all*. It downloads the dataset, builds the feature cache, trains and quantizes the three models and downloads `kws_artifacts.zip`. With Drive enabled, the cache and artifacts live in `MyDrive/kws-freertos-esp32s3/`, so an interrupted session resumes without recomputing features.

Local runs need Python 3.10–3.12 (TensorFlow has no wheels for newer versions):

```bash
python prepare_data.py --data-dir data/speech_commands --cache-dir data/features
python train.py    --model cnn --cache-dir data/features --out artifacts
python quantize.py --model cnn --cache-dir data/features --out artifacts
```

## Feature front end

[`kws/features.py`](kws/features.py) is the reference the firmware must reproduce exactly: 1 s at 16 kHz → 640-sample frames (40 ms) with a 320-sample hop (20 ms) → periodic Hann → 1024-point FFT power spectrum → 40 HTK mel filters (20–4000 Hz) → `log(x + 1e-6)` → orthonormal DCT-II, 40 coefficients → **49 × 40**.

MFCCs are then normalised with the training-set mean and standard deviation (`norm.json`) outside the model graph, so the INT8 model only uses ops supported by TFLM.

## Data pipeline

- Official `validation_list.txt` / `testing_list.txt` splits.
- Silence and unknown each add 10 % of the keyword count in every split (silence = background-noise crops at a random volume).
- The training split holds a clean copy plus one augmented copy: ±100 ms time shift and, with probability 0.8, background noise at volume ≤ 0.1.
- Fixed seed: the cache is reproducible.

## Outputs (`artifacts/<model>/`)

| File | Content |
| --- | --- |
| `model.keras` | best checkpoint (validation accuracy) |
| `norm.json` | MFCC mean and standard deviation |
| `train_report.json`, `history.json` | parameters, float test accuracy, per-class metrics, learning curves |
| `model_int8.tflite` | full-integer INT8 model |
| `<model>_model_data.cc/.h` | aligned C array for TFLM |
| `quant_report.json` | float vs INT8 accuracy, size, input/output scale and zero point |

## Tests

```bash
python -m unittest discover -s tests
```

They cover the MFCC front end and the dataset builder (on a tiny synthetic dataset) and need only numpy.
