# training

Training of the three keyword spotting models on Speech Commands v2
(12 classes: 10 keywords + "silence" + "unknown").

| Model | Architecture |
| --- | --- |
| Small | DNN |
| Medium | CNN |
| Large | DS-CNN |

To do:
- `download_data.py`: download the dataset into `training/data/` (not versioned).
- `train.py`: MFCC extraction (40 coefficients, 40 ms window, 20 ms stride → 49 × 40) and training.
- `quantize.py`: INT8 quantization → `.tflite` → C array for `firmware/`.
