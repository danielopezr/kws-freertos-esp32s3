"""Full-integer INT8 quantization, INT8 evaluation and C-array export for TFLM.

    python quantize.py --model cnn --cache-dir data/features --out artifacts

Writes to artifacts/<model>/:
  model_int8.tflite      flatbuffer for TFLM
  <model>_model_data.cc  C array (aligned) to drop into firmware/
  <model>_model_data.h
  quant_report.json      float vs INT8 accuracy, size, MFCC mean/std, input/output scale and zero point
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from kws import dataset, metrics, models


def representative(x, n=1000, seed=0):
    idx = np.random.default_rng(seed).choice(len(x), min(n, len(x)), replace=False)

    def gen():
        for i in idx:
            yield [x[i:i + 1, ..., None].astype(np.float32)]

    return gen


def convert(model, x_train, export_dir):
    # Keras 3: going through a SavedModel is the most reliable path to the TFLite converter
    model.export(str(export_dir))
    conv = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative(x_train)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def evaluate_int8(tflite_bytes, x, y):
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    preds = np.empty(len(x), dtype=np.int64)
    for i in range(len(x)):
        q = np.clip(np.round(x[i] / in_scale + in_zp), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q[None, ..., None])
        it.invoke()
        preds[i] = int(np.argmax(it.get_tensor(out["index"])[0]))
    return preds, inp, out


def write_c_array(tflite_bytes, name, out_dir):
    var = f"g_{name}_model_data"
    body = ",\n".join(
        "  " + ", ".join(f"0x{b:02x}" for b in tflite_bytes[i:i + 12]) for i in range(0, len(tflite_bytes), 12)
    )
    (out_dir / f"{name}_model_data.h").write_text(
        "#pragma once\n#include <cstdint>\n\n"
        f"extern const unsigned char {var}[];\nextern const unsigned int {var}_len;\n"
    )
    (out_dir / f"{name}_model_data.cc").write_text(
        f'#include "{name}_model_data.h"\n\n'
        f"alignas(16) const unsigned char {var}[] = {{\n{body}\n}};\n"
        f"const unsigned int {var}_len = {len(tflite_bytes)};\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=models.NAMES, required=True)
    ap.add_argument("--cache-dir", default="data/features")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    out = Path(args.out) / args.model
    model = tf.keras.models.load_model(str(out / "model.keras"))
    x_tr, _ = dataset.load_split(args.cache_dir, "train")
    x_te, y_te = dataset.load_split(args.cache_dir, "test")
    norm = json.loads((out / "norm.json").read_text())
    x_tr, x_te = dataset.apply_norm(x_tr, norm), dataset.apply_norm(x_te, norm)

    tflite_bytes = convert(model, x_tr, out / "saved_model")
    (out / "model_int8.tflite").write_bytes(tflite_bytes)
    write_c_array(tflite_bytes, args.model, out)

    y_float = np.argmax(model.predict(x_te[..., None], batch_size=512, verbose=0), axis=1)
    y_int8, inp, outp = evaluate_int8(tflite_bytes, x_te, y_te)
    rep_float = metrics.report(y_te, y_float, dataset.LABELS)
    rep_int8 = metrics.report(y_te, y_int8, dataset.LABELS)
    summary = {
        "model": args.model,
        "tflite_bytes": len(tflite_bytes),
        "test_accuracy_float": rep_float["accuracy"],
        "test_accuracy_int8": rep_int8["accuracy"],
        "test_macro_f1_float": rep_float["macro_f1"],
        "test_macro_f1_int8": rep_int8["macro_f1"],
        "input": {"scale": float(inp["quantization"][0]), "zero_point": int(inp["quantization"][1]),
                  "shape": [int(s) for s in inp["shape"]]},
        "norm": norm,
        "output": {"scale": float(outp["quantization"][0]), "zero_point": int(outp["quantization"][1])},
        "int8_report": rep_int8,
    }
    (out / "quant_report.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.model}: {len(tflite_bytes) / 1024:.1f} KB  "
          f"acc float={summary['test_accuracy_float']:.4f}  int8={summary['test_accuracy_int8']:.4f}")


if __name__ == "__main__":
    main()
