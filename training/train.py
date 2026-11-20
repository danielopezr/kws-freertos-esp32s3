"""Train one keyword spotting model on the cached MFCC features.

    python train.py --model cnn --cache-dir data/features --out artifacts
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from kws import dataset, metrics, models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=models.NAMES, required=True)
    ap.add_argument("--cache-dir", default="data/features")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr = dataset.load_split(args.cache_dir, "train")
    x_va, y_va = dataset.load_split(args.cache_dir, "validation")
    x_te, y_te = dataset.load_split(args.cache_dir, "test")
    n_classes = len(dataset.LABELS)

    norm = dataset.fit_norm(x_tr)
    (out / "norm.json").write_text(json.dumps(norm, indent=2))
    x_tr, x_va, x_te = (dataset.apply_norm(x, norm) for x in (x_tr, x_va, x_te))

    model = models.build(args.model, n_classes)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    best = out / "model.keras"
    history = model.fit(
        x_tr[..., None], y_tr,
        validation_data=(x_va[..., None], y_va),
        epochs=args.epochs, batch_size=args.batch, shuffle=True, verbose=2,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(best), monitor="val_accuracy", save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-5),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ],
    )

    model = tf.keras.models.load_model(str(best))
    y_hat = np.argmax(model.predict(x_te[..., None], batch_size=512, verbose=0), axis=1)
    test = metrics.report(y_te, y_hat, dataset.LABELS)
    summary = {
        "model": args.model,
        "params": int(model.count_params()),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "epochs_run": len(history.history["loss"]),
        "test_float": test,
        "args": vars(args),
    }
    (out / "train_report.json").write_text(json.dumps(summary, indent=2))
    (out / "history.json").write_text(json.dumps({k: [float(v) for v in vs] for k, vs in history.history.items()}, indent=2))
    print(f"{args.model}: params={summary['params']}  test accuracy={test['accuracy']:.4f}  macro F1={test['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
