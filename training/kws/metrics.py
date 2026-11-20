"""Classification metrics with plain numpy."""

import numpy as np


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def report(y_true, y_pred, labels):
    """Accuracy, macro F1 and per-class precision / recall / F1."""
    cm = confusion(y_true, y_pred, len(labels))
    tp = np.diag(cm).astype(np.float64)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for name, p, r, f, s in zip(labels, precision, recall, f1, cm.sum(axis=1))
        },
        "confusion_matrix": cm.tolist(),
    }
