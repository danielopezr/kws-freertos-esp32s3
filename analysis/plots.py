"""Figures: box plot per treatment and interaction plot (mean with 95 % CI)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as st  # noqa: E402

LABELS = {
    "det_latency_ms": "Detection latency (ms)",
    "loss_pct": "Audio blocks lost (%)",
    "f1": "F1",
    "recall": "Recall",
    "inf_mean_us": "Mean inference time (µs)",
    "heap_free_min": "Minimum free heap (B)",
    "heap_largest_block": "Largest free block (B)",
}


def label(response):
    return LABELS.get(response, response)


def boxplot(df, factors, response, path):
    order = df.sort_values(factors)["treatment_label"].unique()
    groups = [(name, df.loc[df["treatment_label"] == name, response].dropna().to_numpy()) for name in order]
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(groups) + 2), 4.5))
    ax.boxplot([g for _, g in groups], showmeans=True)
    ax.set_xticks(range(1, len(groups) + 1), [n for n, _ in groups], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(label(response))
    ax.set_xlabel(" / ".join(factors))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _levels(df, column):
    """Levels in design order for categorical columns, sorted otherwise."""
    if hasattr(df[column], "cat"):
        present = set(df[column].dropna())
        return [lv for lv in df[column].cat.categories if lv in present]
    return sorted(df[column].dropna().unique())


def _mean_ci(y):
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 2:
        return (y.mean() if len(y) else np.nan), 0.0
    return y.mean(), st.t.ppf(0.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y))


def interaction(df, x, trace, response, path, panel=None):
    """Mean +/- 95 % CI of response against x, one line per level of trace, one panel per level of panel."""
    panels = _levels(df, panel) if panel else [None]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), sharey=True, squeeze=False)
    x_levels = _levels(df, x)
    for ax, p in zip(axes[0], panels):
        sub = df if p is None else df[df[panel] == p]
        for t in _levels(sub, trace):
            stats = [_mean_ci(sub[(sub[x] == lv) & (sub[trace] == t)][response]) for lv in x_levels]
            means, halves = zip(*stats)
            ax.errorbar(range(len(x_levels)), means, yerr=halves, marker="o", capsize=4, label=f"{trace}={t}")
        ax.set_xticks(range(len(x_levels)), x_levels)
        ax.set_xlabel(x)
        ax.set_title(f"{panel}={p}" if p is not None else "")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(label(response))
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
