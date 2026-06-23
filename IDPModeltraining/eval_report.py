#!/usr/bin/env python3
"""
eval_report.py  —  Shared evaluation + plotting helpers.

Used by train_fp.py, train_qat.py and test_models.py to produce a consistent
console report and PNG figures (confusion matrix, per-class P/R/F1 bars,
training curves).  Depends on scikit-learn and matplotlib (Agg backend).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)


def classification_metrics(y_true, y_pred, class_names):
    """Return a dict of accuracy + per-class & averaged precision/recall/F1."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "class_names": list(class_names),
        "precision": p, "recall": r, "f1": f, "support": sup,
        "macro_f1": float(macro[2]), "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "weighted_f1": float(weighted[2]),
    }


def print_metrics_table(name, m):
    """Print a framed per-class precision/recall/F1 table."""
    line = "─" * 64
    print(f"\n┌{line}┐")
    print(f"│ {name:<62} │")
    print(f"├{line}┤")
    print(f"│ {'class':<14}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>12} │")
    for i, c in enumerate(m["class_names"]):
        print(f"│ {c:<14}{m['precision'][i]:>12.4f}{m['recall'][i]:>12.4f}"
              f"{m['f1'][i]:>12.4f}{int(m['support'][i]):>12,} │")
    print(f"├{line}┤")
    print(f"│ {'accuracy':<14}{m['accuracy']:>12.4f}"
          f"{'macro-F1':>16}{m['macro_f1']:>10.4f}{'':>12} │")
    print(f"└{line}┘", flush=True)


def plot_confusion_matrix(y_true, y_pred, class_names, out_png, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, sub, fmt in [(axes[0], cm, "counts", "d"),
                               (axes[1], cm_norm, "row-normalized", ".2f")]:
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=(data.max() if sub == "counts" else 1))
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title} ({sub})")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if (v > (data.max() * 0.5 if sub == "counts" else 0.5)) else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  saved {out_png}")


def plot_metric_bars(m, out_png, title="Per-class metrics"):
    cn = m["class_names"]; x = np.arange(len(cn)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, m["precision"], w, label="precision")
    ax.bar(x,     m["recall"],    w, label="recall")
    ax.bar(x + w, m["f1"],        w, label="f1")
    ax.set_xticks(x); ax.set_xticklabels(cn)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.set_title(title); ax.legend()
    for i in range(len(cn)):
        for off, key in [(-w, "precision"), (0, "recall"), (w, "f1")]:
            ax.text(i + off, m[key][i], f"{m[key][i]:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  saved {out_png}")


def plot_training_curves(history, out_png, title="Training curves"):
    """history: dict with lists 'loss', 'train_acc', 'val_acc'."""
    epochs = range(1, len(history.get("val_acc", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if history.get("loss"):
        ax1.plot(epochs, history["loss"], "o-", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title(f"{title} — loss"); ax1.legend()
    if history.get("train_acc"):
        ax2.plot(epochs, history["train_acc"], "o-", label="train acc")
    if history.get("val_acc"):
        ax2.plot(epochs, history["val_acc"], "s-", label="val acc")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_ylim(0, 1.02)
    ax2.set_title(f"{title} — accuracy"); ax2.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"  saved {out_png}")
