#!/usr/bin/env python3
"""
test_models.py  —  Evaluate & compare the FP and QAT models for accuracy AND
inference performance on a held-out split of dataset_final.csv.

Both models are evaluated on the SAME stratified test slice (random_state=42)
that training held out.  Produces a framed FP-vs-QAT comparison table and:
    test_fp_confusion_matrix.png
    test_qat_confusion_matrix.png
    model_comparison.png

    python3 test_models.py --fp model_fp.pth --qat model_qat.pth --runs 1000
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from constants import FEATURE_ORDER, CLASS_NAMES
from model import TrafficClassifier
from train_fp import normalize as normalize_fp
from train_qat import QuantMLP, normalize_signed

import eval_report

LABEL_MAP = {CLASS_NAMES[i]: i for i in CLASS_NAMES}
CLASS_LIST = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]


def banner(t): print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)


@torch.no_grad()
def batched_predict(model, X, bs=4096):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs], dtype=torch.float32)
        out.append(model(xb).argmax(1).numpy())
    return np.concatenate(out)


@torch.no_grad()
def latency_us(model, x_one, runs):
    """Mean / p50 / p95 single-sample latency in microseconds."""
    xb = torch.tensor(x_one, dtype=torch.float32).reshape(1, -1)
    for _ in range(20):                       # warmup
        model(xb)
    t = np.empty(runs)
    for i in range(runs):
        s = time.perf_counter(); model(xb); t[i] = (time.perf_counter() - s) * 1e6
    return t.mean(), np.percentile(t, 50), np.percentile(t, 95)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset_final.csv")
    p.add_argument("--fp", default="model_fp.pth")
    p.add_argument("--qat", default="model_qat.pth")
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    banner("MODEL TEST & COMPARISON  (held-out test split)")
    df = pd.read_csv(args.data)
    df = df[df["label"].isin(LABEL_MAP)]
    Xraw = df[FEATURE_ORDER].values.astype(np.float32)
    y = df["label"].map(LABEL_MAP).values.astype(np.int64)
    # same split as training → second slice is the held-out set
    _, Xte, _, yte = train_test_split(Xraw, y, test_size=0.2, stratify=y,
                                      random_state=args.seed)
    print(f"  test rows = {len(Xte):,}  classes = {CLASS_LIST}")

    results = {}

    # ── FP model ─────────────────────────────────────────────────────────────
    if os.path.exists(args.fp):
        fp = TrafficClassifier(); fp.load_state_dict(torch.load(args.fp, map_location="cpu")); fp.eval()
        Xfp = normalize_fp(Xte)
        yp = batched_predict(fp, Xfp)
        m = eval_report.classification_metrics(yte, yp, CLASS_LIST)
        eval_report.print_metrics_table("FP model (float32)", m)
        eval_report.plot_confusion_matrix(yte, yp, CLASS_LIST, "test_fp_confusion_matrix.png",
                                          title="FP test confusion")
        lat = latency_us(fp, Xfp[0], args.runs)
        results["FP"] = {"acc": m["accuracy"], "f1": m["macro_f1"], "lat": lat,
                         "size": os.path.getsize(args.fp)}
    else:
        print(f"  !! {args.fp} not found — run train_fp.py first")

    # ── QAT model ────────────────────────────────────────────────────────────
    if os.path.exists(args.qat):
        ckpt = torch.load(args.qat, map_location="cpu")
        qat = QuantMLP(); qat.load_state_dict(ckpt["state_dict"])
        qat.shift_l1 = int(ckpt["shift_l1"]); qat.shift_l2 = int(ckpt["shift_l2"])
        qat.quant_enabled = True; qat.quant_strength = 1.0; qat.eval()
        Xq = normalize_signed(Xte)
        yp = batched_predict(qat, Xq)
        m = eval_report.classification_metrics(yte, yp, CLASS_LIST)
        eval_report.print_metrics_table("QAT model (INT8, shifts "
                                        f"L1={qat.shift_l1} L2={qat.shift_l2})", m)
        eval_report.plot_confusion_matrix(yte, yp, CLASS_LIST, "test_qat_confusion_matrix.png",
                                          title="QAT test confusion")
        lat = latency_us(qat, Xq[0], args.runs)
        results["QAT"] = {"acc": m["accuracy"], "f1": m["macro_f1"], "lat": lat,
                          "size": os.path.getsize(args.qat)}
    else:
        print(f"  !! {args.qat} not found — run train_qat.py first")

    # ── comparison table ───────────────────────────────────────────────────
    banner("FP vs QAT — ACCURACY & PERFORMANCE")
    hdr = f"  {'model':<8}{'accuracy':>11}{'macro-F1':>11}{'lat mean(µs)':>15}{'lat p95(µs)':>14}{'size(KB)':>11}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for k in ("FP", "QAT"):
        if k in results:
            r = results[k]
            print(f"  {k:<8}{r['acc']:>11.4f}{r['f1']:>11.4f}"
                  f"{r['lat'][0]:>15.2f}{r['lat'][2]:>14.2f}{r['size']/1024:>11.1f}")

    if len(results) == 2:
        plot_comparison(results, "model_comparison.png")


def plot_comparison(results, out_png):
    models = list(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(models)); w = 0.35
    ax1.bar(x - w/2, [results[m]["acc"] for m in models], w, label="accuracy")
    ax1.bar(x + w/2, [results[m]["f1"] for m in models], w, label="macro-F1")
    ax1.set_xticks(x); ax1.set_xticklabels(models); ax1.set_ylim(0, 1.02)
    ax1.set_title("Accuracy / macro-F1"); ax1.legend()
    for i, m in enumerate(models):
        ax1.text(i - w/2, results[m]["acc"], f"{results[m]['acc']:.3f}", ha="center", va="bottom", fontsize=8)
        ax1.text(i + w/2, results[m]["f1"], f"{results[m]['f1']:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.bar(x, [results[m]["lat"][0] for m in models], 0.5, color="darkorange")
    ax2.set_xticks(x); ax2.set_xticklabels(models)
    ax2.set_ylabel("µs / inference"); ax2.set_title("Mean single-sample latency (CPU)")
    for i, m in enumerate(models):
        ax2.text(i, results[m]["lat"][0], f"{results[m]['lat'][0]:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"\n  saved {out_png}")


if __name__ == "__main__":
    main()
