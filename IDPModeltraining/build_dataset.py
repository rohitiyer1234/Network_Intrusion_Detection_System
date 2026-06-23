#!/usr/bin/env python3
"""
build_dataset.py  —  Produce the final balanced training set and recalibrate
the per-feature quantization ranges.

Input : training_features.csv  (from generate_training_data.py)
Output: dataset_final.csv       — equal #samples per class (normal/ddos/botnet)
        dataset_distribution.png — raw-vs-balanced class counts
        constants.py             — FEATURE_RANGES block rewritten in place

Balancing
---------
The scarce class (botnet) sets the target.  normal and ddos are *randomly
down-sampled* to that count — no oversampling, no synthetic noise.

Range calibration (minimal quantization loss)
---------------------------------------------
For each of the 20 features we compute robust bounds from the balanced data:
    min = max(0, 0.1st percentile)      max = 99.9th percentile (rounded up)
Using a high percentile instead of the raw max stops a handful of extreme
outliers from blowing up the span and starving INT8 resolution.  The feature
SET / ORDER never changes — only the numeric bounds.
"""

import argparse
import math
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import FEATURE_ORDER, FEATURE_RANGES, CLASS_NAMES

CLASSES = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]   # [normal, ddos, botnet]
CONSTANTS_PATH = os.path.join(os.path.dirname(__file__), "constants.py")
BEGIN_MARK = "# ── BEGIN AUTO-CALIBRATED FEATURE_RANGES ──"
END_MARK   = "# ── END AUTO-CALIBRATED FEATURE_RANGES ──"


def banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)


# ── balancing ─────────────────────────────────────────────────────────────────

def balance(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df[df["label"].isin(CLASSES)].copy()
    raw_counts = df["label"].value_counts().to_dict()
    target = min(raw_counts.get(c, 0) for c in CLASSES)

    banner("CLASS BALANCING")
    print(f"  Target per class = {target:,} (smallest class)\n")
    parts = []
    for c in CLASSES:
        sub = df[df["label"] == c]
        if len(sub) > target:
            sub = sub.sample(target, random_state=seed)
        parts.append(sub)
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    bal_counts = out["label"].value_counts().to_dict()
    print(f"  {'class':<10}{'raw':>14}{'balanced':>14}")
    print(f"  {'-'*10}{'-'*14:>14}{'-'*14:>14}")
    for c in CLASSES:
        print(f"  {c:<10}{raw_counts.get(c,0):>14,}{bal_counts.get(c,0):>14,}")
    print(f"  {'TOTAL':<10}{len(df):>14,}{len(out):>14,}")
    return out, raw_counts, bal_counts


def plot_distribution(raw, bal, out_png: str) -> None:
    x = np.arange(len(CLASSES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, [raw.get(c, 0) for c in CLASSES], w, label="raw (pre-balance)")
    ax.bar(x + w/2, [bal.get(c, 0) for c in CLASSES], w, label="balanced (final)")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_ylabel("samples"); ax.set_title("Class distribution — raw vs balanced")
    ax.legend()
    for i, c in enumerate(CLASSES):
        ax.text(i - w/2, raw.get(c, 0), f"{raw.get(c,0):,}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, bal.get(c, 0), f"{bal.get(c,0):,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    print(f"\n  saved {out_png}")


# ── range calibration ─────────────────────────────────────────────────────────

def _round_up_sig(x: float, sig: int = 3) -> float:
    if x <= 0:
        return 0.0
    mag = math.floor(math.log10(x))
    factor = 10 ** (mag - sig + 1)
    return math.ceil(x / factor) * factor


def calibrate_ranges(df: pd.DataFrame, lo_pct=0.1, hi_pct=99.9):
    banner("FEATURE RANGE CALIBRATION (minimal quantization loss)")
    new_ranges = {}
    print(f"  {'feature':<22}{'old range':>26}{'new range':>26}{'clip%':>8}")
    print(f"  {'-'*22}{'-'*26:>26}{'-'*26:>26}{'-'*8:>8}")
    for name in FEATURE_ORDER:
        col = df[name].values.astype(np.float64)
        lo = max(0.0, float(np.percentile(col, lo_pct)))
        hi = float(np.percentile(col, hi_pct))
        hi = _round_up_sig(hi, 3)
        if hi <= lo:
            hi = lo + 1.0
        new_ranges[name] = (round(lo, 6), float(hi))
        clipped = np.mean((col < lo) | (col > hi)) * 100.0
        old = FEATURE_RANGES[name]
        print(f"  {name:<22}{f'({old[0]:g}, {old[1]:g})':>26}"
              f"{f'({new_ranges[name][0]:g}, {new_ranges[name][1]:g})':>26}{clipped:>7.3f}")
    return new_ranges


def write_ranges_to_constants(new_ranges: dict) -> None:
    with open(CONSTANTS_PATH, "r") as f:
        text = f.read()

    lines = [BEGIN_MARK,
             "# Recalibrated from dataset_final.csv by build_dataset.py",
             "# (robust 0.1 / 99.9 percentiles). Keys/order match FEATURE_ORDER.",
             "FEATURE_RANGES = {"]
    width = max(len(n) for n in FEATURE_ORDER) + 4
    for name in FEATURE_ORDER:
        lo, hi = new_ranges[name]
        key = f'"{name}":'
        lines.append(f"    {key:<{width}}({lo:g}, {hi:g}),")
    lines.append("}")
    lines.append(END_MARK)
    block = "\n".join(lines)

    pattern = re.compile(re.escape(BEGIN_MARK) + r".*?" + re.escape(END_MARK), re.DOTALL)
    if not pattern.search(text):
        raise RuntimeError("FEATURE_RANGES markers not found in constants.py")
    text = pattern.sub(block, text)
    with open(CONSTANTS_PATH, "w") as f:
        f.write(text)
    print(f"\n  rewrote FEATURE_RANGES block in {CONSTANTS_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="training_features.csv")
    p.add_argument("--output", default="dataset_final.csv")
    p.add_argument("--dist_png", default="dataset_distribution.png")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_calibrate", action="store_true",
                   help="Skip rewriting FEATURE_RANGES in constants.py.")
    args = p.parse_args()

    banner("STEP 1 — LOAD FEATURES")
    df = pd.read_csv(args.input)
    print(f"  loaded {len(df):,} rows from {args.input}")

    df_bal, raw_counts, bal_counts = balance(df, args.seed)
    plot_distribution(raw_counts, bal_counts, args.dist_png)

    new_ranges = calibrate_ranges(df_bal)
    if not args.no_calibrate:
        write_ranges_to_constants(new_ranges)
    else:
        print("\n  (--no_calibrate) constants.py left unchanged")

    df_bal.to_csv(args.output, index=False)
    banner(f"SAVED {len(df_bal):,} BALANCED ROWS → {args.output}")
    print("  Next: python3 train_fp.py   and   python3 train_qat.py")


if __name__ == "__main__":
    main()
