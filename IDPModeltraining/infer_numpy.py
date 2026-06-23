#!/usr/bin/env python3
"""
infer_numpy.py  —  Pure-NumPy software inference (NO torch).

Loads the floating-point model_fp.npz (BatchNorm already folded) and classifies
flows.  Useful as a torch-free reference and for embedding in lightweight
environments.

    # classify a CSV of 20-feature rows (FEATURE_ORDER columns, optional 'label')
    python3 infer_numpy.py --model model_fp.npz --csv dataset_final.csv --limit 10

    # classify one comma-separated feature vector
    python3 infer_numpy.py --model model_fp.npz --vector "0.1,3,4,..."
"""

import argparse
import numpy as np


def load(path):
    d = np.load(path, allow_pickle=True)
    return d


def normalize(X, feat_min, feat_max):
    span = np.where(feat_max - feat_min == 0, 1.0, feat_max - feat_min)
    X = np.clip(X, feat_min, feat_max)
    return (X - feat_min) / span


def forward(X, d):
    """X: (N,20) normalized.  Returns (probs, pred_idx)."""
    def relu(z): return np.maximum(z, 0.0)
    h = relu(X @ d["W1"].T + d["b1"])
    h = relu(h @ d["W2"].T + d["b2"])
    logits = h @ d["W3"].T + d["b3"]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs, probs.argmax(axis=1)


def predict_raw(X_raw, d):
    """X_raw: (N,20) un-normalized features in FEATURE_ORDER."""
    Xn = normalize(X_raw.astype(np.float64),
                   d["feat_min"].astype(np.float64), d["feat_max"].astype(np.float64))
    return forward(Xn, d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model_fp.npz")
    p.add_argument("--csv", help="CSV with the 20 FEATURE_ORDER columns (+ optional label).")
    p.add_argument("--vector", help="Single comma-separated 20-feature vector.")
    p.add_argument("--limit", type=int, default=20, help="Rows to print from --csv.")
    args = p.parse_args()

    d = load(args.model)
    feature_order = [str(c) for c in d["feature_order"]]
    class_names = [str(c) for c in d["class_names"]]
    print(f"Loaded {args.model}  classes={class_names}  (pure NumPy, no torch)")

    if args.vector:
        vec = np.array([float(x) for x in args.vector.split(",")], dtype=np.float64).reshape(1, -1)
        probs, idx = predict_raw(vec, d)
        print(f"  class={class_names[idx[0]]}  conf={probs[0, idx[0]]:.4f}")
        return

    if args.csv:
        import csv
        rows, labels = [], []
        with open(args.csv) as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                rows.append([float(r[c]) for c in feature_order])
                labels.append(r.get("label"))
        X = np.array(rows, dtype=np.float64)
        probs, idx = predict_raw(X, d)

        if any(labels) and all(labels):
            name_to_i = {c: i for i, c in enumerate(class_names)}
            yt = np.array([name_to_i.get(l, -1) for l in labels])
            mask = yt >= 0
            acc = float((idx[mask] == yt[mask]).mean()) if mask.any() else float("nan")
            print(f"  NumPy inference accuracy over {mask.sum():,} labeled rows: {acc:.4f}")

        print(f"\n  first {min(args.limit, len(X))} predictions:")
        for i in range(min(args.limit, len(X))):
            true = f"  (true={labels[i]})" if labels[i] else ""
            print(f"    row {i:>4}: {class_names[idx[i]]:<8} conf={probs[i, idx[i]]:.4f}{true}")
        return

    p.error("provide --csv or --vector")


if __name__ == "__main__":
    main()
