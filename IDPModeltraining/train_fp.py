#!/usr/bin/env python3
"""
train_fp.py  —  Floating-point training of the 3-class MLP (with BatchNorm).

Reads dataset_final.csv, normalizes with constants.FEATURE_RANGES ([0,1]),
trains TrafficClassifier (model.py), saves the best checkpoint to model_fp.pth,
and writes evaluation figures via eval_report.py.

    python3 train_fp.py            # full run
    python3 train_fp.py --epochs 5 # quick run
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from constants import FEATURE_ORDER, FEATURE_RANGES, CLASS_NAMES
from model import TrafficClassifier
import eval_report

LABEL_MAP = {CLASS_NAMES[i]: i for i in CLASS_NAMES}          # {normal:0, ddos:1, botnet:2}
CLASS_LIST = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]


def banner(t): print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)


def normalize(X):
    mins = np.array([FEATURE_RANGES[f][0] for f in FEATURE_ORDER], dtype=np.float32)
    maxs = np.array([FEATURE_RANGES[f][1] for f in FEATURE_ORDER], dtype=np.float32)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    X = np.clip(X, mins, maxs)
    return ((X - mins) / span).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset_final.csv")
    p.add_argument("--out", default="model_fp.pth")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    banner("FLOATING-POINT TRAINING (MLP + BatchNorm, 3 classes)")
    df = pd.read_csv(args.data)
    df = df[df["label"].isin(LABEL_MAP)]
    X = normalize(df[FEATURE_ORDER].values.astype(np.float32))
    y = df["label"].map(LABEL_MAP).values.astype(np.int64)

    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)
    print(f"  device={device}  train={len(Xtr):,}  val={len(Xval):,}  "
          f"classes={CLASS_LIST}\n  epochs={args.epochs} batch={args.batch_size} lr={args.lr}")

    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=args.batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.tensor(Xval), torch.tensor(yval)),
                    batch_size=args.batch_size)

    model = TrafficClassifier().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    history = {"loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    banner("TRAINING")
    for ep in range(1, args.epochs + 1):
        model.train(); tot_loss = c = n = 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step()
            tot_loss += loss.item()
            c += (out.argmax(1) == yb).sum().item(); n += len(yb)
        tr_acc = c / n

        model.eval(); c = n = 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                c += (model(xb).argmax(1) == yb).sum().item(); n += len(yb)
        val_acc = c / n
        history["loss"].append(tot_loss); history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc; torch.save(model.state_dict(), args.out); flag = "  ← best (saved)"
        print(f"  epoch {ep:02d}/{args.epochs} | loss={tot_loss:8.3f} | "
              f"train_acc={tr_acc:.4f} | val_acc={val_acc:.4f}{flag}")

    banner(f"BEST VAL ACCURACY = {best_acc:.4f}  →  {args.out}")

    # Final evaluation report on the validation set (load best weights)
    model.load_state_dict(torch.load(args.out, map_location=device)); model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in va:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
    yp = np.concatenate(preds)
    m = eval_report.classification_metrics(yval, yp, CLASS_LIST)
    eval_report.print_metrics_table("FP model — validation set", m)
    eval_report.plot_confusion_matrix(yval, yp, CLASS_LIST, "fp_confusion_matrix.png",
                                      title="FP confusion")
    eval_report.plot_metric_bars(m, "fp_metrics.png", title="FP per-class metrics")
    eval_report.plot_training_curves(history, "fp_training_curves.png", title="FP training")


if __name__ == "__main__":
    main()
