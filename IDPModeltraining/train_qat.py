#!/usr/bin/env python3
"""
train_qat.py  —  Quantization-Aware Training of the hardware (FPGA) MLP.

The hardware datapath is BatchNorm-free: Linear → MAC clamp → hardware-ReLU
(clamp) → arithmetic-shift requantize, with INT8 activations/weights.  This
script fine-tunes that datapath with straight-through fake-quant and auto-
calibrated per-layer shifts.

BatchNorm folding
-----------------
The model is initialized from the BN-FOLDED floating-point weights
(model_fp.npz produced by export_fp_npz.py).  Because the QAT model has no
BatchNorm, the folded weights ARE the weights that get quantized — so "BN
constants folded into the weights/biases" holds by construction.  Layer-1 is
remapped for the [0,1]→[-1,1] input rescaling so the init starts close to FP.

    python3 train_qat.py                       # full run
    python3 train_qat.py --epochs 5            # quick run
    python3 train_qat.py --init model_fp.npz   # custom FP init
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from constants import FEATURE_ORDER, FEATURE_RANGES, CLASS_NAMES
import eval_report

LABEL_MAP = {CLASS_NAMES[i]: i for i in CLASS_NAMES}
CLASS_LIST = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]


def banner(t): print(f"\n{'='*70}\n{t}\n{'='*70}", flush=True)


# ── quantization primitives (straight-through) ───────────────────────────────

def ste_round(x):
    return x + (torch.round(x) - x).detach()


def quant_int8(x):                              # signed INT8 fake-quant of [-1,1] input
    return torch.clamp(ste_round(x * 127.0), -127, 127) / 127.0


def relu_hw(x):                                 # hardware ReLU clamp
    return torch.clamp(x, 0, 8192)


def requantize_hw(x, shift):
    return torch.clamp(ste_round(x / float(2 ** shift)), -127, 127)


def compute_shift(max_abs):
    return max(0, int(np.ceil(np.log2(max(float(max_abs), 1.0) / 127.0))))


# ── model ─────────────────────────────────────────────────────────────────────

class QuantMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.quant_enabled = False
        self.quant_strength = 0.0
        self.shift_l1 = 0
        self.shift_l2 = 0

    def _blend(self, x, x_q):
        s = self.quant_strength
        return s * x_q + (1.0 - s) * x

    def forward(self, x, collect_stats=False):
        if self.quant_enabled:
            x = self._blend(x, quant_int8(x))
        x = torch.clamp(self.fc1(x), -524288, 524287)
        l1_max = x.abs().max().item()
        x = relu_hw(x)
        if self.quant_enabled:
            x = self._blend(x, requantize_hw(x, self.shift_l1))
        x = torch.clamp(self.fc2(x), -1048576, 1048575)
        l2_max = x.abs().max().item()
        x = relu_hw(x)
        if self.quant_enabled:
            x = self._blend(x, requantize_hw(x, self.shift_l2))
        x = self.fc3(x)
        return (x, l1_max, l2_max) if collect_stats else x


def init_from_fp(model, npz_path):
    """Init from BN-folded FP weights, remapping fc1 for [0,1]→[-1,1] input."""
    d = np.load(npz_path, allow_pickle=True)
    W1 = d["W1"].astype(np.float32); b1 = d["b1"].astype(np.float32)
    # x_fp in [0,1], x_qat in [-1,1]:  x_fp = (x_qat+1)/2
    # W1·x_fp + b1 = (W1/2)·x_qat + (b1 + W1.sum(1)/2)
    W1r = W1 / 2.0
    b1r = b1 + W1.sum(axis=1) / 2.0
    with torch.no_grad():
        model.fc1.weight.copy_(torch.tensor(W1r)); model.fc1.bias.copy_(torch.tensor(b1r))
        model.fc2.weight.copy_(torch.tensor(d["W2"])); model.fc2.bias.copy_(torch.tensor(d["b2"]))
        model.fc3.weight.copy_(torch.tensor(d["W3"])); model.fc3.bias.copy_(torch.tensor(d["b3"]))
    print(f"  initialized QAT model from BN-folded FP weights: {npz_path}")


def normalize_signed(X):
    mins = np.array([FEATURE_RANGES[f][0] for f in FEATURE_ORDER], dtype=np.float32)
    maxs = np.array([FEATURE_RANGES[f][1] for f in FEATURE_ORDER], dtype=np.float32)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    X = np.clip(X, mins, maxs)
    X = (X - mins) / span                       # [0,1]
    return (X * 2.0 - 1.0).astype(np.float32)   # [-1,1]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); preds, ys = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu().numpy()); ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(preds)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="dataset_final.csv")
    p.add_argument("--out", default="model_qat.pth")
    p.add_argument("--init", default="model_fp.npz",
                   help="BN-folded FP npz for init (use '' to train from scratch).")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--qat_start", type=int, default=5)
    p.add_argument("--qat_ramp", type=int, default=10)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    banner("QAT INT8 TRAINING (BN-free hardware MLP, 3 classes)")
    df = pd.read_csv(args.data)
    df = df[df["label"].isin(LABEL_MAP)]
    X = normalize_signed(df[FEATURE_ORDER].values.astype(np.float32))
    y = df["label"].map(LABEL_MAP).values.astype(np.int64)
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)
    print(f"  device={device}  train={len(Xtr):,}  val={len(Xval):,}  classes={CLASS_LIST}")
    print(f"  epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"qat_start={args.qat_start} ramp={args.qat_ramp}")

    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=args.batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.tensor(Xval), torch.tensor(yval)),
                    batch_size=args.batch_size)

    model = QuantMLP().to(device)
    if args.init:
        init_from_fp(model, args.init)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0; best_ep = 0; patience = 0
    banner("TRAINING")
    for ep in range(1, args.epochs + 1):
        if ep >= args.qat_start:
            model.quant_enabled = True
            model.quant_strength = min(1.0, (ep - args.qat_start + 1) / args.qat_ramp)

        model.train(); tot_loss = c = n = 0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step()
            tot_loss += loss.item()
            c += (out.argmax(1) == yb).sum().item(); n += len(yb)
        tr_acc = c / n

        # Once QAT is on, calibrate + evaluate under FULL quantization so the
        # measured accuracy reflects the deployed INT8 datapath (not the ramp).
        eval_strength = model.quant_strength
        if model.quant_enabled:
            model.quant_strength = 1.0

        # calibrate shifts from validation activation ranges
        model.eval(); l1g = l2g = 0.0
        with torch.no_grad():
            for xb, _ in va:
                _, l1m, l2m = model(xb.to(device), collect_stats=True)
                l1g = max(l1g, l1m); l2g = max(l2g, l2m)
        model.shift_l1 = compute_shift(l1g); model.shift_l2 = compute_shift(l2g)

        yt, yp = evaluate(model, va, device)
        val_acc = float((yt == yp).mean())
        model.quant_strength = eval_strength       # restore ramp for next epoch's training
        history["loss"].append(tot_loss); history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        # Only checkpoint epochs that are FULLY quantized — a full-precision
        # warmup epoch must never win "best" (it collapses under INT8).
        fully_quant = model.quant_enabled and eval_strength >= 1.0
        flag = ""
        if fully_quant and val_acc > best_acc:
            best_acc = val_acc; best_ep = ep; patience = 0
            torch.save({"state_dict": model.state_dict(),
                        "shift_l1": model.shift_l1, "shift_l2": model.shift_l2},
                       args.out)
            flag = "  ← best (saved)"
        elif fully_quant:
            patience += 1
        print(f"  epoch {ep:02d}/{args.epochs} | loss={tot_loss:8.3f} | "
              f"train_acc={tr_acc:.4f} | val_acc={val_acc:.4f} | "
              f"shifts L1={model.shift_l1} L2={model.shift_l2} | "
              f"qstr={model.quant_strength:.2f}{flag}")
        if patience >= args.patience:
            print(f"  early stopping (no improvement for {args.patience} epochs)")
            break

    if best_ep == 0:        # never reached full quantization → save final state
        model.quant_enabled = True; model.quant_strength = 1.0
        torch.save({"state_dict": model.state_dict(),
                    "shift_l1": model.shift_l1, "shift_l2": model.shift_l2}, args.out)
        print("  (no fully-quantized epoch beat baseline; saved final model)")

    banner(f"BEST VAL ACCURACY = {best_acc:.4f} @ epoch {best_ep}  →  {args.out}")

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.shift_l1, model.shift_l2 = ckpt["shift_l1"], ckpt["shift_l2"]
    model.quant_enabled = True; model.quant_strength = 1.0
    yt, yp = evaluate(model, va, device)
    m = eval_report.classification_metrics(yt, yp, CLASS_LIST)
    eval_report.print_metrics_table("QAT INT8 model — validation set", m)
    eval_report.plot_confusion_matrix(yt, yp, CLASS_LIST, "qat_confusion_matrix.png",
                                      title="QAT confusion")
    eval_report.plot_metric_bars(m, "qat_metrics.png", title="QAT per-class metrics")
    eval_report.plot_training_curves(history, "qat_training_curves.png", title="QAT training")
    print(f"\n  final calibrated shifts: L1={model.shift_l1}  L2={model.shift_l2}")


if __name__ == "__main__":
    main()
