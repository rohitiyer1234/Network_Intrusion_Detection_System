#!/usr/bin/env python3
"""
export_model.py — Convert a trained PyTorch MLP checkpoint into the
dependency-free NumPy .npz used by the PYNQ ARM (numpy_model.py).

BatchNorm is FOLDED into the linear layers so the PYNQ forward needs no BN, and
the per-feature normalization bounds + class names are embedded, making the npz
fully self-contained.

Run from anywhere (paths resolved relative to this script):

    python3 LAPTOP/export_model.py \
        --model ../IDPModeltraining/model_fp.pth \
        --out   PYNQ/model_fp.npz

Default source is IDPModeltraining/model_fp.pth (the 3-class normal/ddos/botnet
model). Output layout matches model_fp.npz: W1,b1,W2,b2,W3,b3 (BN folded),
feat_min, feat_max, feature_order, class_names.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PYNQ_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "PYNQ"))
sys.path.insert(0, PYNQ_DIR)
from constants import FEATURE_ORDER, FEATURE_RANGES, CLASS_NAMES   # noqa: E402


class MLP(nn.Module):
    """Architecture matching IDPModeltraining/model.py TrafficClassifier."""

    def __init__(self, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, n_classes)


def fold_bn(fc, bn):
    """Fold BatchNorm1d into the preceding Linear → effective (W', b')."""
    W = fc.weight.data.clone()
    b = fc.bias.data.clone()
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight.data / std
    W_fold = W * scale.reshape(-1, 1)
    b_fold = (b - bn.running_mean) * scale + bn.bias.data
    return W_fold.numpy().astype(np.float32), b_fold.numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",
                    default=os.path.normpath(os.path.join(
                        THIS_DIR, "..", "..", "IDPModeltraining", "model_fp.pth")),
                    help="Trained .pth (MLP+BN, 3 classes)")
    ap.add_argument("--out", default=os.path.join(PYNQ_DIR, "model_fp.npz"))
    args = ap.parse_args()

    class_names = np.array([CLASS_NAMES[i] for i in sorted(CLASS_NAMES)])
    model = MLP(len(class_names))
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    W1, b1 = fold_bn(model.fc1, model.bn1)
    W2, b2 = fold_bn(model.fc2, model.bn2)
    W3 = model.fc3.weight.data.numpy().astype(np.float32)
    b3 = model.fc3.bias.data.numpy().astype(np.float32)

    feat_min = np.array([FEATURE_RANGES[f][0] for f in FEATURE_ORDER], dtype=np.float32)
    feat_max = np.array([FEATURE_RANGES[f][1] for f in FEATURE_ORDER], dtype=np.float32)

    np.savez(args.out,
             W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
             feat_min=feat_min, feat_max=feat_max,
             feature_order=np.array(FEATURE_ORDER), class_names=class_names)

    print(f"Folded BN and exported → {args.out}")
    print(f"  W1{W1.shape} W2{W2.shape} W3{W3.shape}  classes={list(class_names)}")


if __name__ == "__main__":
    main()
