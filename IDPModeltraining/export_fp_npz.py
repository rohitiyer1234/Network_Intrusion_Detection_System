#!/usr/bin/env python3
"""
export_fp_npz.py  —  Fold BatchNorm into the Linear layers and export the
floating-point model as model_fp.npz for torch-free NumPy inference.

The .npz is self-contained: folded float32 weights/biases + the normalization
ranges (feat_min/feat_max in FEATURE_ORDER) + class names.

    python3 export_fp_npz.py --model model_fp.pth --out model_fp.npz
"""

import argparse
import numpy as np
import torch

from constants import FEATURE_ORDER, FEATURE_RANGES, CLASS_NAMES
from model import TrafficClassifier


def fold_bn(fc, bn):
    """Fold a BatchNorm1d into the preceding Linear → effective (W', b')."""
    W = fc.weight.data.clone()
    b = fc.bias.data.clone()
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight.data / std
    W_fold = W * scale.reshape(-1, 1)
    b_fold = (b - bn.running_mean) * scale + bn.bias.data
    return W_fold.numpy().astype(np.float32), b_fold.numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model_fp.pth")
    p.add_argument("--out", default="model_fp.npz")
    args = p.parse_args()

    print(f"Loading {args.model} ...")
    model = TrafficClassifier()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    W1, b1 = fold_bn(model.fc1, model.bn1)
    W2, b2 = fold_bn(model.fc2, model.bn2)
    W3 = model.fc3.weight.data.numpy().astype(np.float32)
    b3 = model.fc3.bias.data.numpy().astype(np.float32)

    feat_min = np.array([FEATURE_RANGES[f][0] for f in FEATURE_ORDER], dtype=np.float32)
    feat_max = np.array([FEATURE_RANGES[f][1] for f in FEATURE_ORDER], dtype=np.float32)
    class_names = np.array([CLASS_NAMES[i] for i in sorted(CLASS_NAMES)])

    np.savez(args.out,
             W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
             feat_min=feat_min, feat_max=feat_max,
             feature_order=np.array(FEATURE_ORDER), class_names=class_names)

    print(f"Saved {args.out} (BatchNorm folded into weights/biases)")
    print(f"  shapes: W1{W1.shape} W2{W2.shape} W3{W3.shape}")
    print(f"  classes: {[str(c) for c in class_names]}")


if __name__ == "__main__":
    main()
