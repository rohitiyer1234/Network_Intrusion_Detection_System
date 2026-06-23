#!/usr/bin/env python3
"""
export_qat_npz.py  —  Quantize the trained QAT model and export model_qat.npz.

Weights → INT8 (round(w*127), clip[-128,127]);  biases → INT16 (round, clip).
The .npz also carries the calibrated arithmetic-shift amounts and PAR, and is
the single source consumed by npz_to_mem.py and for INT8 NumPy correlation.

    python3 export_qat_npz.py --model model_qat.pth --out model_qat.npz
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

from constants import CLASS_NAMES

PAR = 32


class QuantMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)


def weight_shift(w):
    """Largest power-of-2 scale k such that max|w|*2^k <= 127 (symmetric INT8).

    The trained QAT weights are NOT bounded to [-1,1] (they range to ~±13), so a
    fixed `round(w*127)` saturates ~30% of them. A per-layer power-of-2 scale keeps
    the full INT8 range with zero saturation AND maps to a hardware right-shift, so
    the FPGA requant can fold it in.
    """
    amax = float(np.abs(w).max())
    if amax == 0.0:
        return 0
    k = 0
    while amax * (2 ** (k + 1)) <= 127.0:
        k += 1
    return k


def quant_w(w, k):
    w = np.round(w.detach().cpu().numpy() * (2 ** k))
    return np.clip(w, -127, 127).astype(np.int8)


def quant_b(b, bias_scale):
    """Quantize a bias into its layer's INT accumulator domain.

    The accumulator before requant has scale `bias_scale = input_scale * 2^wshift`
    (input_scale = 127 for layer-1 INT8 inputs, 1 for the requantized integer
    activations feeding layers 2/3). Rounding the bias to a plain integer (the old
    behaviour) zeroed the small layer-3 biases; scaling first preserves them.
    """
    b = np.round(b.detach().cpu().numpy() * bias_scale)
    return np.clip(b, -32768, 32767).astype(np.int16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model_qat.pth")
    p.add_argument("--out", default="model_qat.npz")
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    shift_l1 = int(ckpt.get("shift_l1", 0)) if isinstance(ckpt, dict) else 0
    shift_l2 = int(ckpt.get("shift_l2", 0)) if isinstance(ckpt, dict) else 0

    model = QuantMLP()
    model.load_state_dict(state)
    model.eval()

    # Per-layer power-of-2 weight scales (dequant weight = W_int8 / 2^wshift).
    k1 = weight_shift(model.fc1.weight.detach().cpu().numpy())
    k2 = weight_shift(model.fc2.weight.detach().cpu().numpy())
    k3 = weight_shift(model.fc3.weight.detach().cpu().numpy())

    # Bias accumulator scales: layer-1 sees INT8 inputs (scale 127); layers 2/3
    # see requantized integer activations (scale 1).
    W1, b1 = quant_w(model.fc1.weight, k1), quant_b(model.fc1.bias, 127 * (2 ** k1))
    W2, b2 = quant_w(model.fc2.weight, k2), quant_b(model.fc2.bias, 1 * (2 ** k2))
    W3, b3 = quant_w(model.fc3.weight, k3), quant_b(model.fc3.bias, 1 * (2 ** k3))
    class_names = np.array([CLASS_NAMES[i] for i in sorted(CLASS_NAMES)])

    np.savez(args.out,
             W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
             wshift1=np.int32(k1), wshift2=np.int32(k2), wshift3=np.int32(k3),
             shift_l1=np.int32(shift_l1), shift_l2=np.int32(shift_l2),
             PAR=np.int32(PAR), class_names=class_names)

    def _sat(w):
        return 100.0 * ((w >= 127) | (w <= -127)).mean()

    print(f"Saved {args.out}")
    print(f"  W1{W1.shape} int8 (wshift={k1}, sat={_sat(W1):.1f}%)  b1{b1.shape} int16")
    print(f"  W2{W2.shape} int8 (wshift={k2}, sat={_sat(W2):.1f}%)  b2{b2.shape} int16")
    print(f"  W3{W3.shape} int8 (wshift={k3}, sat={_sat(W3):.1f}%)  b3{b3.shape} int16")
    print(f"  activation shifts: L1={shift_l1} L2={shift_l2}  PAR={PAR}")
    print(f"  dequant: W_float ≈ W_int8 / 2^wshift   bias range b1=[{b1.min()},{b1.max()}]")


if __name__ == "__main__":
    main()
