#!/usr/bin/env python3
"""
npz_to_mem.py  —  Generate FPGA .mem files from model_qat.npz.

Reads the INT8 weights / INT16 biases from model_qat.npz (no torch needed) and
writes hardware memory-init files for the 3-layer MLP:

    layer1_weights.mem  layer1_bias.mem
    layer2_weights.mem  layer2_bias.mem
    layer3_weights.mem  layer3_bias.mem

Weights are emitted as 2-hex-digit INT8 (two's complement), grouped into PAR
lanes per output neuron with zero-padded lanes so each group is full width —
matching the PE-array layout (fc1/fc2 use PAR, fc3 uses its 3 output neurons).
Biases are emitted as 4-hex-digit INT16.

    python3 npz_to_mem.py --model model_qat.npz --outdir .
"""

import argparse
import os
import numpy as np


def dump_weights_grouped(W, filename, par):
    """W: int8 (out_dim, in_dim).  PAR-lane grouped row-major hex dump."""
    out_dim, in_dim = W.shape
    groups = (out_dim + par - 1) // par
    with open(filename, "w") as f:
        for g in range(groups):
            for lane in range(par):
                nidx = g * par + lane
                if nidx < out_dim:
                    for val in W[nidx]:
                        f.write(f"{int(val) & 0xFF:02X}\n")
                else:
                    for _ in range(in_dim):
                        f.write("00\n")
    return groups * par * in_dim


def dump_bias(b, filename):
    """b: int16 (out_dim,).  4-hex-digit dump."""
    with open(filename, "w") as f:
        for val in b:
            f.write(f"{int(val) & 0xFFFF:04X}\n")
    return len(b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model_qat.npz")
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    d = np.load(args.model, allow_pickle=True)
    par = int(d["PAR"])
    os.makedirs(args.outdir, exist_ok=True)

    # Per-layer weight shifts (dequant weight = W_int8 / 2^wshift); default 0 for
    # backward compatibility with older npz files.
    wshift = [int(d[k]) if k in d.files else 0 for k in ("wshift1", "wshift2", "wshift3")]

    print(f"Loaded {args.model}  PAR={par}  "
          f"act_shifts L1={int(d['shift_l1'])} L2={int(d['shift_l2'])}  "
          f"weight_shifts={wshift}")

    layers = [("layer1", d["W1"], d["b1"], par),
              ("layer2", d["W2"], d["b2"], par),
              ("layer3", d["W3"], d["b3"], d["W3"].shape[0])]   # fc3 grouped by its 3 neurons

    for name, W, b, group in layers:
        wf = os.path.join(args.outdir, f"{name}_weights.mem")
        bf = os.path.join(args.outdir, f"{name}_bias.mem")
        nw = dump_weights_grouped(W, wf, group)
        nb = dump_bias(b, bf)
        print(f"  {name}: W{W.shape} group={group} → {wf} ({nw} lines), "
              f"b{b.shape} → {bf} ({nb} lines)")

    # Emit the fixed-point shift schedule the FPGA requant needs (weight scale +
    # activation requant), so the hardware can fold 2^wshift back in.
    sf = os.path.join(args.outdir, "shifts.mem")
    with open(sf, "w") as f:
        f.write("# layer  weight_shift  act_shift\n")
        f.write(f"layer1 {wshift[0]} {int(d['shift_l1'])}\n")
        f.write(f"layer2 {wshift[1]} {int(d['shift_l2'])}\n")
        f.write(f"layer3 {wshift[2]} 0\n")
    print(f"  shifts → {sf}  (weight_shifts={wshift})")

    print("✔ MEM files generated")


if __name__ == "__main__":
    main()
