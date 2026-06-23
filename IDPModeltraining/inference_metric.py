#!/usr/bin/env python3

import time
import statistics
import numpy as np
import torch
import torch.nn as nn

# Single source of truth for feature order/ranges/class labels.
from constants import FEATURE_ORDER, FEATURE_RANGES, CLASS_NAMES as _CLASS_MAP

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "model_qat.pth"

NUM_RUNS = 1000

CLASS_NAMES = [_CLASS_MAP[i] for i in sorted(_CLASS_MAP)]   # [normal, ddos, botnet]

# ============================================================
# QUANTIZATION UTILS
# ============================================================

def ste_round(x):
    return x


def quant_int8(x):
    x = x * 127.0
    x = torch.round(x)
    x = torch.clamp(x, -128, 127)
    return x / 127.0


def relu_int16(x):
    return torch.clamp(x, 0, 8192)


def requantize(x, shift):
    scale = 2 ** shift
    x = x / scale
    x = torch.round(x)
    return torch.clamp(x, -128, 127)


# ============================================================
# MODEL
# ============================================================

SHIFT_L1 = 1
SHIFT_L2 = 0


class QuantMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):

        x = quant_int8(x)

        x = self.fc1(x)
        x = relu_int16(x)
        x = requantize(x, SHIFT_L1)

        x = self.fc2(x)
        x = relu_int16(x)
        x = requantize(x, SHIFT_L2)

        x = self.fc3(x)

        return x


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(feature_dict):

    out = []

    for feat in FEATURE_ORDER:

        value = float(feature_dict[feat])

        mn, mx = FEATURE_RANGES[feat]

        value = np.clip(value, mn, mx)

        value = (value - mn) / (mx - mn)

        value = (value * 2.0) - 1.0

        out.append(value)

    return np.array(out, dtype=np.float32)


# ============================================================
# SAMPLE INPUT
# REPLACE WITH REAL FEATURES
# ============================================================

sample_features = {

    "flow_duration":        0.5,
    "total_fwd_packets":    12,
    "total_bwd_packets":    10,
    "total_len_fwd_pkts":   6000,
    "total_len_bwd_pkts":   5200,
    "fwd_pkt_len_max":      800,
    "fwd_pkt_len_min":      40,
    "bwd_pkt_len_max":      700,
    "bwd_pkt_len_min":      40,
    "flow_bytes_per_sec":   22000,
    "flow_packets_per_sec": 44,
    "flow_iat_mean":        0.01,
    "flow_iat_std":         0.002,
    "fwd_iat_mean":         0.01,
    "bwd_iat_mean":         0.012,
    "fwd_psh_flags":        0,
    "bwd_psh_flags":        0,
    "syn_flag_count":       1,
    "rst_flag_count":       0,
    "avg_fwd_segment_size": 500
}

# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("Loading model...")
print("========================================")

device = torch.device("cpu")

model = QuantMLP()

_ckpt = torch.load(MODEL_PATH, map_location=device)
if isinstance(_ckpt, dict) and "state_dict" in _ckpt:
    model.load_state_dict(_ckpt["state_dict"])
    SHIFT_L1 = int(_ckpt.get("shift_l1", SHIFT_L1))
    SHIFT_L2 = int(_ckpt.get("shift_l2", SHIFT_L2))
else:
    model.load_state_dict(_ckpt)

model.eval()

print(f"Model loaded successfully (shifts L1={SHIFT_L1} L2={SHIFT_L2})")

# ============================================================
# PREPARE INPUT
# ============================================================

x = normalize_features(sample_features)

x_tensor = torch.tensor(x).unsqueeze(0)

# ============================================================
# WARMUP
# ============================================================

print("\nRunning warmup...")

with torch.no_grad():

    for _ in range(100):
        _ = model(x_tensor)

print("Warmup complete")

# ============================================================
# BENCHMARK
# ============================================================

print("\n========================================")
print("RUNNING INFERENCE BENCHMARK")
print("========================================")

latencies_us = []

with torch.no_grad():

    for _ in range(NUM_RUNS):

        t0 = time.perf_counter_ns()

        output = model(x_tensor)

        t1 = time.perf_counter_ns()

        latency_us = (t1 - t0) / 1000.0

        latencies_us.append(latency_us)

# ============================================================
# PREDICTION
# ============================================================

logits = output.squeeze().numpy()

pred_class = int(np.argmax(logits))

pred_name = CLASS_NAMES[pred_class]

# ============================================================
# STATISTICS
# ============================================================

avg_latency = statistics.mean(latencies_us)
min_latency = min(latencies_us)
max_latency = max(latencies_us)
std_latency = statistics.stdev(latencies_us)

throughput = 1_000_000.0 / avg_latency

# ============================================================
# OUTPUT REPORT
# ============================================================

print("\n========================================")
print("INFERENCE RESULTS")
print("========================================")

print(f"\nPredicted Class : {pred_name}")
print(f"Class ID        : {pred_class}")

print("\nLogits:")

for i, val in enumerate(logits):
    print(f"  {CLASS_NAMES[i]:12s}: {val:.4f}")

print("\n========================================")
print("LATENCY BENCHMARK")
print("========================================")

print(f"Runs                  : {NUM_RUNS}")
print(f"Average Latency (us)  : {avg_latency:.3f}")
print(f"Minimum Latency (us)  : {min_latency:.3f}")
print(f"Maximum Latency (us)  : {max_latency:.3f}")
print(f"Std Dev Latency (us)  : {std_latency:.3f}")

print("\n========================================")
print("THROUGHPUT")
print("========================================")

print(f"Inferences/sec        : {throughput:.2f}")

print("\n========================================")
print("SYSTEM INFO")
print("========================================")

print(f"Device                : CPU")
print(f"PyTorch Version       : {torch.__version__}")

print("\nBenchmark complete\n")