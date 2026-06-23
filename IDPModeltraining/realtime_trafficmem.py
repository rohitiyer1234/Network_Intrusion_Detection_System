#!/usr/bin/env python3

print("SCRIPT STARTED")

import pandas as pd
import numpy as np

# ============================================================
# INPUT EXCEL FILE
# ============================================================

EXCEL_FILE = "output.xlsx"

# ============================================================
# FEATURE ORDER
# MUST MATCH TRAINING / FPGA ORDER
# ============================================================

FEATURE_ORDER = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "SYN Flag Count",
    "RST Flag Count",
    "Avg Fwd Segment Size"
]

# ============================================================
# FEATURE RANGES
# USE EXACT TRAINING RANGES IF DIFFERENT
# ============================================================

FEATURE_RANGES = {
    "Flow Duration":            (0, 120),
    "Total Fwd Packets":        (0, 1000),
    "Total Backward Packets":   (0, 1000),
    "Fwd Packets Length Total": (0, 100000),
    "Bwd Packets Length Total": (0, 100000),
    "Fwd Packet Length Max":    (0, 1500),
    "Fwd Packet Length Min":    (0, 1500),
    "Bwd Packet Length Max":    (0, 1500),
    "Bwd Packet Length Min":    (0, 1500),
    "Flow Bytes/s":             (0, 10000000),
    "Flow Packets/s":           (0, 1000000),
    "Flow IAT Mean":            (0, 1),
    "Flow IAT Std":             (0, 1),
    "Fwd IAT Mean":             (0, 1),
    "Bwd IAT Mean":             (0, 1),
    "Fwd PSH Flags":            (0, 10),
    "Bwd PSH Flags":            (0, 10),
    "SYN Flag Count":           (0, 20),
    "RST Flag Count":           (0, 20),
    "Avg Fwd Segment Size":     (0, 1500),
}

# ============================================================
# NORMALIZATION
# SAME AS TRAINING
# ============================================================

def normalize_feature(name, value):

    mn, mx = FEATURE_RANGES[name]

    value = float(value)

    # clip to training range
    value = np.clip(value, mn, mx)

    # normalize to [0,1]
    value = (value - mn) / (mx - mn)

    # map to [-1,1]
    value = (value * 2.0) - 1.0

    return value


# ============================================================
# QUANTIZATION
# SAME AS QAT TRAINING
# ============================================================

def quantize_int8(x):

    q = int(round(x * 127.0))

    q = max(-128, min(127, q))

    return q


# ============================================================
# LOAD ONLY FIRST 3 ROWS
# ============================================================

print("Loading Excel file...")

df = pd.read_excel(
    EXCEL_FILE,
    engine="openpyxl",
    nrows=3
)

print("Excel loaded successfully")
print("Rows loaded:", len(df))

# ============================================================
# VERIFY FEATURES EXIST
# ============================================================

for feat in FEATURE_ORDER:

    if feat not in df.columns:

        print("\nAvailable columns:\n")
        print(df.columns.tolist())

        raise ValueError(f"\nMissing feature column: {feat}")

# ============================================================
# GENERATE MEM FILE
# ============================================================

print("\nGenerating tb_input.mem ...")

with open("tb_input.mem", "w") as memfile:

    for row_idx, row in df.iterrows():

        quantized_features = []

        print(f"\n========== SAMPLE {row_idx} ==========")

        for feat in FEATURE_ORDER:

            raw_value = row[feat]

            norm_value = normalize_feature(feat, raw_value)

            q = quantize_int8(norm_value)

            quantized_features.append(q)

            print(
                f"{feat:30s} "
                f"Raw={raw_value} "
                f"Norm={norm_value:.4f} "
                f"Q={q}"
            )

        # signed int8 -> hex byte
        hex_values = []

        for q in quantized_features:

            hex_byte = q & 0xFF

            hex_values.append(f"{hex_byte:02X}")

        line = " ".join(hex_values)

        memfile.write(line + "\n")

        print("\nHEX LINE:")
        print(line)

print("\nGenerated tb_input.mem successfully")