"""
tools/validate_numpy_model.py — Offline sanity check for the exported
NumPy MLP weights. Not part of the live pipeline; run manually after
export_model.py to confirm the .npz produces sane class probabilities
on a few hand-built feature vectors plus a random stress test.

Run from anywhere:
    python3 tools/validate_numpy_model.py
"""

import os
import sys
import time
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PYNQ_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "PYNQ"))
sys.path.insert(0, PYNQ_DIR)

from numpy_model import NumpyMLP

WEIGHTS_PATH = os.path.join(PYNQ_DIR, "model_fp.npz")


# ============================================================
# Utility
# ============================================================

def print_header(title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_test(model, name, feature_vector):

    print_header(f"TEST: {name}")

    print("\nFeature Vector:\n")
    print(np.array(feature_vector))

    t0 = time.perf_counter()

    result = model.predict(feature_vector)

    latency_ms = (
        (time.perf_counter() - t0)
        * 1000.0
    )

    print("\nPrediction:\n")

    print(
        f"Class ID      : {result['class_id']}"
    )

    print(
        f"Class Name    : {result['class_name']}"
    )

    print(
        f"Confidence    : "
        f"{result['confidence']:.6f}"
    )

    print(
        f"Inference Time: "
        f"{latency_ms:.3f} ms"
    )

    print("\nProbabilities:\n")

    for k, v in result["probabilities"].items():

        print(
            f"{k:<12s}: {v:.6f}"
        )

    print("\n")


# ============================================================
# Main
# ============================================================

model = NumpyMLP(
    WEIGHTS_PATH
)

print_header(
    "MODEL VALIDATION STARTED"
)


# ============================================================
# PER-CLASS SAMPLES (drawn from the real balanced dataset)
#
# Hand-built vectors are unreliable once FEATURE_RANGES are recalibrated, so we
# pull one representative real flow per class from dataset_final.csv when it is
# available. Falls back to the random stress test if the CSV is absent.
# ============================================================

from constants import FEATURE_ORDER  # noqa: E402

DATASET_CSV = os.path.normpath(os.path.join(
    THIS_DIR, "..", "..", "IDPModeltraining", "dataset_final.csv"))

if os.path.exists(DATASET_CSV):
    import csv
    seen = {}
    with open(DATASET_CSV) as fh:
        for row in csv.DictReader(fh):
            lbl = row.get("label")
            if lbl in ("normal", "ddos", "botnet") and lbl not in seen:
                seen[lbl] = np.array([float(row[f]) for f in FEATURE_ORDER],
                                     dtype=np.float32)
            if len(seen) == 3:
                break
    for lbl in ("normal", "ddos", "botnet"):
        if lbl in seen:
            run_test(model, f"{lbl.upper()} (real sample)", seen[lbl])
else:
    print(f"\n[info] {DATASET_CSV} not found — skipping per-class real samples; "
          f"see the random stress test below.")


# ============================================================
# RANDOM STRESS TEST
# ============================================================

print_header(
    "RANDOM STRESS TEST"
)

latencies = []

for i in range(100):

    vec = np.random.rand(20).astype(
        np.float32
    )

    t0 = time.perf_counter()

    result = model.predict(vec)

    latency_ms = (
        (time.perf_counter() - t0)
        * 1000.0
    )

    latencies.append(latency_ms)

    if i < 5:

        print(
            f"Sample {i:02d} -> "
            f"{result['class_name']} "
            f"({result['confidence']:.4f}) "
            f"{latency_ms:.3f} ms"
        )

print("\n")

print(
    f"Average Inference Time : "
    f"{np.mean(latencies):.3f} ms"
)

print(
    f"Min Inference Time     : "
    f"{np.min(latencies):.3f} ms"
)

print(
    f"Max Inference Time     : "
    f"{np.max(latencies):.3f} ms"
)

print("\nValidation Complete.\n")