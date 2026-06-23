#!/usr/bin/env python3
"""
tools/evaluate_model.py

OPTIONAL offline evaluation suite for the NumPy-based NIDS MLP classifier.
NOT part of the live demo pipeline and NOT required to run the system.

Validates the exported NumPy inference model against a labeled CSV
dataset (CICIDS-style, columns = constants.FEATURE_ORDER + 'label') and
reports confusion matrix, accuracy, precision, recall, F1, and latency.

Requires (not in the core requirements.txt — install separately if you
want to use this tool):
    pip install pandas scikit-learn

Requires a dataset CSV that is NOT bundled with this repository. Point
--dataset at your own labeled flow dataset before running, e.g.:

    python3 tools/evaluate_model.py --dataset /path/to/labeled_flows.csv
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PYNQ_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "PYNQ"))
sys.path.insert(0, PYNQ_DIR)

from numpy_model import NumpyMLP
from constants import FEATURE_ORDER

ap = argparse.ArgumentParser(description="Offline NIDS model evaluation (optional tool)")
ap.add_argument("--dataset",
                default=os.path.normpath(os.path.join(
                    THIS_DIR, "..", "..", "IDPModeltraining", "dataset_final.csv")),
                help="Path to labeled CSV (columns: FEATURE_ORDER + 'label')")
ap.add_argument("--model", default=os.path.join(PYNQ_DIR, "model_fp.npz"),
                help="Path to exported .npz weights")
args = ap.parse_args()


# ============================================================
# Label Mapping
# ============================================================

LABEL_MAP = {
    "normal": 0,
    "ddos": 1,
    "botnet": 2
}

INV_LABEL_MAP = {
    0: "normal",
    1: "ddos",
    2: "botnet"
}


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = args.dataset

MODEL_PATH = args.model


# ============================================================
# Load Model
# ============================================================

print("\n===================================================")
print(" NumPy-MLP NIDS Evaluation Suite")
print("===================================================\n")

print("[INFO] Loading NumPy inference model...")

model = NumpyMLP(MODEL_PATH)

print("[INFO] Model loaded successfully.\n")


# ============================================================
# Load Dataset
# ============================================================

print("[INFO] Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"[INFO] Original samples : {len(df)}")

# Keep only supported labels
df = df[
    df["label"].isin(LABEL_MAP.keys())
]

print(f"[INFO] Filtered samples: {len(df)}\n")


# ============================================================
# Feature Extraction
# ============================================================

print("[INFO] Extracting feature matrix...")

X = df[FEATURE_ORDER].values.astype(np.float32)

y_true = df["label"].map(
    LABEL_MAP
).values

print(f"[INFO] Feature matrix shape: {X.shape}\n")


# ============================================================
# Inference
# ============================================================

print("[INFO] Running NumPy inference...\n")

predictions = []

latencies_ms = []

start_eval = time.perf_counter()

for i in range(len(X)):

    vec = X[i]

    t0 = time.perf_counter()

    result = model.predict(vec)

    latency_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    latencies_ms.append(latency_ms)

    predictions.append(
        result["class_id"]
    )

    # Progress logging
    if (i + 1) % 5000 == 0:

        print(
            f"[INFO] Processed "
            f"{i+1}/{len(X)} samples"
        )

end_eval = time.perf_counter()

print("\n[INFO] Inference completed.\n")


# ============================================================
# Metrics
# ============================================================

y_pred = np.array(predictions)

acc = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred
)

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "normal",
        "ddos",
        "botnet"
    ],
    digits=4,
    zero_division=0
)


# ============================================================
# Results
# ============================================================

print("===================================================")
print(" EVALUATION RESULTS")
print("===================================================\n")

print(f"Accuracy  : {acc:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1-score  : {f1:.6f}")

print("\n===================================================")
print(" LATENCY STATISTICS")
print("===================================================\n")

print(
    f"Average inference latency : "
    f"{np.mean(latencies_ms):.6f} ms"
)

print(
    f"Minimum inference latency : "
    f"{np.min(latencies_ms):.6f} ms"
)

print(
    f"Maximum inference latency : "
    f"{np.max(latencies_ms):.6f} ms"
)

print(
    f"Total evaluation runtime  : "
    f"{end_eval - start_eval:.3f} s"
)

print("\n===================================================")
print(" CONFUSION MATRIX")
print("===================================================\n")

print(cm)

print("\n===================================================")
print(" CLASSIFICATION REPORT")
print("===================================================\n")

print(report)


# ============================================================
# Misclassification Analysis
# ============================================================

print("\n===================================================")
print(" MISCLASSIFICATION SUMMARY")
print("===================================================\n")

misclassified = np.where(
    y_true != y_pred
)[0]

print(
    f"Total misclassified samples: "
    f"{len(misclassified)}"
)

if len(misclassified) > 0:

    print("\nFirst 10 misclassified examples:\n")

    for idx in misclassified[:10]:

        print(
            f"Sample {idx:6d} | "
            f"True: {INV_LABEL_MAP[y_true[idx]]:<10s} | "
            f"Pred: {INV_LABEL_MAP[y_pred[idx]]:<10s}"
        )

print("\n===================================================")
print(" Evaluation Complete")
print("===================================================\n")
