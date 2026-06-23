#!/usr/bin/env python3
"""Validate the local traffic classifier on a CICIDS parquet dataset.

This script loads one or more CICIDS parquet files from a dataset directory,
extracts the 20 feature columns defined in constants.FEATURE_ORDER,
normalizes them using the same ranges from constants.FEATURE_RANGES,
and evaluates the model predictions against the dataset labels.

The label mapping is intentionally flexible to support common CICIDS label
values such as Benign, DDoS, PortScan, Botnet, Infiltration, and Web Attack.
"""
'''
import argparse
import os
from collections import Counter, defaultdict
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from constants import FEATURE_ORDER, FEATURE_RANGES
from model import CLASS_NAMES, load_model

LABEL_COLUMN_CANDIDATES = [
    "Label", "label", "CLASS", "Class", "attack_type", "Attack", "Category",
]

LABEL_TO_TARGET = {
    "benign": "normal",
    "normal": "normal",
    "ddos": "ddos",
    "dos": "ddos",
    "dos slowloris": "ddos",
    "dos slowhttptest": "ddos",
    "dos hulk": "ddos",
    "dos goldeneye": "ddos",
    "ddos hulk": "ddos",
    "ddos goldeneye": "ddos",
    "bot": "botnet",
    "botnet": "botnet",
    # Non-target classes (infiltration, brute force, port scan, web attacks,
    # sql injection, ...) are intentionally dropped — model has 3 classes.
}

CLASS_NAME_TO_ID = {name: idx for idx, name in CLASS_NAMES.items()}

DATASET_COLUMN_MAP = {
    "flow_duration":        "Flow Duration",
    "total_fwd_packets":    "Total Fwd Packets",
    "total_bwd_packets":    "Total Backward Packets",
    "total_len_fwd_pkts":   "Fwd Packets Length Total",
    "total_len_bwd_pkts":   "Bwd Packets Length Total",
    "fwd_pkt_len_max":      "Fwd Packet Length Max",
    "fwd_pkt_len_min":      "Fwd Packet Length Min",
    "bwd_pkt_len_max":      "Bwd Packet Length Max",
    "bwd_pkt_len_min":      "Bwd Packet Length Min",
    "flow_bytes_per_sec":   "Flow Bytes/s",
    "flow_packets_per_sec": "Flow Packets/s",
    "flow_iat_mean":        "Flow IAT Mean",
    "flow_iat_std":         "Flow IAT Std",
    "fwd_iat_mean":         "Fwd IAT Mean",
    "bwd_iat_mean":         "Bwd IAT Mean",
    "fwd_psh_flags":        "Fwd PSH Flags",
    "bwd_psh_flags":        "Bwd PSH Flags",
    "syn_flag_count":       "SYN Flag Count",
    "rst_flag_count":       "RST Flag Count",
    "avg_fwd_segment_size": "Avg Fwd Segment Size",
}


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    for name in LABEL_COLUMN_CANDIDATES:
        if name in df.columns:
            return name
    # Fallback: find the first string/categorical column outside the features
    for name in df.columns:
        if name not in FEATURE_ORDER and df[name].dtype == object:
            return name
    return None


def normalize_batch(values: np.ndarray) -> torch.Tensor:
    mins = np.array([FEATURE_RANGES[f][0] for f in FEATURE_ORDER], dtype=np.float32)
    maxs = np.array([FEATURE_RANGES[f][1] for f in FEATURE_ORDER], dtype=np.float32)
    spans = np.maximum(maxs - mins, 1e-9).astype(np.float32)
    values = np.clip(values, mins, maxs)
    values = (values - mins) / spans
    return torch.from_numpy(values.astype(np.float32))


def normalize_label(raw_label: object) -> Optional[str]:
    if pd.isna(raw_label):
        return None
    value = str(raw_label).strip().lower()
    if not value:
        return None
    if value in LABEL_TO_TARGET:
        return LABEL_TO_TARGET[value]
    if "benign" in value or "normal" in value:
        return "normal"
    if value == "bot" or "botnet" in value or value.startswith("bot"):
        return "botnet"
    if "dos" in value or "ddos" in value or "hulk" in value or "slowloris" in value or "goldeneye" in value:
        return "ddos"
    # everything else (port scan, infiltration, brute force, web attack, sql) → drop
    return None


def load_dataset_files(path: str, engine: str) -> pd.DataFrame:
    if os.path.isfile(path):
        paths = [path]
    else:
        paths = sorted(glob(os.path.join(path, "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {path}")

    frames = []
    for parquet_path in paths:
        print(f"Loading {parquet_path} ...")
        df = pd.read_parquet(parquet_path, engine=engine)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_confusion_matrix(rows: Iterable[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    for true_label, pred_label in rows:
        matrix[true_label][pred_label] += 1
    return {true: dict(preds) for true, preds in matrix.items()}


def summarize_confusion(matrix: Dict[str, Dict[str, int]]) -> None:
    labels = sorted(matrix.keys())
    header = ["true\\pred"] + labels
    print("\t".join(header))
    for true in labels:
        row = [true] + [str(matrix[true].get(pred, 0)) for pred in labels]
        print("\t".join(row))


def create_validation_report(
    df: pd.DataFrame,
    label_column: str,
    model_path: str,
    batch_size: int = 2048,
) -> None:
    dataset_features = [DATASET_COLUMN_MAP[f] for f in FEATURE_ORDER]
    missing = [f for f in dataset_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df = df[[*dataset_features, label_column]].copy()
    df[label_column] = df[label_column].apply(normalize_label)
    df = df.dropna(subset=[label_column])
    if df.empty:
        raise ValueError(f"No valid labels after normalization in column {label_column}")

    true_targets = df[label_column].astype(str).tolist()

    model = load_model(model_path)
    model = model.eval()

    X = df[dataset_features].to_numpy(dtype=np.float32)
    X_torch = normalize_batch(X)
    total = len(df)

    with torch.no_grad():
        logits = model(X_torch)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confs = probs.max(dim=1).values.cpu().numpy()

    predicted_names = [CLASS_NAMES[int(c)] for c in preds]
    correct = [pred == true for pred, true in zip(predicted_names, true_targets)]

    accuracy = sum(correct) / total if total else 0.0
    print(f"\nDataset rows evaluated: {total}")
    print(f"Overall accuracy: {accuracy:.4f}")

    true_counter = Counter(true_targets)
    pred_counter = Counter(predicted_names)
    print("\nGround truth distribution:")
    for label, count in sorted(true_counter.items()):
        print(f"  {label}: {count}")
    print("\nPredicted distribution:")
    for label, count in sorted(pred_counter.items()):
        print(f"  {label}: {count}")

    matrix = build_confusion_matrix(zip(true_targets, predicted_names))
    print("\nConfusion matrix:")
    summarize_confusion(matrix)

    print("\nSample predictions:")
    sample_count = min(10, total)
    for idx in range(sample_count):
        print(
            f"  row={idx:6d} true={true_targets[idx]} pred={predicted_names[idx]} "
            f"conf={confs[idx]:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate traffic classifier against a CICIDS dataset")
    parser.add_argument("dataset_path", help="Path to a directory containing CICIDS parquet files or a single parquet file")
    parser.add_argument("--model", default="model_fp.pth", help="Path to the trained model file")
    parser.add_argument("--engine", default="pyarrow", choices=["pyarrow", "fastparquet"], help="Parquet engine to use for reading dataset files")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print("Loading dataset...")
    df = load_dataset_files(args.dataset_path, engine=args.engine)

    label_column = find_label_column(df)
    if label_column is None:
        raise ValueError("Could not find a label column in the dataset. "
                         "Expected one of: " + ", ".join(LABEL_COLUMN_CANDIDATES))

    print(f"Detected label column: {label_column}")
    create_validation_report(df, label_column, args.model)


if __name__ == "__main__":
    main()
'''

#!/usr/bin/env python3
"""
validate_model.py  —  Evaluate the MLP classifier on a dataset that was
produced by generate_training_data.py.

IMPORTANT: This script must NOT be used with raw CICIDS parquet files and
DATASET_COLUMN_MAP.  That approach caused ~43% accuracy because the CICIDS
pre-computed columns do not go through the same numerical path as the runtime
extractor.py.  Always validate on CSV files written by generate_training_data.py.

Correct workflow:
    1. python3 generate_training_data.py --dataset /data/cicids/ --output training.csv
    2. python3 train_fp.py --data dataset_final.csv --out model_fp.pth
    3. python3 validate_model.py dataset_final.csv --model model_fp.pth

The validation CSV must have:
    - Columns: <FEATURE_ORDER columns> + "label"
    - Labels:  normal | ddos | botnet   (other labels are dropped)

Usage
-----
  python3 validate_model.py <csv_path> --model model_fp.pth [--batch 2048]
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from constants import FEATURE_ORDER, FEATURE_RANGES, NUM_CLASSES
from model import CLASS_NAMES, load_model, TrafficClassifier

CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in CLASS_NAMES.items()}


# ── Normalization (identical to model.py:normalize_features, batch version) ───

def normalize_batch(X: np.ndarray) -> torch.Tensor:
    mins  = np.array([FEATURE_RANGES[f][0] for f in FEATURE_ORDER], dtype=np.float32)
    maxs  = np.array([FEATURE_RANGES[f][1] for f in FEATURE_ORDER], dtype=np.float32)
    spans = np.maximum(maxs - mins, 1e-9).astype(np.float32)
    X     = np.clip(X, mins, maxs)
    X     = (X - mins) / spans
    return torch.from_numpy(X.astype(np.float32))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load a CSV produced by generate_training_data.py.
    Returns (X: float32 [N, 20], true_labels: list[str]).
    """
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    missing = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing:
        sys.exit(f"CSV is missing feature columns: {missing}")
    if "label" not in df.columns:
        sys.exit(
            "CSV has no 'label' column.\n"
            "This script must be run on output from generate_training_data.py,\n"
            "NOT on raw CICIDS parquet files."
        )

    df = df.dropna(subset=["label"])
    unknown = set(df["label"].unique()) - set(CLASS_NAME_TO_ID.keys())
    if unknown:
        print(f"WARNING: Dropping rows with unknown labels: {unknown}")
        df = df[~df["label"].isin(unknown)]

    X      = df[FEATURE_ORDER].to_numpy(dtype=np.float32)
    labels = df["label"].tolist()
    return X, labels


# ── Evaluation ────────────────────────────────────────────────────────────────

def build_confusion_matrix(
    true_labels: List[str],
    pred_labels: List[str],
) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(true_labels, pred_labels):
        matrix[t][p] += 1
    return {t: dict(preds) for t, preds in matrix.items()}


def print_confusion_matrix(matrix: Dict[str, Dict[str, int]]) -> None:
    all_labels = sorted(set(CLASS_NAMES.values()))
    header = ["true \\ pred"] + all_labels
    print("\t".join(header))
    for t in all_labels:
        row = [t] + [str(matrix.get(t, {}).get(p, 0)) for p in all_labels]
        print("\t".join(row))


def per_class_metrics(true_labels: List[str], pred_labels: List[str]) -> None:
    all_classes = sorted(set(CLASS_NAMES.values()))
    print(f"\n{'Class':<14}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    print("-" * 55)
    for cls in all_classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)
        support   = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  {cls:<12}  {precision:>10.4f}  {recall:>8.4f}  {f1:>8.4f}  {support:>8d}")


def evaluate(csv_path: str, model_path: str, batch_size: int = 2048) -> None:
    X, true_labels = load_csv(csv_path)
    total = len(X)
    print(f"Samples: {total}")

    model: TrafficClassifier = load_model(model_path)
    model.eval()
    print(f"Model loaded from {model_path}\n")

    X_t = normalize_batch(X)

    all_pred_ids: List[int]   = []
    all_confidences: List[float] = []
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch  = X_t[start:start + batch_size]
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_pred_ids.extend(preds.cpu().numpy().tolist())
            all_confidences.extend(confs.cpu().numpy().tolist())

    pred_labels = [CLASS_NAMES[int(i)] for i in all_pred_ids]

    correct  = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / total if total > 0 else 0.0
    avg_conf = float(np.mean(all_confidences))

    print(f"Overall accuracy   : {accuracy:.4f}")
    print(f"Average confidence : {avg_conf:.4f}")

    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)

    print("\nGround truth distribution:")
    for cls in sorted(CLASS_NAMES.values()):
        print(f"  {cls:<12}  {true_counts.get(cls, 0):>8d}")

    print("\nPredicted distribution:")
    for cls in sorted(CLASS_NAMES.values()):
        pct = 100.0 * pred_counts.get(cls, 0) / total
        print(f"  {cls:<12}  {pred_counts.get(cls, 0):>8d}  ({pct:.1f}%)")

    per_class_metrics(true_labels, pred_labels)

    matrix = build_confusion_matrix(true_labels, pred_labels)
    print("\nConfusion matrix:")
    print_confusion_matrix(matrix)

    print("\nSample predictions (first 10):")
    for i in range(min(10, total)):
        ok = "OK" if true_labels[i] == pred_labels[i] else "!!"
        print(
            f"  [{ok}] row={i:6d}  true={true_labels[i]:<12s}"
            f"  pred={pred_labels[i]:<12s}  conf={all_confidences[i]:.3f}"
        )

    dominant_pred = pred_counts.most_common(1)[0]
    dominant_pct  = 100.0 * dominant_pred[1] / total
    if dominant_pct > 70.0:
        print(
            f"\nWARNING: Model is biased toward '{dominant_pred[0]}' "
            f"({dominant_pct:.1f}% of predictions).\n"
            "  Make sure you validated on generate_training_data.py output,\n"
            "  and that the model was trained with class weighting."
        )
    else:
        print(
            f"\nNo severe class bias detected "
            f"(dominant: {dominant_pred[0]} = {dominant_pct:.1f}%)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate traffic classifier on CSV from generate_training_data.py.\n"
            "Do NOT use raw CICIDS parquet files — use generate_training_data.py first."
        )
    )
    parser.add_argument(
        "csv_path",
        help="CSV produced by generate_training_data.py (has FEATURE_ORDER + label column)."
    )
    parser.add_argument(
        "--model", default="model_fp.pth",
        help="Path to .pth model file (default: model_fp.pth)."
    )
    parser.add_argument(
        "--batch", type=int, default=2048,
        help="Inference batch size (default: 2048)."
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        sys.exit(f"CSV not found: {args.csv_path}")
    if not os.path.isfile(args.model):
        sys.exit(f"Model not found: {args.model}")

    evaluate(args.csv_path, args.model, batch_size=args.batch)


if __name__ == "__main__":
    main()