#!/usr/bin/env python3
"""
generate_training_data.py  —  Build a 3-class feature dataset from the
CSE-CIC-IDS2018 parquet files using the SAME runtime feature pipeline.

For every selected row we reconstruct a FlowStats from the CICFlowMeter columns
and run it through the identical extract_features() used by the live receiver
(extractor.py).  This guarantees training and inference see the same feature
distribution.

Classes (see constants.CLASS_NAMES):
    normal  ← Benign
    ddos    ← the DoS / DDoS attack family (Hulk, GoldenEye, LOIC, HOIC, ...)
    botnet  ← Bot

Everything else (Infiltration, BruteForce/Patator, PortScan, Web Attacks,
SQL Injection, Heartbleed, ...) is dropped.

Because the raw dataset is ~8M rows and per-row reconstruction is expensive,
each class is *coarse-capped* while streaming the files (random per-file
sampling).  The exact final balancing happens later in build_dataset.py.

Usage
-----
  python3 generate_training_data.py \
      --dataset dataset/ \
      --output  training_features.csv \
      --per_class_cap 300000
"""

import argparse
import os
import sys
import time
import logging
from collections import Counter
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from constants import FEATURE_ORDER, FEATURE_COUNT, CLASS_NAMES
from extractor import extract_features
from flow_engine import FlowRecord, FlowStats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("generate_training_data")

TARGET_CLASSES = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]  # [normal, ddos, botnet]


# ── visual helpers ────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}", flush=True)


# ── label normalisation → {normal, ddos, botnet} or None (drop) ───────────────

def normalize_label(raw) -> Optional[str]:
    """Map a raw CIC-IDS label to one of {normal, ddos, botnet} or None."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    v = str(raw).strip().lower()
    if not v:
        return None
    # normal / benign
    if "benign" in v or v == "normal":
        return "normal"
    # botnet
    if v == "bot" or "botnet" in v or v.startswith("bot"):
        return "botnet"
    # DoS / DDoS family  (covers "dos hulk", "ddos attacks-loic-http",
    # "ddos attack-hoic", "dos attacks-goldeneye", "heartbleed", ...)
    if "ddos" in v or "dos " in v or v.startswith("dos") or "heartbleed" in v:
        return "ddos"
    # everything else (infiltration, brute/patator, portscan, web attack, sql) → drop
    return None


LABEL_COLUMN_CANDIDATES = ["Label", "label", "CLASS", "Class", "attack_type",
                           "Attack", "Category"]


def find_label_col(columns: List[str]) -> Optional[str]:
    for name in LABEL_COLUMN_CANDIDATES:
        if name in columns:
            return name
    return None


# ── CICFlowMeter columns we read to reconstruct FlowStats ─────────────────────

CICIDS_COL = {
    "duration":      "Flow Duration",            # µs
    "fwd_pkts":      "Total Fwd Packets",
    "fwd_bytes":     "Fwd Packets Length Total",
    "fwd_max_len":   "Fwd Packet Length Max",
    "fwd_min_len":   "Fwd Packet Length Min",
    "fwd_psh":       "Fwd PSH Flags",
    "fwd_iat_mean":  "Fwd IAT Mean",             # µs
    "bwd_pkts":      "Total Backward Packets",
    "bwd_bytes":     "Bwd Packets Length Total",
    "bwd_max_len":   "Bwd Packet Length Max",
    "bwd_min_len":   "Bwd Packet Length Min",
    "bwd_psh":       "Bwd PSH Flags",
    "bwd_iat_mean":  "Bwd IAT Mean",             # µs
    "flow_iat_mean": "Flow IAT Mean",            # µs
    "flow_iat_std":  "Flow IAT Std",             # µs
    "syn_count":     "SYN Flag Count",
    "rst_count":     "RST Flag Count",
}

CICIDS_COL_ALT = {
    "fwd_bytes":     ["Fwd Packets Length Total", "Total Length of Fwd Packets",
                      "Fwd Packet Length Total"],
    "bwd_bytes":     ["Bwd Packets Length Total", "Total Length of Bwd Packets",
                      "Bwd Packet Length Total"],
    "fwd_iat_mean":  ["Fwd IAT Mean", "Fwd IAT Total"],
    "bwd_iat_mean":  ["Bwd IAT Mean", "Bwd IAT Total"],
    "flow_iat_mean": ["Flow IAT Mean"],
    "flow_iat_std":  ["Flow IAT Std"],
}


def _resolve_col(columns, candidates) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def build_col_map(columns: List[str]) -> Dict[str, str]:
    col_map: Dict[str, str] = {}
    missing: List[str] = []
    cols = set(columns)
    for logical, primary in CICIDS_COL.items():
        alternates = CICIDS_COL_ALT.get(logical, [primary])
        if primary not in alternates:
            alternates = [primary] + list(alternates)
        resolved = _resolve_col(cols, alternates)
        if resolved is None:
            missing.append(f"  {logical!r}: tried {alternates}")
        else:
            col_map[logical] = resolved
    if missing:
        log.warning("Missing CICIDS columns (features default to 0.0):\n%s",
                    "\n".join(missing))
    return col_map


# ── FlowStats reconstruction (µs → ns, sentinels, std back-compute) ───────────

def _get(row: dict, col: str, default=0.0) -> float:
    v = row.get(col, default)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def reconstruct_flow_stats(row: dict, col_map: Dict[str, str]) -> FlowStats:
    s = FlowStats()

    dur_us = max(0.0, _get(row, col_map.get("duration", "")))
    s.start_time = 0
    s.last_time  = int(dur_us * 1_000)          # µs → ns

    # forward
    s.fwd_packets = int(_get(row, col_map.get("fwd_pkts", "")))
    s.fwd_bytes   = int(_get(row, col_map.get("fwd_bytes", "")))
    s.fwd_max_len = int(_get(row, col_map.get("fwd_max_len", "")))
    if s.fwd_packets == 0:
        s.fwd_min_len = 0xFFFF
    else:
        s.fwd_min_len = max(0, int(_get(row, col_map.get("fwd_min_len", ""))))
    s.fwd_psh = int(_get(row, col_map.get("fwd_psh", "")))
    fwd_cnt = max(0, s.fwd_packets - 1)
    s.fwd_iat_count = fwd_cnt
    s.fwd_iat_sum   = int(_get(row, col_map.get("fwd_iat_mean", "")) * 1_000 * fwd_cnt)

    # backward
    s.bwd_packets = int(_get(row, col_map.get("bwd_pkts", "")))
    s.bwd_bytes   = int(_get(row, col_map.get("bwd_bytes", "")))
    s.bwd_max_len = int(_get(row, col_map.get("bwd_max_len", "")))
    if s.bwd_packets == 0:
        s.bwd_min_len = 0xFFFF
    else:
        s.bwd_min_len = max(0, int(_get(row, col_map.get("bwd_min_len", ""))))
    s.bwd_psh = int(_get(row, col_map.get("bwd_psh", "")))
    bwd_cnt = max(0, s.bwd_packets - 1)
    s.bwd_iat_count = bwd_cnt
    s.bwd_iat_sum   = int(_get(row, col_map.get("bwd_iat_mean", "")) * 1_000 * bwd_cnt)

    # flow IAT (all packets)
    total = s.fwd_packets + s.bwd_packets
    flow_cnt = max(0, total - 1)
    s.flow_iat_count = flow_cnt
    mean_ns = _get(row, col_map.get("flow_iat_mean", "")) * 1_000
    std_ns  = _get(row, col_map.get("flow_iat_std", "")) * 1_000
    s.flow_iat_sum = int(mean_ns * flow_cnt)
    if flow_cnt > 1:
        ex2 = std_ns ** 2 + mean_ns ** 2          # E[X²] = Var + mean²
        s.flow_iat_sq_sum = int(ex2 * flow_cnt)
    else:
        s.flow_iat_sq_sum = 0

    # TCP flags
    s.syn_count = int(_get(row, col_map.get("syn_count", "")))
    s.rst_count = int(_get(row, col_map.get("rst_count", "")))
    return s


# ── per-file streaming collection with per-class budget ───────────────────────

def collect_capped_rows(files: List[str], per_class_cap: int,
                        seed: int) -> Dict[str, pd.DataFrame]:
    """Stream the parquet files and randomly collect up to `per_class_cap`
    rows per class (botnet is naturally smaller than the cap → all kept)."""
    budget   = {c: per_class_cap for c in TARGET_CLASSES}
    pools     = {c: [] for c in TARGET_CLASSES}
    collected = {c: 0 for c in TARGET_CLASSES}
    raw_totals = Counter()

    col_map = None
    needed: List[str] = []
    label_col = None

    for fi, f in enumerate(files):
        if col_map is None:
            cols = pq.read_schema(f).names
            col_map   = build_col_map(cols)
            label_col = find_label_col(cols)
            if label_col is None:
                sys.exit(f"No label column found in {f}")
            needed = sorted(set(col_map.values()) | {label_col})
            log.info("Resolved %d feature columns; label column = %r",
                     len(col_map), label_col)

        df = pd.read_parquet(f, columns=needed)
        df["__target"] = df[label_col].map(normalize_label)
        raw_totals.update(df["__target"].dropna().value_counts().to_dict())
        df = df[df["__target"].notna()]

        taken_here = {}
        for cls in TARGET_CLASSES:
            remaining = budget[cls] - collected[cls]
            if remaining <= 0:
                continue
            sub = df[df["__target"] == cls]
            if len(sub) > remaining:
                sub = sub.sample(remaining, random_state=seed)
            if len(sub):
                pools[cls].append(sub.drop(columns="__target"))
                collected[cls] += len(sub)
                taken_here[cls] = len(sub)

        log.info("[%2d/%d] %-55s took %s",
                 fi + 1, len(files), os.path.basename(f)[:55],
                 {k: v for k, v in taken_here.items()} or "—")

    banner("RAW TARGET-CLASS AVAILABILITY (before capping)")
    for cls in TARGET_CLASSES:
        print(f"  {cls:<10} available≈{raw_totals.get(cls, 0):>10,}   "
              f"collected={collected[cls]:>10,}  (cap={per_class_cap:,})")

    return {c: (pd.concat(pools[c], ignore_index=True) if pools[c]
               else pd.DataFrame(columns=needed))
            for c in TARGET_CLASSES}, col_map


# ── feature extraction over collected rows (reuses extract_features) ──────────

def extract_class(df: pd.DataFrame, label: str,
                  col_map: Dict[str, str]) -> List[dict]:
    rows_out: List[dict] = []
    records = df.to_dict("records")
    t0 = time.time()
    for i, row in enumerate(records):
        stats = reconstruct_flow_stats(row, col_map)
        rec = FlowRecord(flow_id=i, key=("0.0.0.0", "0.0.0.0", 0, 0, 6),
                         stats=stats, ts_first_realtime_ns=0, ts_completed_ns=0)
        feat = extract_features(rec)               # ← SAME function as runtime
        out = {name: feat[name] for name in FEATURE_ORDER}
        out["label"] = label
        rows_out.append(out)
        if (i + 1) % 50_000 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            log.info("    %s: %d/%d (%.0f rows/s)", label, i + 1, len(records), rate)
    return rows_out


def print_stats(df: pd.DataFrame) -> None:
    banner("FEATURE DATASET — LABEL DISTRIBUTION")
    counts = df["label"].value_counts()
    for label, count in counts.items():
        print(f"  {label:<10} {count:>10,}  ({100.0 * count / len(df):5.1f}%)")
    print(f"  {'TOTAL':<10} {len(df):>10,}")
    print("\nFeature stats (first 5):")
    print(df[FEATURE_ORDER[:5]].describe().to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="dataset/",
                   help="Directory of CSE-CIC-IDS2018 parquet files (default dataset/).")
    p.add_argument("--output", default="training_features.csv",
                   help="Output features CSV (default training_features.csv).")
    p.add_argument("--per_class_cap", type=int, default=300_000,
                   help="Max rows collected per class before extraction (default 300k).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    files = ([args.dataset] if os.path.isfile(args.dataset)
             else sorted(glob(os.path.join(args.dataset, "*.parquet"))))
    if not files:
        sys.exit(f"No parquet files found in: {args.dataset}")

    banner("STEP 1 — STREAMING & CAPPING RAW ROWS")
    log.info("Found %d parquet files; per-class cap = %d", len(files), args.per_class_cap)
    pools, col_map = collect_capped_rows(files, args.per_class_cap, args.seed)

    banner("STEP 2 — FEATURE EXTRACTION (same pipeline as receiver.py)")
    all_rows: List[dict] = []
    for cls in TARGET_CLASSES:
        log.info("Extracting %d %r rows ...", len(pools[cls]), cls)
        all_rows.extend(extract_class(pools[cls], cls, col_map))

    df_out = pd.DataFrame(all_rows, columns=FEATURE_ORDER + ["label"])
    df_out = df_out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print_stats(df_out)

    df_out.to_csv(args.output, index=False)
    banner(f"SAVED {len(df_out):,} ROWS → {args.output}")


if __name__ == "__main__":
    main()
