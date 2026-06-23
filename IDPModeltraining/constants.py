"""Static constants for features, class labels, and IP cores."""

FEATURE_ORDER = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_len_fwd_pkts",
    "total_len_bwd_pkts",
    "fwd_pkt_len_max",
    "fwd_pkt_len_min",
    "bwd_pkt_len_max",
    "bwd_pkt_len_min",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "fwd_psh_flags",
    "bwd_psh_flags",
    "syn_flag_count",
    "rst_flag_count",
    "avg_fwd_segment_size",
]

FEATURE_COUNT = len(FEATURE_ORDER)
NUM_CLASSES = 3

# Per-feature (min, max) quantization ranges.
# ── BEGIN AUTO-CALIBRATED FEATURE_RANGES ──
# Recalibrated from dataset_final.csv by build_dataset.py
# (robust 0.1 / 99.9 percentiles). Keys/order match FEATURE_ORDER.
FEATURE_RANGES = {
    "flow_duration":        (2e-06, 120),
    "total_fwd_packets":    (1, 170),
    "total_bwd_packets":    (0, 196),
    "total_len_fwd_pkts":   (0, 22000),
    "total_len_bwd_pkts":   (0, 286000),
    "fwd_pkt_len_max":      (0, 3310),
    "fwd_pkt_len_min":      (0, 101),
    "bwd_pkt_len_max":      (0, 11600),
    "bwd_pkt_len_min":      (0, 345),
    "flow_bytes_per_sec":   (0, 1.8e+07),
    "flow_packets_per_sec": (0.023379, 1e+06),
    "flow_iat_mean":        (2e-06, 85.5),
    "flow_iat_std":         (0, 58.3),
    "fwd_iat_mean":         (0, 101),
    "bwd_iat_mean":         (0, 101),
    "fwd_psh_flags":        (0, 1),
    "bwd_psh_flags":        (0, 1),
    "syn_flag_count":       (0, 1),
    "rst_flag_count":       (0, 1),
    "avg_fwd_segment_size": (0, 861),
}
# ── END AUTO-CALIBRATED FEATURE_RANGES ──

INT8_MIN = -128
INT8_MAX = 127

# AXI4-Lite register map (example Xilinx base design)
AXILITE_CTRL = 0x00
AXILITE_STATUS = 0x04
AXILITE_VEC_LEN = 0x08
AXILITE_NUM_CLASSES = 0x0C
AXILITE_RESULT = 0x10
AXILITE_CONFIDENCE = 0x14

AXILITE_CTRL_START = 0x01
AXILITE_CTRL_RESET = 0x02

AXILITE_STATUS_DONE = 0x01
AXILITE_STATUS_BUSY = 0x02
AXILITE_STATUS_ERR  = 0x04

# Canonical class labels — single source of truth for the whole project.
CLASS_NAMES = {
    0: "normal",
    1: "ddos",
    2: "botnet",
}
