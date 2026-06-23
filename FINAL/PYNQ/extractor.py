"""
features/extractor.py

Computes the 20-feature vector from a completed FlowRecord.

extract_features() is a direct Python port of extract_features() in
features/feature_extractor.c.  All numerical edge-cases (zero duration,
missing reverse traffic, sentinel min values) are handled identically.

The returned dict maps feature name → float value using the canonical
ordering from config/constants.FEATURE_ORDER.
"""

import math
import time
import logging
import threading
import queue as stdlib_queue
from dataclasses import dataclass
from typing import Dict, List, Optional

from flow_engine import FlowRecord, FlowStats
from constants import FEATURE_ORDER, FEATURE_COUNT


import time

# Python 3.6 compatibility
if not hasattr(time, "monotonic_ns"):

    def monotonic_ns():
        return int(time.monotonic() * 1e9)

    time.monotonic_ns = monotonic_ns

if not hasattr(time, "time_ns"):

    def time_ns():
        return int(time.time() * 1e9)

    time.time_ns = time_ns
    
log = logging.getLogger(__name__)

UINT16_SENTINEL = 0xFFFF   # fwd_min_len / bwd_min_len initial value


# ── extracted flow ────────────────────────────────────────────────────────────

@dataclass
class ExtractedFlow:
    """Feature vector ready for quantization and inference."""
    flow_id:              int
    key:                  tuple
    features:             Dict[str, float]   # name → float
    feature_vec:          List[float]        # ordered list (len=20)
    ts_first_realtime_ns: int
    ts_features_ready_ns: int


# ── core extraction function ──────────────────────────────────────────────────

def extract_features(record: FlowRecord) -> Dict[str, float]:
    """
    Compute the 20-feature dictionary from a FlowRecord.

    Exact port of extract_features() in features/feature_extractor.c.
    Feature indices follow FEATURE_ORDER in config/constants.py.
    """
    s: FlowStats = record.stats

    # Duration
    duration_ns = max(0, s.last_time - s.start_time)
    duration_s  = duration_ns / 1e9

    total_packets = s.fwd_packets + s.bwd_packets
    total_bytes   = s.fwd_bytes   + s.bwd_bytes

    # [0] Flow Duration
    flow_duration = duration_s

    # [1] Total Fwd Packets
    total_fwd_packets = float(s.fwd_packets)

    # [2] Total Backward Packets
    total_bwd_packets = float(s.bwd_packets)

    # [3] Total Length of Fwd Packets
    total_len_fwd_pkts = float(s.fwd_bytes)

    # [4] Total Length of Bwd Packets
    total_len_bwd_pkts = float(s.bwd_bytes)

    # [5] Fwd Packet Length Max
    fwd_pkt_len_max = float(s.fwd_max_len)

    # [6] Fwd Packet Length Min  (sentinel → 0 if no packets)
    fwd_pkt_len_min = 0.0 if s.fwd_min_len == UINT16_SENTINEL else float(s.fwd_min_len)

    # [7] Bwd Packet Length Max
    bwd_pkt_len_max = float(s.bwd_max_len)

    # [8] Bwd Packet Length Min
    bwd_pkt_len_min = 0.0 if s.bwd_min_len == UINT16_SENTINEL else float(s.bwd_min_len)

    # [9] Flow Bytes/s
    flow_bytes_per_sec = (total_bytes / duration_s) if duration_s > 0 else 0.0

    # [10] Flow Packets/s
    flow_packets_per_sec = (total_packets / duration_s) if duration_s > 0 else 0.0

    # [11] Flow IAT Mean (seconds)
    if s.flow_iat_count > 0:
        flow_iat_mean = (s.flow_iat_sum / s.flow_iat_count) / 1e9
    else:
        flow_iat_mean = 0.0

    # [12] Flow IAT Std (seconds)  — population std via E[X²] - (E[X])²
    flow_iat_std = 0.0
    if s.flow_iat_count > 1:
        mean_sq  = flow_iat_mean ** 2
        sq_mean  = (s.flow_iat_sq_sum / s.flow_iat_count) / 1e18
        variance = sq_mean - mean_sq
        if variance > 0.0:
            flow_iat_std = math.sqrt(variance)

    # [13] Fwd IAT Mean (seconds)
    if s.fwd_iat_count > 0:
        fwd_iat_mean = (s.fwd_iat_sum / s.fwd_iat_count) / 1e9
    else:
        fwd_iat_mean = 0.0

    # [14] Bwd IAT Mean (seconds)
    if s.bwd_iat_count > 0:
        bwd_iat_mean = (s.bwd_iat_sum / s.bwd_iat_count) / 1e9
    else:
        bwd_iat_mean = 0.0

    # [15] Fwd PSH Flags
    fwd_psh_flags = float(s.fwd_psh)

    # [16] Bwd PSH Flags
    bwd_psh_flags = float(s.bwd_psh)

    # [17] SYN Flag Count
    syn_flag_count = float(s.syn_count)

    # [18] RST Flag Count
    rst_flag_count = float(s.rst_count)

    # [19] Avg Fwd Segment Size
    avg_fwd_segment_size = (s.fwd_bytes / s.fwd_packets) if s.fwd_packets > 0 else 0.0

    return {
        "flow_duration":          flow_duration,
        "total_fwd_packets":      total_fwd_packets,
        "total_bwd_packets":      total_bwd_packets,
        "total_len_fwd_pkts":     total_len_fwd_pkts,
        "total_len_bwd_pkts":     total_len_bwd_pkts,
        "fwd_pkt_len_max":        fwd_pkt_len_max,
        "fwd_pkt_len_min":        fwd_pkt_len_min,
        "bwd_pkt_len_max":        bwd_pkt_len_max,
        "bwd_pkt_len_min":        bwd_pkt_len_min,
        "flow_bytes_per_sec":     flow_bytes_per_sec,
        "flow_packets_per_sec":   flow_packets_per_sec,
        "flow_iat_mean":          flow_iat_mean,
        "flow_iat_std":           flow_iat_std,
        "fwd_iat_mean":           fwd_iat_mean,
        "bwd_iat_mean":           bwd_iat_mean,
        "fwd_psh_flags":          fwd_psh_flags,
        "bwd_psh_flags":          bwd_psh_flags,
        "syn_flag_count":         syn_flag_count,
        "rst_flag_count":         rst_flag_count,
        "avg_fwd_segment_size":   avg_fwd_segment_size,
    }


def features_to_vector(feat_dict: Dict[str, float]) -> List[float]:
    """
    Convert feature dictionary to an ordered list following FEATURE_ORDER.
    This is the canonical order expected by the quantizer and FPGA model.
    """
    return [feat_dict[name] for name in FEATURE_ORDER]


# ── feature extractor thread ──────────────────────────────────────────────────

class FeatureExtractor:
    """
    Reads FlowRecord from in_queue, computes the 20-feature vector, and
    pushes ExtractedFlow onto out_queue.
    """

    def __init__(self, in_queue: stdlib_queue.Queue,
                 out_queue: stdlib_queue.Queue):
        self.in_queue   = in_queue
        self.out_queue  = out_queue
        self._stop_flag = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True,
                                           name="feature_extractor")
        self.flows_extracted = 0

    def start(self) -> None:
        self._thread.start()
        log.info("[features] Extractor started.")

    def stop(self) -> None:
        self._stop_flag.set()
        try:
            self.in_queue.put_nowait(None)
        except stdlib_queue.Full:
            pass
        self._thread.join(timeout=5.0)
        log.info("[features] Stopped. extracted=%d", self.flows_extracted)

    def _run(self) -> None:
        while not self._stop_flag.is_set():
            try:
                item = self.in_queue.get(timeout=1.0)
            except stdlib_queue.Empty:
                continue

            if item is None:
                break

            record: FlowRecord = item
            feat_dict = extract_features(record)
            feat_vec  = features_to_vector(feat_dict)
            ts_ready  = time.monotonic_ns()

            ef = ExtractedFlow(
                flow_id              = record.flow_id,
                key                  = record.key,
                features             = feat_dict,
                feature_vec          = feat_vec,
                ts_first_realtime_ns = record.ts_first_realtime_ns,
                ts_features_ready_ns = ts_ready,
            )

            try:
                self.out_queue.put_nowait(ef)
                self.flows_extracted += 1
            except stdlib_queue.Full:
                log.warning("[features] Output queue full – dropped flow %d",
                            record.flow_id)