"""
flow/flow_engine.py

Bidirectional flow construction engine.

Consumes ParsedPacket objects from the capture queue and maintains a flow
table keyed by the canonical 5-tuple.  When a flow completes (TCP FIN/RST
or timeout), a FlowRecord is placed on the output queue for the feature
extractor.

Design decisions ported from flow_table.c:
  - Canonical key: the endpoint with the numerically smaller (IP, port) is
    always placed as src so both directions hash to the same entry.
  - TCP FIN/RST triggers immediate completion.
  - A periodic timeout sweep runs inside the processing loop.
  - The flow table is a plain dict (CPython dict is thread-safe for single-
    writer use; we run one flow engine thread).
"""

import time
import logging
import threading
import queue as stdlib_queue
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from capture import ParsedPacket, TCP_FLAG_FIN, TCP_FLAG_RST, TCP_FLAG_PSH, TCP_FLAG_SYN


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

# 5-tuple canonical flow key
FlowKey = Tuple[str, str, int, int, int]   # (src_ip, dst_ip, sport, dport, proto)


# ── per-flow statistics accumulator ──────────────────────────────────────────

@dataclass
class FlowStats:
    """
    Incrementally-updated statistics for one bidirectional flow.
    Field names mirror flow_stats_t in flow_table.h for consistency.
    All time values are nanoseconds.
    """
    # Timing
    start_time:       int = 0    # ts_ns of first packet
    last_time:        int = 0    # ts_ns of most recent packet

    # Forward direction
    fwd_packets:   int = 0
    fwd_bytes:     int = 0
    fwd_max_len:   int = 0
    fwd_min_len:   int = 0xFFFF  # sentinel
    fwd_psh:       int = 0
    fwd_iat_sum:   int = 0       # nanoseconds
    fwd_iat_count: int = 0
    last_fwd_time: int = 0

    # Backward direction
    bwd_packets:   int = 0
    bwd_bytes:     int = 0
    bwd_max_len:   int = 0
    bwd_min_len:   int = 0xFFFF
    bwd_psh:       int = 0
    bwd_iat_sum:   int = 0
    bwd_iat_count: int = 0
    last_bwd_time: int = 0

    # Flow-level IAT (all packets)
    flow_iat_sum:    int = 0
    flow_iat_sq_sum: int = 0     # ns² for variance
    flow_iat_count:  int = 0

    # TCP flags
    syn_count: int = 0
    rst_count: int = 0


@dataclass
class FlowRecord:
    """
    Completed flow handed to the feature extractor.
    Contains the 5-tuple key, accumulated statistics, and timing metadata.
    """
    flow_id:       int
    key:           FlowKey
    stats:         FlowStats
    ts_first_realtime_ns: int   # CLOCK_REALTIME of first packet
    ts_completed_ns:      int   # CLOCK_MONOTONIC when flow was evicted


# ── canonical key construction ────────────────────────────────────────────────

def _make_canonical_key(pkt: ParsedPacket) -> Tuple[FlowKey, bool]:
    """
    Return (canonical_key, is_forward).

    The canonical direction places the numerically smaller (IP, port) as src.
    This ensures packets from both directions hash to the same flow entry.
    Mirrors normalise_key() in flow_table.c exactly.
    """
    a = (pkt.src_ip, pkt.src_port)
    b = (pkt.dst_ip, pkt.dst_port)
    forward = a < b  # tuple comparison: IP string first, then port

    if forward:
        key: FlowKey = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol)
    else:
        key = (pkt.dst_ip, pkt.src_ip, pkt.dst_port, pkt.src_port, pkt.protocol)
    return key, forward


# ── flow stats update ─────────────────────────────────────────────────────────

def _update_stats(stats: FlowStats, pkt: ParsedPacket, is_forward: bool) -> None:
    """
    Incrementally update flow statistics for one packet.
    Exact Python port of update_stats() in flow_table.c.
    """
    ts  = pkt.ts_ns
    plen = pkt.payload_len

    # Flow-level IAT
    if stats.last_time > 0 and ts > stats.last_time:
        iat = ts - stats.last_time
        stats.flow_iat_sum    += iat
        stats.flow_iat_sq_sum += iat * iat
        stats.flow_iat_count  += 1
    stats.last_time = ts

    # TCP flags
    if pkt.tcp_flags & TCP_FLAG_SYN:
        stats.syn_count += 1
    if pkt.tcp_flags & TCP_FLAG_RST:
        stats.rst_count += 1

    if is_forward:
        if stats.fwd_packets > 0 and ts > stats.last_fwd_time:
            stats.fwd_iat_sum   += ts - stats.last_fwd_time
            stats.fwd_iat_count += 1
        stats.last_fwd_time = ts
        stats.fwd_packets += 1
        stats.fwd_bytes   += plen
        if plen > stats.fwd_max_len: stats.fwd_max_len = plen
        if plen < stats.fwd_min_len: stats.fwd_min_len = plen
        if pkt.tcp_flags & TCP_FLAG_PSH:
            stats.fwd_psh += 1
    else:
        if stats.bwd_packets > 0 and ts > stats.last_bwd_time:
            stats.bwd_iat_sum   += ts - stats.last_bwd_time
            stats.bwd_iat_count += 1
        stats.last_bwd_time = ts
        stats.bwd_packets += 1
        stats.bwd_bytes   += plen
        if plen > stats.bwd_max_len: stats.bwd_max_len = plen
        if plen < stats.bwd_min_len: stats.bwd_min_len = plen
        if pkt.tcp_flags & TCP_FLAG_PSH:
            stats.bwd_psh += 1


# ── flow engine ───────────────────────────────────────────────────────────────

class FlowEngine:
    """
    Reads ParsedPacket from in_queue, maintains the flow table, and puts
    completed FlowRecord objects onto out_queue.

    Call start() to launch the background thread and stop() to shut down.
    """

    def __init__(self,
                 in_queue:     stdlib_queue.Queue,
                 out_queue:    stdlib_queue.Queue,
                 timeout_s:    float = 30.0,
                 max_flows:    int   = 65536,
                 debug:        bool  = False):
        self.in_queue   = in_queue
        self.out_queue  = out_queue
        self.timeout_ns = int(timeout_s * 1e9)
        self.max_flows  = max_flows
        self.debug      = debug

        self._table:     Dict[FlowKey, FlowStats]  = {}
        self._meta:      Dict[FlowKey, Tuple[int, int, int]] = {}
        # meta[key] = (flow_id, ts_first_realtime_ns, ts_first_monotonic_ns)
        self._flow_counter = 0

        self._stop_flag = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True,
                                           name="flow_engine")

        # Statistics
        self.packets_processed = 0
        self.flows_completed   = 0
        self._last_expire_ns = time.time_ns()


    def start(self) -> None:
        self._thread.start()
        log.info("[flow] Engine started. timeout=%.1fs max_flows=%d",
                 self.timeout_ns / 1e9, self.max_flows)

    def stop(self) -> None:
        self._stop_flag.set()
        # Unblock the thread if it's waiting on an empty queue
        try:
            self.in_queue.put_nowait(None)
        except stdlib_queue.Full:
            pass
        self._thread.join(timeout=5.0)
        log.info("[flow] Stopped. packets=%d flows_completed=%d",
                 self.packets_processed, self.flows_completed)

    # ── internal ──────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._flow_counter += 1
        return self._flow_counter

    def _complete_flow(self, key: FlowKey, reason: str) -> None:
        stats = self._table.pop(key, None)
        meta  = self._meta.pop(key, None)
        if stats is None or meta is None:
            return
        flow_id, ts_first_rt, ts_first_mono = meta
        rec = FlowRecord(
            flow_id               = flow_id,
            key                   = key,
            stats                 = stats,
            ts_first_realtime_ns  = ts_first_rt,
            ts_completed_ns       = time.monotonic_ns(),
        )
        self.flows_completed += 1
        if self.debug:
            print(
                "[FLOW COMPLETED]",
                rec.key[0],
                "->",
                rec.key[1],
                rec.key[2],
                rec.key[3],
                "pkts:",
                rec.stats.fwd_packets + rec.stats.bwd_packets
            )
        try:
            self.out_queue.put_nowait(rec)
        except stdlib_queue.Full:
            log.warning("[flow] Output queue full – dropped flow %d (%s)", flow_id, reason)

    def _expire_idle_flows(self) -> None:

        now = time.time_ns()

        if now - self._last_expire_ns < 1_000_000_000:
            return

        self._last_expire_ns = now

        cutoff = now - self.timeout_ns

        expired_keys = [
            k for k, s in self._table.items()
            if s.last_time < cutoff
        ]

        for key in expired_keys:
            self._complete_flow(key, "timeout")

        if expired_keys:
            log.debug(
                "[flow] Expired %d idle flows",
                len(expired_keys)
            )
    def _process_packet(self, pkt: ParsedPacket) -> None:
        key, is_forward = _make_canonical_key(pkt)

        if key not in self._table:
            # New flow
            if len(self._table) >= self.max_flows:
                log.warning("[flow] Table full – dropping packet")
                return
            stats = FlowStats(start_time=pkt.ts_ns)
            self._table[key] = stats
            self._meta[key]  = (self._next_id(), pkt.ts_ns, time.monotonic_ns())
        else:
            stats = self._table[key]

        _update_stats(stats, pkt, is_forward)

        # TCP FIN or RST → immediate flow completion
        tcp_done = (pkt.protocol == 6) and bool(pkt.tcp_flags & 0x05)  # FIN | RST
        if self.debug and pkt.protocol == 6:
            print(
                "[TCP FLAGS]",
                pkt.tcp_flags,
                hex(pkt.tcp_flags)
            )
        if tcp_done:
            self._complete_flow(key, "tcp_fin_rst")

        self.packets_processed += 1

    def _run(self) -> None:
        while not self._stop_flag.is_set():
            try:
                item = self.in_queue.get(timeout=1.0)
            except stdlib_queue.Empty:
                self._expire_idle_flows()
                continue

            if item is None:
                break   # Sentinel from stop()

            self._process_packet(item)
            self._expire_idle_flows()

        # Final drain: expire all remaining flows on shutdown
        for key in list(self._table.keys()):
            self._complete_flow(key, "shutdown")