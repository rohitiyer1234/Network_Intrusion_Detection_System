#!/usr/bin/env python3
"""
receiver.py  —  PYNQ-side FPGA-NIDS real-time traffic classifier.

Receives synthetic 60-byte UDP frames from the laptop traffic generator,
reconstructs bidirectional flows, extracts 20 CICIDS-style features, runs
the MLP classifier, and streams JSON inference results back to the laptop
via a persistent TCP connection on port 9000.

Pipeline (all threaded, queue-connected):

    UDP socket (port 5555)
          ↓  ParsedPacket  [pkt_queue, depth 16384]
    FlowEngine         — bidirectional flow table, TCP FIN/RST + timeout eviction
          ↓  FlowRecord   [flow_queue, depth 16384]
    FeatureExtractor   — 20 CICIDS-style features
          ↓  ExtractedFlow [feat_queue, depth 16384]
    InferenceEngine    — MLP float inference (CPU, no quantization)
          ↓  JSON result   [result_queue, depth 4096]
    ResultSender       — TCP connection to laptop:9000, newline-delimited JSON

Usage:
    python3 receiver.py --model balanced_dataset_mlp_model.pth \
                        --udp_port 5555 \
                        --laptop_ip 192.168.2.1 \
                        --laptop_port 9000

Requirements (PYNQ):
    pip install numpy
"""

import argparse
import collections
import json
import logging
#import os
import queue
import signal
import socket
import struct
import threading
import time
from typing import Dict, Optional

#import torch

# ── local imports ─────────────────────────────────────────────────────────────
from capture     import ParsedPacket, TCP_FLAG_FIN, TCP_FLAG_RST
from flow_engine import FlowEngine #FlowRecord
from extractor   import FeatureExtractor, ExtractedFlow #features_to_vector
#from PYNQ.model       import load_model, normalize_features, CLASS_NAMES
from numpy_model import NumpyMLP, CLASS_NAMES

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



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("receiver")

# ── wire format ───────────────────────────────────────────────────────────────
# Exactly matches advanced_traffic_generator.py and traffic_generator.py.
# "!B4s4sHHHBHQ" = proto(1) src_ip(4) dst_ip(4) sport(2) dport(2)
#                  plen(2)  flags(1)  ip_total(2) ts_ns(8) = 26 bytes
# Padded to 60 bytes total.

FRAME_STRUCT = struct.Struct("!B4s4sHHHBHQ")
FRAME_SIZE   = 60   # bytes


# ── frame parser ─────────────────────────────────────────────────────────────

def _unpack_frame(data: bytes) -> Optional[ParsedPacket]:
    """
    Decode one 60-byte synthetic frame into a ParsedPacket.

    Returns None on any parse error so the caller can count bad frames
    without crashing.
    """
    if len(data) < FRAME_STRUCT.size:
        return None
    try:
        (proto, src_raw, dst_raw,
         sport, dport, plen, flags, ip_total, ts_ns) = FRAME_STRUCT.unpack_from(data)
    except struct.error:
        return None

    if proto not in (6, 17):   # TCP or UDP only
        return None

    #print(
    #"[PARSED]",
    #socket.inet_ntoa(src_raw),
    #socket.inet_ntoa(dst_raw),
    #sport,
    #dport,
    #proto,
    #plen
    #)
    
    return ParsedPacket(
        ts_ns        = ts_ns,
        src_ip       = socket.inet_ntoa(src_raw),
        dst_ip       = socket.inet_ntoa(dst_raw),
        protocol     = proto,
        src_port     = sport,
        dst_port     = dport,
        payload_len  = plen,
        tcp_flags    = flags,
        ip_total_len = ip_total,
    )


# ── UDP receiver thread ───────────────────────────────────────────────────────

class UDPReceiver:
    """
    Listens on a UDP socket and pushes ParsedPacket objects onto out_queue.

    Buffer is sized to 4 MB to absorb burst traffic from the laptop generator.
    Frames that fail parsing are silently counted in err_count.
    """

    def __init__(self, port: int, out_queue: queue.Queue):
        self.port      = port
        self.out_queue = out_queue
        self._stop     = threading.Event()
        self._thread   = threading.Thread(
            target=self._run, daemon=True, name="udp_rx"
        )
        # Counters (no lock needed — only written by _run thread)
        self.rx_count  = 0
        self.err_count = 0
        self.drop_count = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        log.info("[udp_rx] Stopped. rx=%d err=%d drop=%d",
                 self.rx_count, self.err_count, self.drop_count)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        sock.settimeout(1.0)
        sock.bind(("0.0.0.0", self.port))
        log.info("[udp_rx] Bound to 0.0.0.0:%d (rcvbuf=4MB)", self.port)

        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(FRAME_SIZE + 16)
            except socket.timeout:
                continue
            except OSError as exc:
                log.warning("[udp_rx] recv error: %s", exc)
                self.err_count += 1
                continue

            pkt = _unpack_frame(data)
            if pkt is None:
                self.err_count += 1
                continue

            self.rx_count += 1
            try:
                self.out_queue.put_nowait(pkt)
            except queue.Full:
                self.drop_count += 1

        sock.close()


# ── inference engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Reads ExtractedFlow objects, normalizes features, runs MLP inference,
    and places JSON-ready result dicts onto result_queue.

    Uses the float (non-quantized) model loaded via model.load_model().
    Inference runs on CPU — appropriate for PYNQ-Z2 Cortex-A9.
    """

    def __init__(
        self,
        in_queue:     queue.Queue,
        result_queue: queue.Queue,
        model:        NumpyMLP,
        stats:        Dict[str, int],
        stats_lock:   threading.Lock,
        debug:        bool = False,
    ):
        self.in_queue     = in_queue
        self.result_queue = result_queue
        self.model        = model
        self.stats        = stats
        self.stats_lock   = stats_lock
        self.debug        = debug
        self._stop        = threading.Event()
        self._thread      = threading.Thread(
            target=self._run, daemon=True, name="inference"
        )
        self.count = 0
        # Inference-time metric (milliseconds). Single-writer (this thread).
        self._lat_sum_ms   = 0.0      # cumulative, for lifetime mean
        self._lat_max_ms   = 0.0
        self._lat_window   = collections.deque(maxlen=500)  # recent latencies

    @property
    def mean_latency_ms(self) -> float:
        return (self._lat_sum_ms / self.count) if self.count else 0.0

    @property
    def recent_latency_ms(self) -> float:
        w = self._lat_window
        return (sum(w) / len(w)) if w else 0.0

    def start(self) -> None:
        self._thread.start()
        log.info("[inference] Engine started.")

    def stop(self) -> None:
        self._stop.set()
        try:
            self.in_queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        log.info("[inference] Stopped. classified=%d  mean_infer=%.3f ms  max=%.3f ms",
                 self.count, self.mean_latency_ms, self._lat_max_ms)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.in_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            ef: ExtractedFlow = item

            # Normalize the 20-feature float vector to [0,1] range
            try:
                t0 = time.perf_counter()

                prediction = self.model.predict(
                    ef.feature_vec
                )

                latency_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                # Update inference-time metric (ms)
                self._lat_sum_ms += latency_ms
                self._lat_window.append(latency_ms)
                if latency_ms > self._lat_max_ms:
                    self._lat_max_ms = latency_ms

            except Exception as exc:

                log.error(
                    "[inference] Model error: %s",
                    exc
                    )

                continue

            cls_id = prediction["class_id"]

            conf = prediction["confidence"]

            cls_name = prediction["class_name"]
            '''try:
                tensor = normalize_features(ef.feature_vec)
                classes, confs = self.model.predict(tensor)
            except Exception as exc:
                log.error("[inference] Model error: %s", exc)
                continue

            cls_id   = int(classes[0])
            conf     = float(confs[0])
            cls_name = CLASS_NAMES.get(cls_id, "other")'''
            protocol = "tcp" if ef.key[4] == 6 else "udp"

            # Update shared stats
            with self.stats_lock:
                self.stats[cls_name] = self.stats.get(cls_name, 0) + 1

            self.count += 1

            # Build dashboard-ready JSON result
            result = {
                "flow_id":        ef.flow_id,
                "timestamp_ns":   ef.ts_first_realtime_ns,
                "src_ip":         ef.key[0],
                "dst_ip":         ef.key[1],
                "src_port":       ef.key[2],
                "dst_port":       ef.key[3],
                "protocol":       protocol,
                "classification": cls_name,
                "class_id":       cls_id,
                "confidence":     round(conf, 4),
                "inference_ms":   round(latency_ms, 4),
                "features": {
                    name: round(val, 6)
                    for name, val in zip(
                        _FEATURE_ORDER, ef.feature_vec
                    )
                },
            }

            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                pass  # Drop result rather than stall inference

            if self.debug:
                # Concise, always-on per-flow line (demonstrability without spam)
                icon = {"normal": "✓", "ddos": "⚠", "botnet": "☣"}.get(cls_name, "•")
                print(
                    f"  {icon} flow {ef.flow_id:<6d} "
                    f"{ef.key[0]}:{ef.key[2]:<5} → {ef.key[1]}:{ef.key[3]:<5} "
                    f"{protocol:<3} | {cls_name:<7} {conf*100:5.1f}% | "
                    f"infer {latency_ms:.3f} ms"
                )

            if self.count % 100 == 0:
                with self.stats_lock:
                    totals = dict(self.stats)
                log.info("[inference] flows=%d  mean_infer=%.3f ms (recent %.3f ms, max %.3f ms)  stats=%s",
                         self.count, self.mean_latency_ms, self.recent_latency_ms,
                         self._lat_max_ms, totals)


# CICIDS feature order — must match extractor.py / constants.py
_FEATURE_ORDER = [
    "flow_duration", "total_fwd_packets", "total_bwd_packets",
    "total_len_fwd_pkts", "total_len_bwd_pkts",
    "fwd_pkt_len_max", "fwd_pkt_len_min",
    "bwd_pkt_len_max", "bwd_pkt_len_min",
    "flow_bytes_per_sec", "flow_packets_per_sec",
    "flow_iat_mean", "flow_iat_std",
    "fwd_iat_mean", "bwd_iat_mean",
    "fwd_psh_flags", "bwd_psh_flags",
    "syn_flag_count", "rst_flag_count",
    "avg_fwd_segment_size",
]


# ── TCP result sender ─────────────────────────────────────────────────────────

class ResultSender:
    """
    Reads JSON result dicts from result_queue and sends them as newline-
    delimited JSON over a persistent TCP connection to the laptop receiver.

    Automatically reconnects if the connection drops.  This makes the sender
    resilient to laptop-side restarts.

    JSON format: one compact JSON object per line, terminated with '\\n'.
    This matches the schema expected by classification_receiver.py and
    is ready for consumption by a React dashboard WebSocket bridge.
    """

    _RECONNECT_DELAY_S = 2.0
    _SEND_TIMEOUT_S    = 5.0

    def __init__(
        self,
        result_queue: queue.Queue,
        laptop_ip:    str,
        laptop_port:  int,
    ):
        self.result_queue = result_queue
        self.laptop_ip    = laptop_ip
        self.laptop_port  = laptop_port
        self._stop        = threading.Event()
        self._thread      = threading.Thread(
            target=self._run, daemon=True, name="result_sender"
        )
        self.sent_count   = 0
        self.drop_count   = 0

    def start(self) -> None:
        self._thread.start()
        log.info("[sender] Started. Target: %s:%d", self.laptop_ip, self.laptop_port)

    def stop(self) -> None:
        self._stop.set()
        try:
            self.result_queue.put_nowait(None)   # Unblock get()
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        log.info("[sender] Stopped. sent=%d dropped=%d", self.sent_count, self.drop_count)

    def _connect(self) -> Optional[socket.socket]:
        """Attempt to open a TCP connection, returning the socket or None."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
            s.settimeout(self._SEND_TIMEOUT_S)
            s.connect((self.laptop_ip, self.laptop_port))
            log.info("[sender] Connected to %s:%d", self.laptop_ip, self.laptop_port)
            return s
        except (OSError, ConnectionRefusedError) as exc:
            log.warning("[sender] Cannot connect to %s:%d — %s",
                        self.laptop_ip, self.laptop_port, exc)
            return None

    def _run(self) -> None:
        sock: Optional[socket.socket] = None

        while not self._stop.is_set():
            # Ensure we have a live connection
            if sock is None:
                sock = self._connect()
                if sock is None:
                    time.sleep(self._RECONNECT_DELAY_S)
                    continue

            # Drain the result queue and send
            try:
                item = self.result_queue.get(timeout=1.0)
            except queue.Empty:
                # Send a heartbeat keepalive every second of inactivity
                try:
                    sock.sendall(b'{"heartbeat":true}\n')
                except OSError:
                    sock.close()
                    sock = None
                continue

            if item is None:
                break   # Stop signal

            payload = (json.dumps(item, separators=(",", ":")) + "\n").encode("utf-8")
            try:
                sock.sendall(payload)
                self.sent_count += 1
            except (OSError, BrokenPipeError) as exc:
                log.warning("[sender] Send failed: %s — reconnecting", exc)
                self.drop_count += 1
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None

        if sock:
            try:
                sock.close()
            except OSError:
                pass


# ── shared stats ──────────────────────────────────────────────────────────────

STATS: Dict[str, int] = {
    "normal":   0,
    "ddos":     0,
    "botnet":   0,
    "other":    0,
}
STATS_LOCK = threading.Lock()


# ── pipeline factory ──────────────────────────────────────────────────────────

def build_pipeline(args):
    """
    Instantiate and wire all pipeline stages.

    Queue depths are sized to buffer ~1 second of traffic at the expected
    peak rate of ~1000 flows/sec to absorb burst without dropping.
    """
    model = NumpyMLP(args.model)
    classes = list(getattr(model, "class_names", CLASS_NAMES).values())
    log.info("[main] Model loaded: %s  (NumPy, no torch)  classes=%s",
             args.model, classes)

    Q_DEPTH = 16384

    pkt_q    = queue.Queue(maxsize=Q_DEPTH)
    flow_q   = queue.Queue(maxsize=Q_DEPTH)
    feat_q   = queue.Queue(maxsize=Q_DEPTH)
    result_q = queue.Queue(maxsize=4096)

    udp_rx    = UDPReceiver(port=args.udp_port, out_queue=pkt_q)
    flow_eng  = FlowEngine(
        in_queue    = pkt_q,
        out_queue   = flow_q,
        timeout_s   = args.flow_timeout,
        max_flows   = 65536,
        debug       = not args.quiet,
    )
    feat_ext  = FeatureExtractor(in_queue=flow_q, out_queue=feat_q)
    infer_eng = InferenceEngine(
        in_queue     = feat_q,
        result_queue = result_q,
        model        = model,
        stats        = STATS,
        stats_lock   = STATS_LOCK,
        debug        = not args.quiet,
    )
    sender    = ResultSender(
        result_queue = result_q,
        laptop_ip    = args.laptop_ip,
        laptop_port  = args.laptop_port,
    )

    return udp_rx, flow_eng, feat_ext, infer_eng, sender


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="FPGA-NIDS PYNQ Receiver — Flow Reconstruction + MLP Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",        default="model_fp.npz",
                   help="Path to NumPy .npz weight file (BN-folded float model)")
    p.add_argument("--udp_port",     type=int, default=5555,
                   help="UDP port to listen on (receives traffic from laptop)")
    p.add_argument("--laptop_ip",    default="192.168.2.1",
                   help="Laptop IP for streaming JSON results back")
    p.add_argument("--laptop_port",  type=int, default=9000,
                   help="Laptop TCP port for classification results")
    p.add_argument("--flow_timeout", type=float, default=1.0,
                   help="Flow idle timeout in seconds (lower = faster flow eviction)")
    p.add_argument("--quiet",        action="store_true",
                   help="Suppress per-flow console output")
    args = p.parse_args()

    log.info("[main] PYNQ NIDS Receiver starting")
    log.info("[main] UDP listen: 0.0.0.0:%d", args.udp_port)
    log.info("[main] Results → %s:%d", args.laptop_ip, args.laptop_port)

    stages = build_pipeline(args)
    udp_rx, flow_eng, feat_ext, infer_eng, sender = stages

    # Start pipeline stages in order (source first)
    udp_rx.start()
    flow_eng.start()
    feat_ext.start()
    infer_eng.start()
    sender.start()

    log.info("[main] All pipeline stages running. Ctrl-C to stop.")

    stop_event = threading.Event()

    def _shutdown(sig, frame):
        log.info("[main] Signal %s — shutting down...", sig)
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while not stop_event.is_set():
            time.sleep(5)
            with STATS_LOCK:
                totals = dict(STATS)
            log.info(
                "[main] udp_rx=%d  flows=%d  classified=%d  sent=%d  "
                "mean_infer=%.3f ms (recent %.3f ms)  stats=%s",
                udp_rx.rx_count,
                flow_eng.flows_completed,
                infer_eng.count,
                sender.sent_count,
                infer_eng.mean_latency_ms,
                infer_eng.recent_latency_ms,
                totals,
            )
    except KeyboardInterrupt:
        pass

    # Stop in reverse pipeline order (sink first)
    sender.stop()
    infer_eng.stop()
    feat_ext.stop()
    flow_eng.stop()
    udp_rx.stop()

    with STATS_LOCK:
        final = dict(STATS)
    log.info("[main] Final classification stats: %s", final)
    log.info("[main] Shutdown complete.")


if __name__ == "__main__":
    main()