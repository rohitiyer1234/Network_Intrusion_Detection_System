''' """
receiver.py  —  PYNQ-side real-time network traffic classifier.

Receives synthetic UDP frames from traffic_generator.py, reconstructs
bidirectional flows, extracts 20 CICIDS-aligned features, runs the MLP
classifier, maintains running class counts, and prints per-flow results.

The WebSocket streaming of stats is handled by websocket_client.py which
reads from a shared stats dict updated here.

Architecture (all threaded, queue-connected):
    UDP socket
        ↓  ParsedPacket queue
    FlowEngine           — builds bidirectional flows, evicts on TCP FIN/RST or timeout
        ↓  FlowRecord queue
    FeatureExtractor     — computes 20-feature vector (from extractor.py)
        ↓  ExtractedFlow queue
    InferenceThread      — runs model.predict(), updates stats
        ↓  InferenceResult (printed + stats dict)

Usage:
    python3 receiver.py --model best_model.pth --port 5555

Requirements (PYNQ):
    pip install torch numpy pyyaml websockets
"""

import argparse
import queue
import socket
import struct
import threading
import time
import logging
from typing import Dict, NamedTuple, Optional

import torch

# ── local imports (same directory) ───────────────────────────────────────────

# These modules are taken verbatim from the NETWORK02 package.
# Copy extractor.py, flow_engine.py, capture.py, quantizer.py, and
# the config/ directory alongside this file before running.

from capture     import ParsedPacket
from flow_engine import FlowEngine, FlowRecord
from extractor   import FeatureExtractor, ExtractedFlow, features_to_vector, extract_features
from model       import load_model, normalize_features, CLASS_NAMES
from websocket_client import start_ws_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("receiver")

# ── wire format (must match traffic_generator.py) ────────────────────────────
# B  4s  4s  HH  HBH  Q = 1+4+4+2+2+2+1+2+8 = 26 bytes, padded to 60

FRAME_STRUCT = struct.Struct("!B4s4sHHHBHQ")
FRAME_SIZE   = 60   # bytes


def _unpack_frame(data: bytes) -> Optional[ParsedPacket]:
    """Decode one 60-byte frame from traffic_generator into a ParsedPacket."""
    if len(data) < FRAME_STRUCT.size:
        return None
    try:
        (proto, src_raw, dst_raw, sport, dport,
         plen, flags, ip_total, ts_ns) = FRAME_STRUCT.unpack_from(data)
    except struct.error:
        return None

    if proto not in (6, 17):
        return None

    src_ip = socket.inet_ntoa(src_raw)
    dst_ip = socket.inet_ntoa(dst_raw)
    return ParsedPacket(
        ts_ns        = ts_ns,
        src_ip       = src_ip,
        dst_ip       = dst_ip,
        protocol     = proto,
        src_port     = sport,
        dst_port     = dport,
        payload_len  = plen,
        tcp_flags    = flags,
        ip_total_len = ip_total,
    )


# ── UDP receiver thread ───────────────────────────────────────────────────────

class UDPReceiver:
    """Reads UDP datagrams and pushes ParsedPacket objects onto out_queue."""

    def __init__(self, port: int, out_queue: queue.Queue):
        self.port       = port
        self.out_queue  = out_queue
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True,
                                           name="udp_receiver")
        self.rx_count   = 0
        self.err_count  = 0

    def start(self) -> None:
        self._thread.start()
        log.info("UDP receiver listening on port %d", self.port)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        sock.settimeout(1.0)
        sock.bind(("0.0.0.0", self.port))
        log.info("[udp] Bound to 0.0.0.0:%d", self.port)

        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(FRAME_SIZE + 16)
            except socket.timeout:
                continue
            except OSError as exc:
                log.error("[udp] Recv error: %s", exc)
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
                pass  # drop under burst

        sock.close()
        log.info("[udp] Stopped. rx=%d err=%d", self.rx_count, self.err_count)


# ── inference thread ──────────────────────────────────────────────────────────

class InferenceThread:
    """
    Reads ExtractedFlow objects from in_queue, runs model inference, updates
    the shared stats dict, and prints per-flow debug lines.
    """

    def __init__(self, in_queue: queue.Queue,
                 model: torch.nn.Module,
                 stats: Dict[str, int],
                 stats_lock: threading.Lock,
                 debug: bool = True):
        self.in_queue   = in_queue
        self.model      = model
        self.stats      = stats
        self.stats_lock = stats_lock
        self.debug      = debug
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True,
                                           name="inference")
        self.count      = 0

    def start(self) -> None:
        self._thread.start()
        log.info("Inference thread started.")

    def stop(self) -> None:
        self._stop.set()
        try:
            self.in_queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        log.info("Inference thread stopped. classified=%d", self.count)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.in_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            ef: ExtractedFlow = item
            tensor = normalize_features(ef.feature_vec)

            try:
                classes, confs = self.model.predict(tensor)
            except Exception as exc:
                log.error("[inference] Model error: %s", exc)
                continue

            cls_id   = int(classes[0])
            conf     = float(confs[0])
            cls_name = CLASS_NAMES.get(cls_id, "other")

            with self.stats_lock:
                self.stats[cls_name] = self.stats.get(cls_name, 0) + 1

            self.count += 1

            if self.debug:
                print(
                    f"[flow {ef.flow_id:06d}] "
                    f"{ef.key[0]}:{ef.key[2]} → {ef.key[1]}:{ef.key[3]} "
                    f"| class={cls_name} ({cls_id}) conf={conf:.3f}"
                )

            # Print running totals every 50 flows
            if self.count % 50 == 0:
                with self.stats_lock:
                    totals = dict(self.stats)
                print(f"[stats] total_flows={self.count}  counts={totals}")


# ── main ──────────────────────────────────────────────────────────────────────

# Shared stats dict — written by InferenceThread, read by websocket_client.py
STATS: Dict[str, int] = {
    "normal":   0,
    "ddos":     0,
    "botnet":   0,
    "other":    0,
}
STATS_LOCK = threading.Lock()


def build_pipeline(args) -> tuple:
    """Instantiate and wire all pipeline stages.  Returns list of stoppable objects."""

    model = load_model(args.model)
    log.info("Model loaded from %s", args.model)

    Q_DEPTH = 16384

    pkt_q    = queue.Queue(maxsize=Q_DEPTH)
    flow_q   = queue.Queue(maxsize=Q_DEPTH)
    feat_q   = queue.Queue(maxsize=Q_DEPTH)

    udp_rx    = UDPReceiver(port=args.port, out_queue=pkt_q)
    flow_eng  = FlowEngine(in_queue=pkt_q, out_queue=flow_q,
                           timeout_s=args.flow_timeout, max_flows=65536)
    feat_ext  = FeatureExtractor(in_queue=flow_q, out_queue=feat_q)
    infer_thr = InferenceThread(in_queue=feat_q, model=model,
                                stats=STATS, stats_lock=STATS_LOCK,
                                debug=not args.quiet)

    return udp_rx, flow_eng, feat_ext, infer_thr


def main() -> None:
    p = argparse.ArgumentParser(description="Local real-time traffic classifier receiver")
    p.add_argument("--model",        default="best_model.pth", help="Path to .pth model file")
    p.add_argument("--port",         type=int, default=5555,   help="UDP listen port (default 5555)")
    p.add_argument("--flow_timeout", type=float, default=1.0,  help="Flow idle timeout in seconds")
    p.add_argument("--quiet",        action="store_true",       help="Suppress per-flow debug output")
    args = p.parse_args()

    udp_rx, flow_eng, feat_ext, infer_thr = build_pipeline(args)

    # Start all stages
    udp_rx.start()
    flow_eng.start()
    feat_ext.start()
    infer_thr.start()

    ws = start_ws_server(STATS, STATS_LOCK)

    log.info("Pipeline running. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(5)
            log.info(
                "RX pkts=%d  flows=%d  classified=%d  stats=%s",
                udp_rx.rx_count,
                flow_eng.flows_completed,
                infer_thr.count,
                STATS,
            )
    except KeyboardInterrupt:
        log.info("Shutting down…")

    ws.stop()

    # Stop in reverse order
    infer_thr.stop()
    feat_ext.stop()
    flow_eng.stop()
    udp_rx.stop()

    log.info("Final stats: %s", STATS)


if __name__ == "__main__":
    main()
'''

"""
receiver.py  —  PYNQ-side real-time network traffic classifier.

Receives synthetic UDP frames from traffic_generator.py, reconstructs
bidirectional flows, extracts 20 CICIDS-aligned features, runs the MLP
classifier, maintains running class counts, and prints per-flow results.

The WebSocket streaming of stats is handled by websocket_client.py which
reads from a shared stats dict updated here.

Architecture (all threaded, queue-connected):
    UDP socket
        ↓  ParsedPacket queue
    FlowEngine           — builds bidirectional flows, evicts on TCP FIN/RST or timeout
        ↓  FlowRecord queue
    FeatureExtractor     — computes 20-feature vector (from extractor.py)
        ↓  ExtractedFlow queue
    InferenceThread      — runs model.predict(), updates stats
        ↓  InferenceResult (printed + stats dict)

Usage:
    python3 receiver.py --model best_model.pth --port 5555

Requirements (PYNQ):
    pip install torch numpy pyyaml websockets
"""

import argparse
import queue
import socket
import struct
import threading
import time
import logging
from typing import Dict, NamedTuple, Optional

import torch

# ── local imports (same directory) ───────────────────────────────────────────

# These modules are taken verbatim from the NETWORK02 package.
# Copy extractor.py, flow_engine.py, capture.py, quantizer.py, and
# the config/ directory alongside this file before running.

from capture     import ParsedPacket
from flow_engine import FlowEngine, FlowRecord
from extractor   import FeatureExtractor, ExtractedFlow, features_to_vector, extract_features
from model       import load_model, normalize_features, CLASS_NAMES
from websocket_client import start_ws_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("receiver")

# ── wire format (must match traffic_generator.py) ────────────────────────────
# B  4s  4s  HH  HBH  Q = 1+4+4+2+2+2+1+2+8 = 26 bytes, padded to 60

FRAME_STRUCT = struct.Struct("!B4s4sHHHBHQ")
FRAME_SIZE   = 60   # bytes


def _unpack_frame(data: bytes) -> Optional[ParsedPacket]:
    """Decode one 60-byte frame from traffic_generator into a ParsedPacket."""
    if len(data) < FRAME_STRUCT.size:
        return None
    try:
        (proto, src_raw, dst_raw, sport, dport,
         plen, flags, ip_total, ts_ns) = FRAME_STRUCT.unpack_from(data)
    except struct.error:
        return None

    if proto not in (6, 17):
        return None

    src_ip = socket.inet_ntoa(src_raw)
    dst_ip = socket.inet_ntoa(dst_raw)
    return ParsedPacket(
        ts_ns        = ts_ns,
        src_ip       = src_ip,
        dst_ip       = dst_ip,
        protocol     = proto,
        src_port     = sport,
        dst_port     = dport,
        payload_len  = plen,
        tcp_flags    = flags,
        ip_total_len = ip_total,
    )


# ── UDP receiver thread ───────────────────────────────────────────────────────

class UDPReceiver:
    """Reads UDP datagrams and pushes ParsedPacket objects onto out_queue."""

    def __init__(self, port: int, out_queue: queue.Queue):
        self.port       = port
        self.out_queue  = out_queue
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True,
                                           name="udp_receiver")
        self.rx_count   = 0
        self.err_count  = 0

    def start(self) -> None:
        self._thread.start()
        log.info("UDP receiver listening on port %d", self.port)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        sock.settimeout(1.0)
        sock.bind(("0.0.0.0", self.port))
        log.info("[udp] Bound to 0.0.0.0:%d", self.port)

        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(FRAME_SIZE + 16)
            except socket.timeout:
                continue
            except OSError as exc:
                log.error("[udp] Recv error: %s", exc)
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
                pass  # drop under burst

        sock.close()
        log.info("[udp] Stopped. rx=%d err=%d", self.rx_count, self.err_count)


# ── inference thread ──────────────────────────────────────────────────────────

class InferenceThread:
    """
    Reads ExtractedFlow objects from in_queue, runs model inference, updates
    the shared stats dict, and prints per-flow debug lines.
    """

    def __init__(self, in_queue: queue.Queue,
                 model: torch.nn.Module,
                 stats: Dict[str, int],
                 stats_lock: threading.Lock,
                 debug: bool = True,
                 debug_features: bool = False):
        self.in_queue   = in_queue
        self.model      = model
        self.stats      = stats
        self.stats_lock = stats_lock
        self.debug      = debug
        self.debug_features = debug_features
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True,
                                           name="inference")
        self.count      = 0

    def start(self) -> None:
        self._thread.start()
        log.info("Inference thread started.")

    def stop(self) -> None:
        self._stop.set()
        try:
            self.in_queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        log.info("Inference thread stopped. classified=%d", self.count)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.in_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            ef: ExtractedFlow = item
            tensor = normalize_features(ef.feature_vec)

            # ── FEATURE DEBUG ─────────────────────────────────────────────
            # Enabled with --debug_features flag.  Prints the raw feature
            # vector and its normalized counterpart so you can verify that
            # training data and runtime data are in the same range.
            if self.debug_features:
                from constants import FEATURE_ORDER
                print(f"\n[debug] flow {ef.flow_id:06d}  raw feature_vec:")
                for name, val in zip(FEATURE_ORDER, ef.feature_vec):
                    print(f"  {name:<28s} = {val:.6g}")
                normed = tensor.squeeze(0).tolist()
                print(f"[debug] flow {ef.flow_id:06d}  normalized tensor:")
                for name, val in zip(FEATURE_ORDER, normed):
                    print(f"  {name:<28s} = {val:.6f}")
            # ─────────────────────────────────────────────────────────────

            try:
                classes, confs = self.model.predict(tensor)
            except Exception as exc:
                log.error("[inference] Model error: %s", exc)
                continue

            cls_id   = int(classes[0])
            conf     = float(confs[0])
            cls_name = CLASS_NAMES.get(cls_id, "unknown")

            with self.stats_lock:
                self.stats[cls_name] = self.stats.get(cls_name, 0) + 1

            self.count += 1

            if self.debug:
                print(
                    f"[flow {ef.flow_id:06d}] "
                    f"{ef.key[0]}:{ef.key[2]} → {ef.key[1]}:{ef.key[3]} "
                    f"| class={cls_name} ({cls_id}) conf={conf:.3f}"
                )

            # Print running totals every 50 flows
            if self.count % 50 == 0:
                with self.stats_lock:
                    totals = dict(self.stats)
                print(f"[stats] total_flows={self.count}  counts={totals}")


# ── main ──────────────────────────────────────────────────────────────────────

# Shared stats dict — written by InferenceThread, read by websocket_client.py
STATS: Dict[str, int] = {
    "normal":   0,
    "ddos":     0,
    "botnet":   0
}
STATS_LOCK = threading.Lock()


def build_pipeline(args) -> tuple:
    """Instantiate and wire all pipeline stages.  Returns list of stoppable objects."""

    model = load_model(args.model)
    log.info("Model loaded from %s", args.model)

    Q_DEPTH = 16384

    pkt_q    = queue.Queue(maxsize=Q_DEPTH)
    flow_q   = queue.Queue(maxsize=Q_DEPTH)
    feat_q   = queue.Queue(maxsize=Q_DEPTH)

    udp_rx    = UDPReceiver(port=args.port, out_queue=pkt_q)
    flow_eng  = FlowEngine(in_queue=pkt_q, out_queue=flow_q,
                           timeout_s=args.flow_timeout, max_flows=65536)
    feat_ext  = FeatureExtractor(in_queue=flow_q, out_queue=feat_q)
    infer_thr = InferenceThread(in_queue=feat_q, model=model,
                                stats=STATS, stats_lock=STATS_LOCK,
                                debug=not args.quiet,
                                debug_features=args.debug_features)

    return udp_rx, flow_eng, feat_ext, infer_thr


def main() -> None:
    p = argparse.ArgumentParser(description="Local real-time traffic classifier receiver")
    p.add_argument("--model",        default="best_model.pth", help="Path to .pth model file")
    p.add_argument("--port",         type=int, default=5555,   help="UDP listen port (default 5555)")
    p.add_argument("--flow_timeout", type=float, default=1.0,  help="Flow idle timeout in seconds")
    p.add_argument("--quiet",         action="store_true",       help="Suppress per-flow debug output")
    p.add_argument("--debug_features", action="store_true",
                   help="Print raw and normalized feature_vec for every flow (for pipeline alignment verification)")
    args = p.parse_args()

    udp_rx, flow_eng, feat_ext, infer_thr = build_pipeline(args)

    # Start all stages
    udp_rx.start()
    flow_eng.start()
    feat_ext.start()
    infer_thr.start()

    ws = start_ws_server(STATS, STATS_LOCK)

    log.info("Pipeline running. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(5)
            log.info(
                "RX pkts=%d  flows=%d  classified=%d  stats=%s",
                udp_rx.rx_count,
                flow_eng.flows_completed,
                infer_thr.count,
                STATS,
            )
    except KeyboardInterrupt:
        log.info("Shutting down…")

    ws.stop()

    # Stop in reverse order
    infer_thr.stop()
    feat_ext.stop()
    flow_eng.stop()
    udp_rx.stop()

    log.info("Final stats: %s", STATS)


if __name__ == "__main__":
    main()