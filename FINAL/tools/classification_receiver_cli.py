#!/usr/bin/env python3
"""
tools/classification_receiver_cli.py  —  OPTIONAL console-only alternative
to LAPTOP/dashboard_server.py.

Both this script and dashboard_server.py implement the same role: a TCP
server on port 9000 that receives newline-delimited JSON results from
receiver.py and tracks live statistics. They are NOT meant to run at the
same time (both bind the same TCP port by default).

Use dashboard_server.py for the live browser dashboard (the supported
demo path). Use this script only if you want a colour terminal dashboard
with no browser/Flask dependency — e.g. on a machine without a GUI.

    python3 tools/classification_receiver_cli.py --host 0.0.0.0 --port 9000

JSON message schema (one per line, newline-delimited):
{
  "flow_id":        int,
  "timestamp_ns":   int,
  "src_ip":         str,
  "dst_ip":         str,
  "src_port":       int,
  "dst_port":       int,
  "protocol":       str,          // "tcp" | "udp"
  "classification": str,          // "normal" | "ddos" | "botnet"
  "class_id":       int,
  "confidence":     float,        // [0.0, 1.0]
  "features": {
    "flow_duration":          float,
    "total_fwd_packets":      float,
    "total_bwd_packets":      float,
    "total_len_fwd_pkts":     float,
    "total_len_bwd_pkts":     float,
    "fwd_pkt_len_max":        float,
    "fwd_pkt_len_min":        float,
    "bwd_pkt_len_max":        float,
    "bwd_pkt_len_min":        float,
    "flow_bytes_per_sec":     float,
    "flow_packets_per_sec":   float,
    "flow_iat_mean":          float,
    "flow_iat_std":           float,
    "fwd_iat_mean":           float,
    "bwd_iat_mean":           float,
    "fwd_psh_flags":          float,
    "bwd_psh_flags":          float,
    "syn_flag_count":         float,
    "rst_flag_count":         float,
    "avg_fwd_segment_size":   float
  }
}

Usage:
    python3 classification_receiver.py --host 0.0.0.0 --port 9000

Requirements: stdlib only.
"""

import argparse
import collections
import json
import signal
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


# ── classification result dataclass ──────────────────────────────────────────

@dataclass
class ClassificationResult:
    """Parsed inference result from the PYNQ."""
    flow_id:        int
    timestamp_ns:   int
    src_ip:         str
    dst_ip:         str
    src_port:       int
    dst_port:       int
    protocol:       str
    classification: str
    class_id:       int
    confidence:     float
    inference_ms:   float
    features:       Dict[str, float]

    @classmethod
    def from_json(cls, obj: dict) -> "ClassificationResult":
        return cls(
            flow_id        = int(obj.get("flow_id", 0)),
            timestamp_ns   = int(obj.get("timestamp_ns", 0)),
            src_ip         = str(obj.get("src_ip", "")),
            dst_ip         = str(obj.get("dst_ip", "")),
            src_port       = int(obj.get("src_port", 0)),
            dst_port       = int(obj.get("dst_port", 0)),
            protocol       = str(obj.get("protocol", "unknown")),
            classification = str(obj.get("classification", "unknown")),
            class_id       = int(obj.get("class_id", -1)),
            confidence     = float(obj.get("confidence", 0.0)),
            inference_ms   = float(obj.get("inference_ms", 0.0)),
            features       = dict(obj.get("features", {})),
        )


# ── running statistics ────────────────────────────────────────────────────────

class LiveStats:
    """
    Thread-safe running statistics accumulator.

    Maintains:
    - Per-class flow counts
    - Rolling window of recent results for flows/sec calculation
    - Confidence value history (rolling 1000)
    - Top attack source IPs
    """

    _WINDOW_SECS = 10.0   # sliding window for rate calculation

    def __init__(self):
        self._lock           = threading.Lock()
        self.class_counts:   Dict[str, int]   = collections.defaultdict(int)
        self.total_flows:    int              = 0
        self.total_bytes:    int              = 0
        self._recent_ts:     Deque[float]     = collections.deque()
        self._confidences:   Deque[float]     = collections.deque(maxlen=1000)
        self._attack_srcs:   Dict[str, int]   = collections.defaultdict(int)
        self.start_time:     float            = time.monotonic()
        self.last_result:    Optional[ClassificationResult] = None

    def record(self, result: ClassificationResult) -> None:
        now = time.monotonic()
        with self._lock:
            self.class_counts[result.classification] += 1
            self.total_flows += 1
            self._recent_ts.append(now)
            self._confidences.append(result.confidence)
            self.last_result = result

            # Track attack source IPs (non-normal traffic)
            if result.classification not in ("normal",):
                self._attack_srcs[result.src_ip] += 1

            # Evict timestamps outside the sliding window
            cutoff = now - self._WINDOW_SECS
            while self._recent_ts and self._recent_ts[0] < cutoff:
                self._recent_ts.popleft()

    def snapshot(self) -> dict:
        """Return a consistent snapshot of current stats (no lock held by caller)."""
        now = time.monotonic()
        with self._lock:
            counts      = dict(self.class_counts)
            total       = self.total_flows
            window_count = len(self._recent_ts)
            flows_per_sec = window_count / self._WINDOW_SECS
            avg_conf    = (sum(self._confidences) / len(self._confidences)
                           if self._confidences else 0.0)
            elapsed     = now - self.start_time
            top_attacks = sorted(
                self._attack_srcs.items(), key=lambda x: x[1], reverse=True
            )[:5]
            last = self.last_result

        return {
            "total_flows":    total,
            "elapsed_s":      elapsed,
            "flows_per_sec":  flows_per_sec,
            "avg_confidence": avg_conf,
            "class_counts":   counts,
            "top_attack_srcs": top_attacks,
            "last_result":    last,
        }


# ── TCP client connection handler ─────────────────────────────────────────────

class ClientHandler(threading.Thread):
    """
    Handles one PYNQ TCP client connection.

    Reads newline-delimited JSON from the socket, parses each message,
    updates LiveStats, and optionally prints per-flow debug lines.
    """

    def __init__(
        self,
        conn: socket.socket,
        addr: tuple,
        stats: LiveStats,
        debug: bool = False,
    ):
        super().__init__(daemon=True, name=f"client-{addr[0]}:{addr[1]}")
        self.conn   = conn
        self.addr   = addr
        self.stats  = stats
        self.debug  = debug
        self.rx     = 0
        self.errors = 0

    def run(self) -> None:
        print(f"[receiver] PYNQ connected from {self.addr[0]}:{self.addr[1]}")
        buf = b""
        try:
            while True:
                chunk = self.conn.recv(65536)
                if not chunk:
                    break
                buf += chunk

                # Process all complete newline-delimited JSON messages
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._process_line(line)

        except (OSError, ConnectionResetError) as exc:
            print(f"[receiver] Connection {self.addr} closed: {exc}")
        finally:
            self.conn.close()
            print(f"[receiver] {self.addr} disconnected. "
                  f"rx={self.rx}  errors={self.errors}")

    def _process_line(self, line: bytes) -> None:
        try:
            obj    = json.loads(line.decode("utf-8"))
            result = ClassificationResult.from_json(obj)
            self.stats.record(result)
            self.rx += 1

            if self.debug:
                print(
                    f"  [flow {result.flow_id:06d}] "
                    f"{result.src_ip}:{result.src_port} → "
                    f"{result.dst_ip}:{result.dst_port}  "
                    f"proto={result.protocol}  "
                    f"class={result.classification}  "
                    f"conf={result.confidence:.3f}  "
                    f"infer={result.inference_ms:.3f}ms"
                )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            self.errors += 1
            if self.debug:
                print(f"  [receiver] Parse error: {exc}  raw={line[:80]!r}")


# ── TCP server ────────────────────────────────────────────────────────────────

class ClassificationServer:
    """
    TCP server that accepts connections from the PYNQ and routes each
    connection to a ClientHandler thread.
    """

    def __init__(self, host: str, port: int, stats: LiveStats, debug: bool = False):
        self.host     = host
        self.port     = port
        self.stats    = stats
        self.debug    = debug
        self._stop    = threading.Event()
        self._sock:   Optional[socket.socket] = None
        self._thread  = threading.Thread(target=self._run, daemon=True, name="tcp_server")

    def start(self) -> None:
        self._thread.start()
        print(f"[server] TCP classification receiver listening on {self.host}:{self.port}")

    def stop(self) -> None:
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        self._thread.join(timeout=3.0)
        print("[server] Stopped.")

    def _run(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        self._sock.settimeout(1.0)
        self._sock.bind((self.host, self.port))
        self._sock.listen(4)

        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            handler = ClientHandler(conn, addr, self.stats, debug=self.debug)
            handler.start()


# ── dashboard printer ─────────────────────────────────────────────────────────

# ANSI colour codes for terminal output
_RESET  = "\033[0m"
_RED    = "\033[31m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"

_CLASS_COLORS = {
    "normal":   _GREEN,
    "ddos":     _RED,
    "botnet":   _YELLOW,
    "other":    _CYAN,
    "unknown":  _RESET,
}


class DashboardPrinter(threading.Thread):
    """
    Periodically clears the terminal and prints a live statistics dashboard.
    """

    def __init__(self, stats: LiveStats, stop_event: threading.Event, interval: float = 3.0):
        super().__init__(daemon=True, name="dashboard")
        self.stats      = stats
        self.stop_event = stop_event
        self.interval   = interval

    def run(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(self.interval)
            self._print()

    def _print(self) -> None:
        snap = self.stats.snapshot()
        now  = time.strftime("%H:%M:%S")
        W    = 70
        sep  = "═" * W

        print(f"\n╔{sep}╗")
        print(f"║  {_BOLD}NIDS Classification Receiver — Live Dashboard{_RESET}"
              f"{'':>{W - 47}}  ║")
        print(f"║  Time: {now}   "
              f"Elapsed: {snap['elapsed_s']:.0f}s   "
              f"Total flows: {snap['total_flows']:,}"
              f"{'':>10}║")
        print(f"╠{sep}╣")

        # Flows/sec
        fps = snap["flows_per_sec"]
        print(f"║  Flows/sec (10s window): {_BOLD}{fps:6.1f}{_RESET}"
              f"   Avg confidence: {snap['avg_confidence']:.3f}"
              f"{'':>14}║")
        print(f"╠{sep}╣")

        # Per-class counts
        print(f"║  {'Classification':<18} {'Count':>10}  {'%':>7}  {'Bar':<20}  ║")
        print(f"╠{'─'*W}╣")
        total = max(snap["total_flows"], 1)
        for cls, count in sorted(snap["class_counts"].items(),
                                  key=lambda x: x[1], reverse=True):
            pct  = count / total * 100
            bar  = "█" * int(pct / 5)   # 20 chars = 100%
            color = _CLASS_COLORS.get(cls, _RESET)
            print(f"║  {color}{cls:<18}{_RESET} {count:>10,}  {pct:>6.1f}%  {bar:<20}  ║")

        # Top attack sources
        if snap["top_attack_srcs"]:
            print(f"╠{sep}╣")
            print(f"║  Top Attack Sources{' ' * (W - 20)}║")
            for ip, cnt in snap["top_attack_srcs"]:
                print(f"║    {_RED}{ip:<18}{_RESET}  {cnt:>8,} flows{' ' * (W - 38)}║")

        # Last classified flow
        lr = snap["last_result"]
        if lr:
            print(f"╠{sep}╣")
            color = _CLASS_COLORS.get(lr.classification, _RESET)
            print(
                f"║  Last: flow {lr.flow_id:06d}  "
                f"{lr.src_ip}:{lr.src_port} → {lr.dst_ip}:{lr.dst_port}  "
                f"{color}{lr.classification}{_RESET} ({lr.confidence:.3f})"
                f"{' ' * max(0, W - 60)}║"
            )

        print(f"╚{sep}╝")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FPGA-NIDS Classification Result Receiver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",     default="0.0.0.0",  help="TCP bind address")
    parser.add_argument("--port",     type=int, default=9000, help="TCP listen port")
    parser.add_argument("--debug",    action="store_true", help="Print per-flow lines")
    parser.add_argument("--interval", type=float, default=3.0,
                        help="Dashboard refresh interval (seconds)")
    args = parser.parse_args()

    stats      = LiveStats()
    stop_event = threading.Event()

    server     = ClassificationServer(args.host, args.port, stats, debug=args.debug)
    dashboard  = DashboardPrinter(stats, stop_event, interval=args.interval)

    def _shutdown(sig, frame):
        print(f"\n[receiver] Signal {sig} — shutting down...")
        stop_event.set()
        server.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.start()
    dashboard.start()

    print(f"[receiver] Waiting for PYNQ at {args.host}:{args.port} ...")
    print(f"[receiver] Dashboard refreshes every {args.interval}s. Ctrl-C to stop.\n")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)

    # Print final summary
    snap = stats.snapshot()
    print(f"\n[receiver] Final summary:")
    print(f"  Total flows classified: {snap['total_flows']:,}")
    print(f"  Runtime:                {snap['elapsed_s']:.1f}s")
    for cls, cnt in sorted(snap["class_counts"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:<20} {cnt:>8,}")


if __name__ == "__main__":
    main()