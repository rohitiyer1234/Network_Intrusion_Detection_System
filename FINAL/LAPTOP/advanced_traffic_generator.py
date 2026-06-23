#!/usr/bin/env python3
"""
advanced_traffic_generator.py  —  Laptop-side concurrent synthetic traffic generator.

Generates 9 concurrent traffic patterns simultaneously using one thread per class.
Each thread uses Poisson-distributed inter-arrival times with random jitter,
producing realistic overlapping flows that stress-test the PYNQ classifier pipeline.
Patterns map to the 3 model classes (normal / ddos / botnet); the port-scan
threads are out-of-distribution traffic kept to exercise robustness.

Wire format (60 bytes, big-endian):
    Offset  Len   Field
    0       1     Protocol  (6=TCP, 17=UDP)
    1       4     Src IP    (packed 4 bytes)
    5       4     Dst IP    (packed 4 bytes)
    9       2     Src port
    11      2     Dst port
    13      2     Payload length
    15      1     TCP flags
    16      2     IP total length
    18      8     Timestamp nanoseconds
    26      34    Padding (zero-filled)

Total: 60 bytes exactly.  struct format: "!B4s4sHHHBHQ"

Traffic patterns (all concurrent):
    1. Normal Web Browsing      — TCP, HTTP/HTTPS, bidirectional, PSH+ACK   [normal]
    2. DNS Chatter              — UDP, port 53, small bursty packets        [normal]
    3. UDP Video Streaming      — RTP-like UDP, large payloads              [normal]
    4. UDP DDoS Flood           — many source IPs, one victim, high rate    [ddos]
    5. TCP SYN Flood            — SYN-only, incomplete handshakes           [ddos]
    6. Botnet C2 Beaconing      — bots phoning home to C2, bursty, small    [botnet]
    7. Botnet Exfiltration      — large outbound push to C2 host            [botnet]
    8. Vertical Port Scan       — one src, one dst, sequential ports        [OOD]
    9. Horizontal Port Scan     — one src, many dsts, fixed port            [OOD]

Usage:
    python3 advanced_traffic_generator.py --target_ip 192.168.2.99 --target_port 5555

Requirements: stdlib only (no Scapy, no root).
"""

import argparse
import math
import os
import random
import signal
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ── wire format ───────────────────────────────────────────────────────────────

FRAME_STRUCT  = struct.Struct("!B4s4sHHHBHQ")
FRAME_SIZE    = 60        # total bytes per datagram (header=26, padding=34)
PADDING       = bytes(FRAME_SIZE - FRAME_STRUCT.size)   # 34 zero bytes

# TCP flag constants
TCP_SYN = 0x02
TCP_ACK = 0x10
TCP_FIN = 0x01
TCP_RST = 0x04
TCP_PSH = 0x08
TCP_URG = 0x20

PROTO_TCP = 6
PROTO_UDP = 17


# ── utilities ─────────────────────────────────────────────────────────────────

def _pack_ip(dotted: str) -> bytes:
    """Convert dotted-decimal IP string to packed 4-byte big-endian bytes."""
    return socket.inet_aton(dotted)


def build_frame(
    proto: int,
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    payload_len: int,
    tcp_flags: int,
    ip_total_len: int,
    ts_ns: int,
) -> bytes:
    """
    Pack one 60-byte synthetic packet frame.

    The PYNQ receiver decodes this with the same struct format to reconstruct
    a ParsedPacket and feed it into the flow engine.
    """
    header = FRAME_STRUCT.pack(
        proto,
        _pack_ip(src_ip),
        _pack_ip(dst_ip),
        src_port,
        dst_port,
        payload_len,
        tcp_flags,
        ip_total_len,
        ts_ns,
    )
    return header + PADDING


def poisson_interval(rate_hz: float) -> float:
    """
    Return the next inter-arrival time drawn from an exponential distribution,
    which is the inter-event time for a Poisson process at `rate_hz` events/sec.

    Clipped to at most 5 seconds to prevent starvation at very low rates.
    """
    if rate_hz <= 0:
        return 1.0
    return min(-math.log(1.0 - random.random()) / rate_hz, 5.0)


def jitter(base: float, fraction: float = 0.2) -> float:
    """Add ±fraction random jitter to a base value. Returns non-negative result."""
    return max(0.0, base * (1.0 + random.uniform(-fraction, fraction)))


def rand_ip(prefix: str) -> str:
    """Generate a random IP in a /24 subnet given a prefix like '10.0.1.'"""
    return prefix + str(random.randint(2, 254))


def rand_port_ephemeral() -> int:
    """Random ephemeral port in the IANA range 49152–65535."""
    return random.randint(49152, 65535)


# ── per-thread statistics ─────────────────────────────────────────────────────

@dataclass
class ThreadStats:
    """Mutable counters updated by each generator thread."""
    name:        str
    sent:        int = 0
    errors:      int = 0
    start_time:  float = field(default_factory=time.monotonic)
    lock:        threading.Lock = field(default_factory=threading.Lock)

    def record_sent(self) -> None:
        with self.lock:
            self.sent += 1

    def record_error(self) -> None:
        with self.lock:
            self.errors += 1

    def pps(self) -> float:
        """Current average packets/sec since thread start."""
        elapsed = time.monotonic() - self.start_time
        with self.lock:
            return self.sent / elapsed if elapsed > 0 else 0.0


# ── base generator thread ─────────────────────────────────────────────────────

class GeneratorThread(threading.Thread):
    """
    Abstract base for all traffic generator threads.

    Subclasses implement _next_packet() to produce one (fields...) tuple.
    The base class handles socket management, Poisson timing, jitter,
    and graceful shutdown via the shared stop_event.
    """

    def __init__(
        self,
        name: str,
        target_ip: str,
        target_port: int,
        rate_hz: float,
        sock: socket.socket,
        stop_event: threading.Event,
        stats: ThreadStats,
    ):
        super().__init__(name=name, daemon=True)
        self.target      = (target_ip, target_port)
        self.rate_hz     = rate_hz
        self.sock        = sock
        self.stop_event  = stop_event
        self.stats       = stats

    def _next_packet(self) -> Optional[Tuple]:
        """
        Return a tuple of (proto, src_ip, dst_ip, sport, dport,
        payload_len, tcp_flags, ip_total_len) for the next packet.

        Return None to skip this iteration without sending.
        """
        raise NotImplementedError

    def run(self) -> None:
        """Main loop: Poisson-timed sends with jitter."""
        next_send = time.monotonic()

        while not self.stop_event.is_set():
            now = time.monotonic()

            # Busy-wait with small sleep when ahead of schedule
            if now < next_send:
                time.sleep(max(0, min(next_send - now, 0.001)))
                continue

            fields = self._next_packet()
            if fields is not None:
                ts_ns = time.time_ns()
                frame = build_frame(*fields, ts_ns=ts_ns)
                try:
                    self.sock.sendto(frame, self.target)
                    self.stats.record_sent()
                except OSError:
                    self.stats.record_error()

            # Schedule next send: Poisson interval with jitter
            interval  = poisson_interval(self.rate_hz)
            next_send = now + jitter(interval, fraction=0.15)


# ── traffic class 1: normal web browsing ─────────────────────────────────────

class NormalWebThread(GeneratorThread):
    """
    Simulates realistic HTTP/HTTPS web browsing traffic.

    - TCP, bidirectional flows
    - Ports: 80, 443, 8080
    - PSH+ACK for data transfer, SYN for new sessions, FIN for teardown
    - Varied packet sizes (40–1400 bytes)
    - Multiple concurrent sessions maintained across calls
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sessions: List[Dict] = []   # active simulated sessions
        self._session_lock = threading.Lock()

    def _new_session(self) -> Dict:
        return {
            "src_ip":  rand_ip("10.0.0."),
            "dst_ip":  rand_ip("10.0.1."),
            "sport":   rand_port_ephemeral(),
            "dport":   random.choice([80, 443, 8080]),
            "pkts":    0,
            "max_pkts": random.randint(8, 60),   # session lifetime in packets
        }

    def _next_packet(self) -> Optional[Tuple]:
        with self._session_lock:
            # Spawn new sessions stochastically
            if len(self._sessions) < 3 or random.random() < 0.04:
                self._sessions.append(self._new_session())

            if not self._sessions:
                return None

            s = random.choice(self._sessions)
            s["pkts"] += 1

            # Alternate direction: even packets go src→dst, odd go dst→src
            if s["pkts"] % 3 != 0:
                src_ip, dst_ip = s["src_ip"], s["dst_ip"]
                sport, dport   = s["sport"], s["dport"]
            else:
                src_ip, dst_ip = s["dst_ip"], s["src_ip"]
                sport, dport   = s["dport"], s["sport"]

            # Determine flags based on session state
            if s["pkts"] == 1:
                flags = TCP_SYN
                plen  = 0
            elif s["pkts"] >= s["max_pkts"]:
                flags = TCP_FIN | TCP_ACK
                plen  = random.randint(0, 100)
                self._sessions.remove(s)
            else:
                flags = TCP_ACK | TCP_PSH
                plen  = random.randint(40, 1400)

        return (PROTO_TCP, src_ip, dst_ip, sport, dport, plen, flags, plen + 40)


# ── traffic class 2: DNS chatter ──────────────────────────────────────────────

class DNSChatterThread(GeneratorThread):
    """
    Simulates DNS query/response traffic.

    - UDP
    - Destination port 53
    - Small packets (28–512 bytes)
    - Bursty behavior: clusters of queries followed by silence
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._burst_remaining = 0

    def _next_packet(self) -> Optional[Tuple]:
        # Burst mode: if in a burst, send back-to-back
        if self._burst_remaining > 0:
            self._burst_remaining -= 1
        elif random.random() < 0.3:
            # Start new burst of 1–8 queries
            self._burst_remaining = random.randint(1, 8)
        else:
            return None   # Silence period: skip this tick

        src_ip = rand_ip(random.choice(["10.0.0.", "10.0.1."]))
        sport  = rand_port_ephemeral()
        plen   = random.choice([28, 44, 60, 128, 256, 512])   # DNS sizes
        # 20% of traffic is responses (src port 53)
        if random.random() < 0.2:
            return (PROTO_UDP, rand_ip("10.0.1."), src_ip, 53, sport, plen, 0, plen + 28)
        return (PROTO_UDP, src_ip, rand_ip("10.0.1."), sport, 53, plen, 0, plen + 28)


# ── traffic class 3: UDP video streaming ─────────────────────────────────────

class VideoStreamThread(GeneratorThread):
    """
    Simulates RTP-like UDP video streaming.

    - UDP, high payload (800–1400 bytes per packet)
    - Continuous high-throughput flow
    - Low and stable inter-arrival times
    - Multiple concurrent streams
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._streams: List[Dict] = []

    def _new_stream(self) -> Dict:
        return {
            "src_ip": rand_ip("10.0.0."),
            "dst_ip": rand_ip("10.0.1."),
            "sport":  random.randint(5004, 5100),  # RTP-range ports
            "dport":  random.randint(5004, 5100),
            "seq":    0,
        }

    def _next_packet(self) -> Optional[Tuple]:
        if len(self._streams) < 2 or random.random() < 0.01:
            self._streams.append(self._new_stream())
        if len(self._streams) > 5:
            self._streams.pop(0)   # Evict oldest stream

        s = random.choice(self._streams)
        s["seq"] += 1
        plen = random.randint(800, 1400)   # Large RTP payload
        return (PROTO_UDP, s["src_ip"], s["dst_ip"], s["sport"], s["dport"], plen, 0, plen + 28)


# ── traffic class 4: UDP DDoS flood ──────────────────────────────────────────

class UDPDDoSThread(GeneratorThread):
    """
    Simulates a volumetric UDP DDoS flood attack.

    - Many random source IPs (spoofed)
    - Single victim IP and port
    - Very high packet rate with large payloads
    - Randomized source ports
    """

    VICTIM_IP   = "10.0.1.200"
    VICTIM_PORT = 80

    # Random-looking source prefixes (spoofed addresses)
    _SRC_NETS = ["5.5.5.", "8.8.8.", "1.2.3.", "77.77.77.", "100.200."]

    def _next_packet(self) -> Optional[Tuple]:
        src_ip  = rand_ip(random.choice(self._SRC_NETS))
        sport   = random.randint(1024, 65535)
        plen    = random.randint(512, 1400)   # Large flood packets
        return (PROTO_UDP, src_ip, self.VICTIM_IP, sport, self.VICTIM_PORT, plen, 0, plen + 28)


# ── traffic class 5: TCP SYN flood ────────────────────────────────────────────

class SYNFloodThread(GeneratorThread):
    """
    Simulates a TCP SYN flood attack (incomplete handshakes).

    - Only SYN packets, never ACK
    - Many random source IPs
    - One victim IP and port
    - High rate to exhaust connection table
    """

    VICTIM_IP   = "10.0.1.100"
    VICTIM_PORT = 443

    def _next_packet(self) -> Optional[Tuple]:
        src_ip = rand_ip(random.choice(["6.6.6.", "7.7.7.", "9.9.9.", "11.22."]))
        sport  = random.randint(1024, 65535)
        return (PROTO_TCP, src_ip, self.VICTIM_IP, sport, self.VICTIM_PORT, 0, TCP_SYN, 40)


# ── traffic class 6: vertical port scan ───────────────────────────────────────

class VerticalPortScanThread(GeneratorThread):
    """
    Simulates a vertical (single-host) port scan.

    - One attacker IP
    - One victim IP
    - Sequential destination ports (1, 2, 3, ...)
    - SYN packets (Nmap-style SYN scan)
    """

    ATTACKER_IP = "10.0.0.50"
    VICTIM_IP   = "10.0.1.150"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_port = 1

    def _next_packet(self) -> Optional[Tuple]:
        dport = self._next_port
        self._next_port = (self._next_port % 65535) + 1   # Wrap around
        sport = rand_port_ephemeral()
        return (PROTO_TCP, self.ATTACKER_IP, self.VICTIM_IP, sport, dport, 0, TCP_SYN, 40)


# ── traffic class 7: horizontal port scan ────────────────────────────────────

class HorizontalPortScanThread(GeneratorThread):
    """
    Simulates a horizontal (network sweep) port scan.

    - One attacker IP
    - Many victim IPs across the subnet
    - Fixed destination port (common service discovery)
    - SYN packets
    """

    ATTACKER_IP = "10.0.0.51"
    SCAN_PORT   = 22   # SSH port sweep

    # Subnet prefixes to sweep
    _TARGET_NETS = ["10.0.1.", "10.0.2.", "172.16.0."]

    def _next_packet(self) -> Optional[Tuple]:
        victim_ip = rand_ip(random.choice(self._TARGET_NETS))
        sport     = rand_port_ephemeral()
        return (PROTO_TCP, self.ATTACKER_IP, victim_ip, sport, self.SCAN_PORT, 0, TCP_SYN, 40)


# ── traffic class 8: botnet C2 beaconing ─────────────────────────────────────

class BotnetBeaconThread(GeneratorThread):
    """
    Simulates botnet command-and-control (C2) beaconing.

    - A handful of infected internal hosts (10.0.0.x) periodically "phone home"
      to external C2 servers on suspicious ports (IRC/HTTP-alt/backdoor).
    - Small payloads (20–300 B), bursty: short clusters of packets then idle.
    - Mixed flags (PSH+ACK keep-alives, occasional SYN re-connect).
    - Bidirectional: the C2 replies to the bot.
    This is the canonical 'botnet' class the MLP was trained on.
    """

    _C2_PORTS  = [6667, 8080, 4444, 1080, 31337, 65000]
    _C2_NETS   = ["45.77.", "185.220.", "194.5.", "91.219."]   # external C2 ranges

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bots: List[Dict] = []
        self._burst_remaining = 0

    def _new_bot(self) -> Dict:
        return {
            "bot_ip": rand_ip("10.0.0."),
            "c2_ip":  rand_ip(random.choice(self._C2_NETS)),
            "sport":  rand_port_ephemeral(),
            "dport":  random.choice(self._C2_PORTS),
            "pkts":   0,
        }

    def _next_packet(self) -> Optional[Tuple]:
        if len(self._bots) < 3 or random.random() < 0.02:
            self._bots.append(self._new_bot())
        if len(self._bots) > 8:
            self._bots.pop(0)

        # Bursty beacon: clusters of packets, then idle ticks
        if self._burst_remaining > 0:
            self._burst_remaining -= 1
        elif random.random() < 0.25:
            self._burst_remaining = random.randint(2, 6)
        else:
            return None

        b = random.choice(self._bots)
        b["pkts"] += 1
        plen = random.randint(20, 300)

        # Bot → C2 (request) vs C2 → bot (response)
        if b["pkts"] % 2 == 1:
            src_ip, dst_ip, sport, dport = b["bot_ip"], b["c2_ip"], b["sport"], b["dport"]
        else:
            src_ip, dst_ip, sport, dport = b["c2_ip"], b["bot_ip"], b["dport"], b["sport"]

        flags = TCP_SYN if b["pkts"] == 1 else random.choice(
            [TCP_PSH | TCP_ACK, TCP_ACK, TCP_PSH | TCP_ACK])
        return (PROTO_TCP, src_ip, dst_ip, sport, dport, plen, flags, plen + 40)


# ── traffic class 9: botnet data exfiltration ────────────────────────────────

class BotnetExfilThread(GeneratorThread):
    """
    Simulates a botnet exfiltration / payload-download flow.

    - An infected host pushes larger outbound TCP transfers to a C2 host
      (or pulls a payload), in sustained bursts with full-size segments.
    - Distinguishable from normal web by destination (external C2 port) and
      one-directional, high-volume push pattern.
    """

    _C2_IP   = "185.220.101.45"
    _C2_PORT = 4444

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session = None
        self._remaining = 0

    def _next_packet(self) -> Optional[Tuple]:
        if self._remaining <= 0:
            # Start a new exfil session occasionally
            if random.random() < 0.15:
                self._session = {
                    "bot_ip": rand_ip("10.0.0."),
                    "sport":  rand_port_ephemeral(),
                }
                self._remaining = random.randint(40, 200)   # burst length
            else:
                return None

        self._remaining -= 1
        s = self._session
        plen = random.randint(900, 1400)   # large outbound segments
        flags = TCP_PSH | TCP_ACK
        return (PROTO_TCP, s["bot_ip"], self._C2_IP, s["sport"], self._C2_PORT,
                plen, flags, plen + 40)


# ── statistics printer ────────────────────────────────────────────────────────

class StatsPrinter(threading.Thread):
    """
    Background thread that prints aggregate and per-class packet statistics
    to the console every `interval` seconds.
    """

    def __init__(
        self,
        all_stats: List[ThreadStats],
        stop_event: threading.Event,
        interval: float = 5.0,
    ):
        super().__init__(name="stats_printer", daemon=True)
        self.all_stats  = all_stats
        self.stop_event = stop_event
        self.interval   = interval
        self._global_start = time.monotonic()

    def run(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(self.interval)
            self._print()

    def _print(self) -> None:
        elapsed  = time.monotonic() - self._global_start
        total_tx = sum(s.sent for s in self.all_stats)
        total_er = sum(s.errors for s in self.all_stats)
        avg_pps  = total_tx / elapsed if elapsed > 0 else 0

        border = "─" * 68
        print(f"\n┌{border}┐")
        print(f"│  NIDS Traffic Generator — Live Statistics                        │")
        print(f"├{border}┤")
        print(f"│  Elapsed: {elapsed:6.1f}s   Total sent: {total_tx:8d}   "
              f"Avg PPS: {avg_pps:7.1f}   Errors: {total_er:4d}  │")
        print(f"├{'─'*20}┬{'─'*15}┬{'─'*15}┬{'─'*13}┤")
        print(f"│  {'Thread':<18} │ {'Sent':>13} │ {'PPS':>13} │ {'Errors':>11} │")
        print(f"├{'─'*20}┼{'─'*15}┼{'─'*15}┼{'─'*13}┤")
        for s in self.all_stats:
            with s.lock:
                sent = s.sent
                errs = s.errors
            pps = sent / elapsed if elapsed > 0 else 0
            print(f"│  {s.name:<18} │ {sent:>13,d} │ {pps:>13.1f} │ {errs:>11d} │")
        print(f"└{'─'*20}┴{'─'*15}┴{'─'*15}┴{'─'*13}┘")


# ── orchestrator ─────────────────────────────────────────────────────────────

class TrafficOrchestrator:
    """
    Launches all 7 generator threads concurrently and manages their lifecycle.

    Each traffic class gets its own dedicated thread and socket fd so sends
    never block each other. A shared UDP socket is fine here since sendto()
    is effectively lock-free for datagram sockets on Linux, but we keep one
    socket per thread for clean per-thread socket buffering.

    Thread rates (packets/sec):
        normal_web:    50   — realistic web traffic
        dns_chatter:   30   — bursty DNS
        video_stream: 150   — high-throughput video
        udp_ddos:     300   — attack flood (highest rate)
        syn_flood:    250   — SYN flood
        vert_scan:     80   — sequential port sweep
        horiz_scan:    80   — network sweep
    """

    # (class_name, thread_class, rate_hz)
    # Maps to the 3 model classes:
    #   normal  ← normal_web, dns_chatter, video_stream
    #   ddos    ← udp_ddos, syn_flood
    #   botnet  ← botnet_c2, botnet_exfil
    # The port-scan threads are out-of-distribution traffic (the model has no
    # portscan class) and are kept to show robustness under unseen patterns.
    _CLASSES = [
        ("normal_web",    NormalWebThread,       50),
        ("dns_chatter",   DNSChatterThread,       30),
        ("video_stream",  VideoStreamThread,     150),
        ("udp_ddos",      UDPDDoSThread,         300),
        ("syn_flood",     SYNFloodThread,        250),
        ("botnet_c2",     BotnetBeaconThread,     70),
        ("botnet_exfil",  BotnetExfilThread,      50),
        ("vert_scan",     VerticalPortScanThread, 60),
        ("horiz_scan",    HorizontalPortScanThread, 60),
    ]

    def __init__(self, target_ip: str, target_port: int, rate_scale: float = 1.0):
        self.target_ip   = target_ip
        self.target_port = target_port
        self.rate_scale  = rate_scale

        self._stop_event  = threading.Event()
        self._threads:     List[GeneratorThread] = []
        self._all_stats:   List[ThreadStats]     = []
        self._stats_printer: Optional[StatsPrinter] = None

    def start(self) -> None:
        """Instantiate and start all generator threads plus the stats printer."""
        print(f"[orchestrator] Starting NIDS traffic generator")
        print(f"[orchestrator] Target: {self.target_ip}:{self.target_port}")
        print(f"[orchestrator] Rate scale: {self.rate_scale}x")
        print(f"[orchestrator] Launching {len(self._CLASSES)} concurrent traffic threads...\n")

        for name, cls, base_rate in self._CLASSES:
            stats  = ThreadStats(name=name)
            sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)

            t = cls(
                name       = name,
                target_ip  = self.target_ip,
                target_port= self.target_port,
                rate_hz    = base_rate * self.rate_scale,
                sock       = sock,
                stop_event = self._stop_event,
                stats      = stats,
            )
            t.start()
            self._threads.append(t)
            self._all_stats.append(stats)
            print(f"  ✓ {name:<22} started  (target {base_rate * self.rate_scale:.0f} pkt/s)")

        self._stats_printer = StatsPrinter(
            all_stats  = self._all_stats,
            stop_event = self._stop_event,
            interval   = 5.0,
        )
        self._stats_printer.start()
        print(f"\n[orchestrator] All threads running. Press Ctrl-C to stop.\n")

    def wait(self) -> None:
        """Block until stop_event is set (e.g. by signal handler)."""
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self) -> None:
        """Signal all threads to stop and wait for them to finish."""
        print("\n[orchestrator] Stopping all threads...")
        self._stop_event.set()

        for t in self._threads:
            t.join(timeout=3.0)

        # Print final statistics
        if self._stats_printer:
            self._stats_printer._print()

        total = sum(s.sent for s in self._all_stats)
        print(f"\n[orchestrator] Shutdown complete. Total packets sent: {total:,}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FPGA-NIDS Advanced Concurrent Traffic Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target_ip", default="192.168.2.99",
        help="PYNQ board IP address",
    )
    parser.add_argument(
        "--target_port", type=int, default=5555,
        help="PYNQ UDP receiver port",
    )
    parser.add_argument(
        "--rate_scale", type=float, default=1.0,
        help="Global rate multiplier (e.g. 0.5 for half speed, 2.0 for double)",
    )
    args = parser.parse_args()

    orch = TrafficOrchestrator(
        target_ip   = args.target_ip,
        target_port = args.target_port,
        rate_scale  = args.rate_scale,
    )

    # Register SIGINT/SIGTERM for graceful shutdown
    def _signal_handler(sig, frame):
        print(f"\n[orchestrator] Signal {sig} received — shutting down...")
        orch.stop()
        os._exit(0)

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    orch.start()
    orch.wait()


if __name__ == "__main__":
    main()