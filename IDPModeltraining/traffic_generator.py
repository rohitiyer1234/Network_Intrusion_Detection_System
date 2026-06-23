#!/usr/bin/env python3
"""
traffic_generator.py  —  Laptop-side real-time UDP traffic generator.

Simulates realistic network traffic patterns as UDP datagrams sent to the
PYNQ receiver.  Each datagram encodes a synthetic TCP/IP packet so the PYNQ
can construct bidirectional flows and extract CICIDS-compatible features.

Wire format (60 bytes, big-endian):
    Offset  Len   Field
    0       1     Protocol  (6=TCP, 17=UDP)
    1       4     Src IP (packed)
    5       4     Dst IP (packed)
    9       2     Src port
    11      2     Dst port
    13      2     Payload length
    15      1     TCP flags
    16      2     IP total length
    18      8     Timestamp ns
    26      34    Padding

Usage:
    python3 traffic_generator.py --target_ip 192.168.2.99 --target_port 5555 \\
        --mode mixed --rate 200

Requirements:
    pip install  (none – stdlib only)

Run as a normal user; no root required.
"""

import argparse
import random
import socket
import struct
import time
import threading
from typing import Dict, Callable, Tuple

# ── wire format ───────────────────────────────────────────────────────────────

FRAME_SIZE = 60   # bytes per datagram

TCP_FLAG_SYN = 0x02
TCP_FLAG_ACK = 0x10
TCP_FLAG_FIN = 0x01
TCP_FLAG_RST = 0x04
TCP_FLAG_PSH = 0x08

PROTO_TCP = 6
PROTO_UDP = 17

# Simulated internal subnets
_PRIVATE_NETS = [
    "10.0.0.",
    "10.0.1.",
    "192.168.1.",
    "172.16.0.",
]


def _rand_ip(net_prefix: str = None) -> str:
    if net_prefix is None:
        net_prefix = random.choice(_PRIVATE_NETS)
    return net_prefix + str(random.randint(2, 254))


def _pack_frame(
    proto: int,
    src_ip: str, dst_ip: str,
    src_port: int, dst_port: int,
    payload_len: int,
    tcp_flags: int,
    ip_total_len: int,
    ts_ns: int,
) -> bytes:
    src_packed = socket.inet_aton(src_ip)
    dst_packed = socket.inet_aton(dst_ip)
    header = struct.pack(
        "!B4s4sHHHBHQ",
        proto,
        src_packed,
        dst_packed,
        src_port,
        dst_port,
        payload_len,
        tcp_flags,
        ip_total_len,
        ts_ns,
    )
    padding = bytes(FRAME_SIZE - len(header))
    return header + padding


# ── traffic patterns ──────────────────────────────────────────────────────────

class _Session:
    """Tracks a simulated TCP session so ACK/PSH flags are realistic."""
    def __init__(self, src_ip, dst_ip, sport, dport):
        self.src_ip  = src_ip
        self.dst_ip  = dst_ip
        self.sport   = sport
        self.dport   = dport
        self.packets = 0
        self.done    = False


def _gen_normal(sessions: list) -> Tuple:
    """HTTP/HTTPS-like web browsing: moderate rate, PSH-heavy, varied sizes."""
    if not sessions or random.random() < 0.05:
        s = _Session(
            src_ip=_rand_ip("10.0.0."),
            dst_ip=_rand_ip("10.0.1."),
            sport=random.randint(49152, 65535),
            dport=random.choice([80, 443, 8080]),
        )
        sessions.append(s)
    s = random.choice(sessions)
    s.packets += 1
    # Alternate request / response direction
    if s.packets % 2 == 0:
        src_ip, dst_ip = s.dst_ip, s.src_ip
        sport, dport   = s.dport, s.sport
    else:
        src_ip, dst_ip = s.src_ip, s.dst_ip
        sport, dport   = s.sport, s.dport
    flags = TCP_FLAG_ACK | TCP_FLAG_PSH
    if s.packets == 1:
        flags = TCP_FLAG_SYN
    plen = random.randint(40, 1400)
    if s.packets > random.randint(10, 40):
        flags = TCP_FLAG_FIN | TCP_FLAG_ACK
        sessions.remove(s)
    return (PROTO_TCP, src_ip, dst_ip, sport, dport, plen, flags, plen + 40)


def _gen_ddos(_sessions: list) -> Tuple:
    """DDoS: high-rate SYN flood toward a single victim, many sources."""
    victim_ip   = "10.0.1.100"
    victim_port = 80
    src_ip   = _rand_ip()
    src_port = random.randint(1024, 65535)
    flags    = TCP_FLAG_SYN
    plen     = 40   # SYN has no payload
    return (PROTO_TCP, src_ip, victim_ip, src_port, victim_port, plen, flags, 40)


def _gen_botnet(sessions: list) -> Tuple:
    """Botnet: periodic short bursts to random C2 IPs, mixed flags."""
    c2_ip    = _rand_ip("172.16.0.")
    bot_ip   = _rand_ip("10.0.0.")
    sport    = random.randint(49152, 65535)
    dport    = random.choice([6667, 1080, 4444, 31337, 65000])
    flags    = random.choice([TCP_FLAG_ACK, TCP_FLAG_PSH | TCP_FLAG_ACK,
                               TCP_FLAG_RST, TCP_FLAG_SYN])
    plen     = random.randint(20, 200)
    return (PROTO_TCP, bot_ip, c2_ip, sport, dport, plen, flags, plen + 40)


def _gen_bulk(sessions: list) -> Tuple:
    """Bulk transfer: large FTP/SMB-like flows, high byte count per flow."""
    if not sessions or random.random() < 0.02:
        s = _Session(
            src_ip=_rand_ip("10.0.0."),
            dst_ip=_rand_ip("10.0.1."),
            sport=random.randint(49152, 65535),
            dport=random.choice([21, 445, 2049, 8888]),
        )
        sessions.append(s)
    s = random.choice(sessions)
    s.packets += 1
    flags = TCP_FLAG_ACK | TCP_FLAG_PSH
    plen  = 1460   # full MSS
    if s.packets > random.randint(200, 1000):
        flags = TCP_FLAG_FIN | TCP_FLAG_ACK
        sessions.remove(s)
    return (PROTO_TCP, s.src_ip, s.dst_ip, s.sport, s.dport, plen, flags, plen + 40)


_GENERATORS: Dict[str, Callable] = {
    "normal":   _gen_normal,
    "ddos":     _gen_ddos,
    "botnet":   _gen_botnet,
    "bulk":     _gen_bulk,
}

# Mixed mode weights (roughly matches real-world distributions)
_MIXED_MODES = ["normal", "bulk", "ddos", "botnet"]
_MIXED_WEIGHTS = [0.50, 0.25, 0.15, 0.10]


# ── sender ────────────────────────────────────────────────────────────────────

class TrafficGenerator:
    def __init__(self, target_ip: str, target_port: int,
                 mode: str, rate: float):
        self.target_ip   = target_ip
        self.target_port = target_port
        self.mode        = mode
        self.rate        = rate

        self._sock      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sessions: Dict[str, list] = {m: [] for m in _GENERATORS}
        self._stop      = threading.Event()
        self._sent      = 0
        self._start_ts  = 0.0

    def _choose_generator(self) -> Tuple[Callable, list]:
        if self.mode == "mixed":
            chosen = random.choices(_MIXED_MODES, weights=_MIXED_WEIGHTS, k=1)[0]
        else:
            chosen = self.mode
        return _GENERATORS[chosen], self._sessions[chosen]

    def run(self) -> None:
        interval = 1.0 / self.rate if self.rate > 0 else 0.0
        self._start_ts = time.monotonic()
        print(f"[generator] Sending to {self.target_ip}:{self.target_port}  "
              f"mode={self.mode}  rate={self.rate} pkt/s")
        print("[generator] Press Ctrl-C to stop.")

        next_send = time.monotonic()
        stats_interval = 5.0
        last_stats = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_send:
                time.sleep(min(next_send - now, 0.001))
                continue

            gen_fn, sessions = self._choose_generator()
            try:
                fields = gen_fn(sessions)
            except Exception:
                continue

            ts_ns = time.time_ns()
            frame = _pack_frame(*fields, ts_ns=ts_ns)
            try:
                self._sock.sendto(frame, (self.target_ip, self.target_port))
                self._sent += 1
            except OSError as exc:
                print(f"[generator] Send error: {exc}")

            next_send += interval

            if now - last_stats >= stats_interval:
                elapsed = now - self._start_ts
                pps = self._sent / elapsed if elapsed > 0 else 0
                print(f"[generator] sent={self._sent}  avg={pps:.0f} pkt/s")
                last_stats = now

    def stop(self) -> None:
        self._stop.set()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Real-time UDP traffic generator for PYNQ classifier")
    p.add_argument("--target_ip",   required=True,           help="PYNQ board IP address")
    p.add_argument("--target_port", type=int, default=5555,  help="Receiver UDP port (default 5555)")
    p.add_argument("--mode",        default="mixed",
                   choices=["normal", "ddos", "botnet", "bulk", "mixed"],
                   help="Traffic mode (default: mixed)")
    p.add_argument("--rate",        type=float, default=200, help="Packets per second (default 200)")
    args = p.parse_args()

    gen = TrafficGenerator(args.target_ip, args.target_port, args.mode, args.rate)
    try:
        gen.run()
    except KeyboardInterrupt:
        gen.stop()
        print(f"\n[generator] Stopped. Total sent: {gen._sent}")


if __name__ == "__main__":
    main()
