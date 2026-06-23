#!/usr/bin/env python3
"""
LAPTOP/tests/run_classification_demo.py  —  FIXED

Sends traffic whose flow-level feature vectors land in the correct regions
of the trained model's decision space.

Root-cause of the original failure (both bugs fixed here):
─────────────────────────────────────────────────────────
1. BUFFERING / LATENCY
   The original demo sent packets with time.sleep(0.002) between them, which
   spreads 3000 DDoS packets over 6 full seconds.  The flow engine (timeout=1s)
   cannot complete a flow until it goes idle, so the dashboard lags by many
   seconds.  Fix: send each burst tightly (sub-millisecond sleeps) so flows
   complete quickly via TCP RST, then pause between bursts.

2. ONLY "NORMAL" CLASSIFIED  — the real bug
   The model was trained on CSE-CIC-IDS2018 flows.  Its feature_max bounds are:
       total_len_fwd_pkts  =  22 000 bytes
       flow_bytes_per_sec  =  18 000 000 bps
       total_fwd_packets   =  170 packets

   DDoS signature (confirmed by probing the model weights):
       ~100 fwd packets × ~220 bytes each = 22 kB total
       completed in << 100 ms  →  flow_bytes_per_sec ≈ 220 kbps–22 Mbps
       unidirectional (bwd_packets ≈ 0)
       All other features near zero → model gives >99% DDoS
   The original demo sent each DDoS packet from a DIFFERENT random src_ip.
   The flow engine keys flows on the full 5-tuple, so each packet became its
   own 1-packet flow with duration=0, bps=0 → classified as normal.
   Fix: use a fixed small set of bot IPs with fixed ports so the flow engine
   accumulates ALL packets into ONE flow per bot, completing via TCP RST.

   Botnet signature (confirmed by probing the model weights):
       ~30 fwd packets × ~20 bytes each (very small C2 keep-alives)
       PSH flag set on EVERY data packet  ← primary discriminator
       ~2 bwd packets (server ACK/response)
       Duration ~5 s  →  flow_iat_mean ≈ 160 ms (periodic beaconing)
       avg_fwd_segment_size ≈ 20 bytes  ← secondary discriminator
   The original demo used payload sizes of 20–280 bytes (too large) and did
   not set PSH on every packet, so avg_fwd_segment_size was too high and
   fwd_psh_flags was <<1 → classified as normal.
   Fix: force payload=20 bytes and PSH flag on every packet.

Usage:
    python3 LAPTOP/tests/run_classification_demo.py
    python3 LAPTOP/tests/run_classification_demo.py --target_ip 192.168.2.99 --target_port 5555
"""

import argparse
import socket
import struct
import time
import random

# ── CLI ───────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser(
    description="FPGA-NIDS Classification Demo  (normal / ddos / botnet)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
ap.add_argument("--target_ip",   default="192.168.2.99", help="PYNQ board IP")
ap.add_argument("--target_port", type=int, default=5555,  help="PYNQ UDP port")
ap.add_argument("--flow_timeout", type=float, default=1.0,
                help="Must match --flow_timeout used on the PYNQ receiver")
args = ap.parse_args()

TARGET = (args.target_ip, args.target_port)

# ── Wire format (matches receiver._unpack_frame exactly) ──────────────────────
#   !B4s4sHHHBHQ  = proto(1) src_ip(4) dst_ip(4) sport(2) dport(2)
#                   plen(2)  flags(1)  ip_total(2) ts_ns(8)  = 26 bytes
#   Padded to 60 bytes total.

FRAME_STRUCT = struct.Struct("!B4s4sHHHBHQ")
FRAME_PAD    = bytes(60 - FRAME_STRUCT.size)   # 34 zero bytes

PROTO_TCP = 6
PROTO_UDP = 17

# TCP flag constants
TCP_SYN = 0x02
TCP_ACK = 0x10
TCP_FIN = 0x01
TCP_RST = 0x04
TCP_PSH = 0x08

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)


def _ip(dotted: str) -> bytes:
    return socket.inet_aton(dotted)


def send(proto, src_ip, dst_ip, sport, dport, plen, flags):
    frame = FRAME_STRUCT.pack(
        proto, _ip(src_ip), _ip(dst_ip),
        sport, dport, plen, flags,
        plen + 40, time.time_ns(),
    ) + FRAME_PAD
    sock.sendto(frame, TARGET)


def close_flow_rst(proto, src_ip, dst_ip, sport, dport):
    """Send TCP RST to immediately complete the flow in the flow engine."""
    send(proto, src_ip, dst_ip, sport, dport, 0, TCP_RST)


# ═══════════════════════════════════════════════════════════════════════════════
# NORMAL: bidirectional web/HTTPS traffic
#
# Feature targets:
#   fwd_packets ≈ 20, bwd_packets ≈ 15, payload 400–1400 B
#   PSH set on data packets, bidirectional, flow_duration ≈ 2–4 s
#   → model gives ≥ 99 % normal
# ═══════════════════════════════════════════════════════════════════════════════

def run_normal(n_flows: int = 5):
    print("\n[DEMO] ── NORMAL flows ──────────────────────────────────────────")
    for i in range(n_flows):
        client = f"10.0.0.{random.randint(10, 50)}"
        server = f"10.0.1.{random.randint(10, 50)}"
        sport  = random.randint(49152, 65535)
        dport  = random.choice([80, 443, 8080])

        # TCP SYN (opens flow)
        send(PROTO_TCP, client, server, sport, dport, 0, TCP_SYN)
        time.sleep(0.01)

        # Bidirectional data exchange — 20 fwd + 15 bwd packets, ~800 B each
        # Spread over 2–3 s to produce realistic IAT and flow duration
        n_data = 18
        interval = 2.0 / n_data
        for p in range(n_data):
            plen = random.randint(400, 1200)
            send(PROTO_TCP, client, server, sport, dport, plen, TCP_ACK | TCP_PSH)
            if p % 3 == 0:
                rlen = random.randint(400, 1200)
                send(PROTO_TCP, server, client, dport, sport, rlen, TCP_ACK | TCP_PSH)
            time.sleep(interval)

        # TCP FIN (completes flow immediately — no need to wait for timeout)
        send(PROTO_TCP, client, server, sport, dport, 0, TCP_FIN | TCP_ACK)

        print(f"  [normal] flow {i+1}/{n_flows}  {client}:{sport} → {server}:{dport}")
        time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# DDoS: compact high-rate UDP bursts, FIXED src IPs
#
# Feature targets (reverse-engineered from model_fp.npz weights):
#   fwd_packets ≈ 100,  total_len_fwd_pkts ≈ 22 000 B (feat_max)
#   bwd_packets = 0  (unidirectional flood)
#   flow_duration ≈ 0.005–0.05 s  →  bps ≈ 440 kbps – 4.4 Mbps  → >99 % DDoS
#   fwd_pkt_len_max = fwd_pkt_len_min = 220 B (uniform size)
#
# Critical fix: use FIXED (src_ip, dst_ip, sport, dport) tuples so the flow
# engine accumulates ALL 100 packets into ONE flow instead of 100 × 1-pkt flows.
# Each burst is sent in <50 ms, then closed with TCP RST (for immediate eviction).
# ═══════════════════════════════════════════════════════════════════════════════

# A small fixed set of bot IPs — each bot uses a fixed 5-tuple per burst
_DDOS_BOTS = [
    ("10.10.0.11", 10011),
    ("10.10.0.22", 10022),
    ("10.10.0.33", 10033),
    ("10.10.0.44", 10044),
    ("10.10.0.55", 10055),
]
_DDOS_VICTIM      = "10.0.1.200"
_DDOS_VICTIM_PORT = 80
_DDOS_PKT_SIZE    = 220     # bytes — gives total_len_fwd = 22 000 at 100 packets
_DDOS_N_PKTS      = 100     # packets per burst — stays within feat_max=170... wait
#   feat_max total_fwd_packets = 170.  We send 100 → well within range.
#   feat_max total_len_fwd_pkts = 22 000.  100 × 220 = 22 000 → exactly at cap.


def run_ddos(n_bursts: int = 20):
    print("\n[DEMO] ── DDoS bursts ──────────────────────────────────────────")
    for i in range(n_bursts):
        src_ip, sport = _DDOS_BOTS[i % len(_DDOS_BOTS)]

        # Send 100 UDP packets as fast as possible (< 50 ms total)
        # UDP has no FIN/RST, so we "trick" the flow engine by using TCP RST at the end.
        # To make the RST work: send the burst as TCP (SYN omitted — just data + RST)
        # The flow engine treats the first packet as a new flow regardless of flags.
        for p in range(_DDOS_N_PKTS):
            # TCP ACK|PSH so these look like data packets (not SYN flood)
            send(PROTO_TCP, src_ip, _DDOS_VICTIM, sport, _DDOS_VICTIM_PORT,
                 _DDOS_PKT_SIZE, TCP_ACK | TCP_PSH)

        # RST closes the flow immediately — features computed right away
        close_flow_rst(PROTO_TCP, src_ip, _DDOS_VICTIM, sport, _DDOS_VICTIM_PORT)

        if i % 5 == 0:
            print(f"  [ddos]   burst {i+1}/{n_bursts}  {src_ip}:{sport} → "
                  f"{_DDOS_VICTIM}:{_DDOS_VICTIM_PORT}  ({_DDOS_N_PKTS} pkts × {_DDOS_PKT_SIZE} B)")

        # Tiny gap between bursts — keeps the live dashboard updating
        time.sleep(0.08)


# ═══════════════════════════════════════════════════════════════════════════════
# Botnet: C2 beaconing with tiny PSH-flagged payloads
#
# Feature targets (reverse-engineered from model_fp.npz weights):
#   fwd_packets ≈ 30,  avg_fwd_segment_size ≈ 20 B  ← key discriminators
#   fwd_psh_flags = 1.0  (PSH on EVERY fwd data packet)  ← primary discriminator
#   bwd_packets ≈ 2  (C2 server sends a short ACK response)
#   flow_duration ≈ 5 s  →  flow_iat_mean ≈ 160 ms  (periodic beaconing)
#   total_len_fwd_pkts = 30 × 20 = 600 B  (well within feat_max = 22 000)
#
# The original demo used payloads of 20–280 B (avg too high) and set PSH
# inconsistently, pushing avg_fwd_segment_size and psh_flag ratio out of the
# botnet zone.  Every single fwd packet here is exactly 20 bytes with PSH set.
# ═══════════════════════════════════════════════════════════════════════════════

_BOT_HOSTS = [
    ("10.0.0.31", "45.77.10.12",      6667),
    ("10.0.0.57", "185.220.101.45",   4444),
    ("10.0.0.88", "91.219.7.30",     31337),
]
_BOT_PKT_SIZE    = 20    # bytes — avg_fwd_segment_size target
_BOT_FWD_PKTS    = 30    # fwd packets per flow
_BOT_BWD_PAYLOAD = 200   # bytes for the C2 "response"
_BOT_IAT_S       = 0.16  # 160 ms between beacons  →  5s / 30 pkts


def run_botnet(n_flows: int = 10):
    print("\n[DEMO] ── Botnet C2 beaconing ──────────────────────────────────")
    for i in range(n_flows):
        bot_ip, c2_ip, c2_port = _BOT_HOSTS[i % len(_BOT_HOSTS)]
        sport = random.randint(49152, 65530)

        # SYN to open flow
        send(PROTO_TCP, bot_ip, c2_ip, sport, c2_port, 0, TCP_SYN)
        time.sleep(0.01)

        # Beacon loop: 30 tiny PSH data packets, 160 ms apart
        for p in range(_BOT_FWD_PKTS):
            # PSH set on EVERY fwd data packet — this is the key discriminator
            send(PROTO_TCP, bot_ip, c2_ip, sport, c2_port,
                 _BOT_PKT_SIZE, TCP_ACK | TCP_PSH)

            # C2 server sends a short ACK/response every ~15 beacons
            if p in (7, 22):
                send(PROTO_TCP, c2_ip, bot_ip, c2_port, sport,
                     _BOT_BWD_PAYLOAD, TCP_ACK | TCP_PSH)

            time.sleep(_BOT_IAT_S)

        # RST closes the flow → features extracted immediately
        close_flow_rst(PROTO_TCP, bot_ip, c2_ip, sport, c2_port)

        print(f"  [botnet] flow {i+1}/{n_flows}  {bot_ip} → {c2_ip}:{c2_port}  "
              f"({_BOT_FWD_PKTS} pkts × {_BOT_PKT_SIZE} B PSH, "
              f"IAT={_BOT_IAT_S*1000:.0f} ms)")
        time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# Mixed finale — all three classes interleaved
# ═══════════════════════════════════════════════════════════════════════════════

def run_mixed(n_rounds: int = 15):
    print("\n[DEMO] ── Mixed finale ──────────────────────────────────────────")
    for i in range(n_rounds):
        r = random.random()

        if r < 0.34:
            # Normal: quick 8-packet web exchange, completed via FIN
            client = f"10.0.0.{random.randint(10,50)}"
            server = f"10.0.1.{random.randint(10,50)}"
            sp     = random.randint(49152, 65535)
            dp     = random.choice([80, 443])
            send(PROTO_TCP, client, server, sp, dp, 0, TCP_SYN)
            for _ in range(7):
                send(PROTO_TCP, client, server, sp, dp,
                     random.randint(400, 1200), TCP_ACK | TCP_PSH)
                time.sleep(0.12)
            send(PROTO_TCP, server, client, dp, sp,
                 random.randint(400, 1200), TCP_ACK | TCP_PSH)
            send(PROTO_TCP, client, server, sp, dp, 0, TCP_FIN | TCP_ACK)
            print(f"  [mixed #{i+1:2d}] NORMAL  {client} → {server}:{dp}")

        elif r < 0.67:
            # DDoS: 100-packet burst, RST to close immediately
            src_ip, sport = _DDOS_BOTS[i % len(_DDOS_BOTS)]
            for _ in range(_DDOS_N_PKTS):
                send(PROTO_TCP, src_ip, _DDOS_VICTIM, sport,
                     _DDOS_VICTIM_PORT, _DDOS_PKT_SIZE, TCP_ACK | TCP_PSH)
            close_flow_rst(PROTO_TCP, src_ip, _DDOS_VICTIM, sport, _DDOS_VICTIM_PORT)
            time.sleep(0.08)
            print(f"  [mixed #{i+1:2d}] DDOS    {src_ip} → {_DDOS_VICTIM}:{_DDOS_VICTIM_PORT}")

        else:
            # Botnet: 10 PSH beacons (abbreviated), RST to close
            bot_ip, c2_ip, c2_port = _BOT_HOSTS[i % len(_BOT_HOSTS)]
            sp = random.randint(49152, 65530)
            send(PROTO_TCP, bot_ip, c2_ip, sp, c2_port, 0, TCP_SYN)
            for p in range(10):
                send(PROTO_TCP, bot_ip, c2_ip, sp, c2_port,
                     _BOT_PKT_SIZE, TCP_ACK | TCP_PSH)
                time.sleep(_BOT_IAT_S)
            send(PROTO_TCP, c2_ip, bot_ip, c2_port, sp, _BOT_BWD_PAYLOAD, TCP_ACK)
            close_flow_rst(PROTO_TCP, bot_ip, c2_ip, sp, c2_port)
            print(f"  [mixed #{i+1:2d}] BOTNET  {bot_ip} → {c2_ip}:{c2_port}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  NIDS DEMO  —  normal / ddos / botnet")
print(f"  Target: {args.target_ip}:{args.target_port}")
print("=" * 60)
print("""
Traffic signatures used (verified against model_fp.npz):
  NORMAL  : TCP, bidirectional, 400–1200 B payloads, PSH+ACK
  DDOS    : TCP, 100 pkts × 220 B, same 5-tuple, RST to close,
            total_len_fwd_pkts=22 000 B (feat_max), bps≈440 kbps–22 Mbps
  BOTNET  : TCP, 30 pkts × 20 B, PSH on EVERY packet (key!),
            IAT=160 ms (beaconing), 2 bwd ACKs, RST to close
""")

time.sleep(2)

#run_normal(n_flows=4)
#print("\n[DEMO] Waiting for flows to propagate through PYNQ pipeline ...")
#time.sleep(args.flow_timeout + 1.0)

#run_ddos(n_bursts=15)
#print("\n[DEMO] Waiting for flows to propagate through PYNQ pipeline ...")
#time.sleep(args.flow_timeout + 1.0)

run_botnet(n_flows=6)
print("\n[DEMO] Waiting for flows to propagate through PYNQ pipeline ...")
time.sleep(args.flow_timeout + 1.0)

run_mixed(n_rounds=12)
print("\n[DEMO] Waiting for final flows to propagate ...")
time.sleep(args.flow_timeout + 2.0)

print("\n[DEMO] All phases complete.\n")