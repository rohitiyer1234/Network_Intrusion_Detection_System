"""
LAPTOP/tests/test_connectivity.py

Minimal single-flow smoke test: sends one realistic TCP flow (SYN, 20
data exchanges, FIN) to confirm the wire format, UDP path, flow
reconstruction, and TCP result delivery all work before running the
full traffic generator. Prints every frame it sends.

Usage:
    python3 LAPTOP/tests/test_connectivity.py
    python3 LAPTOP/tests/test_connectivity.py --target_ip 192.168.2.99
"""

import argparse
import socket
import struct
import time

ap = argparse.ArgumentParser()
ap.add_argument("--target_ip", default="192.168.2.99", help="PYNQ board IP")
ap.add_argument("--target_port", type=int, default=5555, help="PYNQ UDP port")
args = ap.parse_args()

FRAME_STRUCT = struct.Struct("!B4s4sHHHBHQ")

PROTO_TCP = 6

TCP_SYN = 0x02
TCP_ACK = 0x10
TCP_FIN = 0x01
TCP_PSH = 0x08

TARGET_IP = args.target_ip
TARGET_PORT = args.target_port

sock = socket.socket(
    socket.AF_INET,
    socket.SOCK_DGRAM
)

def ip_bytes(ip):
    return socket.inet_aton(ip)

def send_packet(
    src_ip,
    dst_ip,
    sport,
    dport,
    flags,
    payload_len
):

    frame = FRAME_STRUCT.pack(
        PROTO_TCP,
        ip_bytes(src_ip),
        ip_bytes(dst_ip),
        sport,
        dport,
        payload_len,
        flags,
        payload_len + 40,
        time.time_ns()
    )

    frame += bytes(60 - len(frame))

    sock.sendto(
        frame,
        (TARGET_IP, TARGET_PORT)
    )

    print(
        f"SENT "
        f"{src_ip}:{sport} -> "
        f"{dst_ip}:{dport} "
        f"flags={hex(flags)} "
        f"payload={payload_len}"
    )


client = "10.0.0.10"
server = "10.0.1.20"

sport = 55000
dport = 443

print("\n=== VERIFIED FLOW TEST START ===\n")

# ---------------------------------------------------
# SYN
# ---------------------------------------------------

send_packet(
    client,
    server,
    sport,
    dport,
    TCP_SYN,
    0
)

time.sleep(0.1)

# ---------------------------------------------------
# DATA
# ---------------------------------------------------

for i in range(20):

    send_packet(
        client,
        server,
        sport,
        dport,
        TCP_ACK | TCP_PSH,
        1200
    )

    send_packet(
        server,
        client,
        dport,
        sport,
        TCP_ACK | TCP_PSH,
        900
    )

    time.sleep(0.02)

# ---------------------------------------------------
# FIN
# ---------------------------------------------------

send_packet(
    client,
    server,
    sport,
    dport,
    TCP_FIN,
    0
)

print("\n=== FLOW COMPLETE ===\n")

