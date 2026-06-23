"""
LAPTOP/tests/run_classification_demo.py

Single-script classification demo: sends a clean Normal flow, then a
DDoS burst, then Botnet C2 beaconing, then a mixed finale — sequentially
with pauses — so the dashboard clearly shows each of the 3 model classes
(normal / ddos / botnet) lighting up in turn.

This is the recommended "smallest possible command" demo traffic source
once the pipeline is already running (receiver.py + dashboard_server.py).

Usage:
    python3 LAPTOP/tests/run_classification_demo.py
    python3 LAPTOP/tests/run_classification_demo.py --target_ip 192.168.2.99
"""

import argparse
import socket
import time
import struct
import random

ap = argparse.ArgumentParser()
ap.add_argument("--target_ip", default="192.168.2.99", help="PYNQ board IP")
ap.add_argument("--target_port", type=int, default=5555, help="PYNQ UDP port")
args = ap.parse_args()

FRAME_STRUCT = struct.Struct("!B4s4sHHHBHQ")

PROTO_TCP = 6
PROTO_UDP = 17

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

# ============================================================
# SEND FRAME
# ============================================================

def send_frame(
    proto,
    src_ip,
    dst_ip,
    sport,
    dport,
    payload_len,
    flags
):

    frame = FRAME_STRUCT.pack(
        proto,
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

# ============================================================
# NORMAL FLOW
# ============================================================

def run_normal_test():

    print("\n=== NORMAL FLOW TEST ===\n")

    client = "10.0.0.10"
    server = "10.0.1.20"

    sport = 55000
    dport = 443

    # SYN
    send_frame(
        PROTO_TCP,
        client,
        server,
        sport,
        dport,
        0,
        TCP_SYN
    )

    time.sleep(0.01)

    # Data packets
    for _ in range(40):

        send_frame(
            PROTO_TCP,
            client,
            server,
            sport,
            dport,
            random.randint(500,1400),
            TCP_ACK | TCP_PSH
        )

        send_frame(
            PROTO_TCP,
            server,
            client,
            dport,
            sport,
            random.randint(500,1400),
            TCP_ACK | TCP_PSH
        )

        time.sleep(0.01)

    # FIN
    send_frame(
        PROTO_TCP,
        client,
        server,
        sport,
        dport,
        0,
        TCP_FIN
    )

    print("[TEST] Normal flow completed")

# ============================================================
# DDOS TEST
# ============================================================

def run_ddos_test():

    print("\n=== DDOS TEST ===\n")

    victim = "10.0.1.200"

    bot_ips = [
        f"10.10.0.{i}"
        for i in range(2,40)
    ]

    for i in range(3000):

        src = random.choice(bot_ips)

        send_frame(
            PROTO_UDP,
            src,
            victim,
            random.randint(1024,65535),
            80,
            random.randint(1000,1400),
            0
        )

        if i % 500 == 0:

            print(
                f"[DDOS] packets={i}"
            )

        time.sleep(0.002)

    print("[TEST] DDOS completed")

# ============================================================
# BOTNET TEST
# ============================================================

def run_botnet_test():

    print("\n=== BOTNET (C2 BEACONING) TEST ===\n")

    # A few infected hosts beaconing to external C2 servers on suspicious
    # ports, with small bidirectional payloads in periodic bursts.
    bots = ["10.0.0.31", "10.0.0.57", "10.0.0.88"]
    c2_servers = [("45.77.10.12", 6667),
                  ("185.220.101.45", 4444),
                  ("91.219.7.30", 31337)]

    for beacon in range(60):

        bot = random.choice(bots)
        c2_ip, c2_port = random.choice(c2_servers)
        sport = random.randint(49152, 65535)

        # short burst of keep-alives, bot <-> C2
        for p in range(random.randint(3, 7)):
            small = random.randint(20, 280)
            if p == 0:
                send_frame(PROTO_TCP, bot, c2_ip, sport, c2_port, 0, TCP_SYN)
            elif p % 2:
                send_frame(PROTO_TCP, bot, c2_ip, sport, c2_port, small,
                           TCP_ACK | TCP_PSH)
            else:
                send_frame(PROTO_TCP, c2_ip, bot, c2_port, sport, small,
                           TCP_ACK | TCP_PSH)
            time.sleep(0.01)

        if beacon % 15 == 0:
            print(f"[BOTNET] beacons={beacon}")

        time.sleep(0.05)   # idle gap between beacons

    print("[TEST] Botnet completed")

# ============================================================
# MIXED FINALE
# ============================================================

def run_mixed_finale():

    print("\n=== MIXED FINALE (all 3 classes interleaved) ===\n")

    victim = "10.0.1.200"
    for i in range(400):
        r = random.random()
        if r < 0.45:   # normal
            send_frame(PROTO_TCP, "10.0.0.10", "10.0.1.20",
                       55000, 443, random.randint(400, 1400), TCP_ACK | TCP_PSH)
        elif r < 0.75:  # ddos
            send_frame(PROTO_UDP, f"10.10.0.{random.randint(2,40)}", victim,
                       random.randint(1024, 65535), 80, random.randint(1000, 1400), 0)
        else:           # botnet
            send_frame(PROTO_TCP, "10.0.0.57", "185.220.101.45",
                       random.randint(49152, 65535), 4444,
                       random.randint(20, 280), TCP_ACK | TCP_PSH)
        time.sleep(0.005)

    print("[TEST] Mixed finale completed")

# ============================================================
# MAIN
# ============================================================

print("\n================================================")
print("NIDS ARCHITECTURAL VALIDATION  (normal / ddos / botnet)")
print("================================================")

time.sleep(2)

run_normal_test()

time.sleep(3)

run_ddos_test()

time.sleep(3)

run_botnet_test()

time.sleep(3)

run_mixed_finale()

print("\nAll architectural tests completed.\n")