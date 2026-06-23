"""
capture/capture.py

Python packet capture using AF_PACKET raw sockets.

Runs in its own thread and pushes ParsedPacket namedtuples onto a
thread-safe queue consumed by the flow construction engine.

Why AF_PACKET instead of Scapy?
  - Scapy's sniff() cannot be cleanly stopped from another thread.
  - AF_PACKET + socket.recv() gives nanosecond-stamped frames with no
    external dependency, and works identically on x86 Ubuntu and PYNQ-Z2.
  - Root / CAP_NET_RAW is required (standard on PYNQ).

Packet filtering:
  Only IPv4 TCP and UDP frames are forwarded. All other frames (ARP,
  IPv6, ICMP, …) are silently dropped at this stage.
"""

import socket
import struct
import time
import logging
import threading
import queue as stdlib_queue

from typing import Optional, NamedTuple

log = logging.getLogger(__name__)

# ── Ethernet / IP / TCP / UDP header constants ────────────────────────────────

ETH_HDR_LEN = 14
ETH_P_IP    = 0x0800

IPPROTO_TCP = 6
IPPROTO_UDP = 17

TCP_FLAG_FIN = 0x01
TCP_FLAG_SYN = 0x02
TCP_FLAG_RST = 0x04
TCP_FLAG_PSH = 0x08
TCP_FLAG_ACK = 0x10
TCP_FLAG_URG = 0x20


# ── Parsed packet type ────────────────────────────────────────────────────────

class ParsedPacket(NamedTuple):
    """Decoded Layer-3/4 fields extracted from one Ethernet frame."""
    ts_ns:        int    # CLOCK_REALTIME in nanoseconds
    src_ip:       str    # dotted-decimal string
    dst_ip:       str    # dotted-decimal string
    protocol:     int    # 6=TCP, 17=UDP
    src_port:     int
    dst_port:     int
    payload_len:  int    # IP payload minus transport header
    tcp_flags:    int    # Raw TCP flags byte (0 for UDP)
    ip_total_len: int    # Full IP datagram length


# ── Low-level parsing ─────────────────────────────────────────────────────────

def _parse_frame(data: bytes, ts_ns: int) -> Optional[ParsedPacket]:
    """
    Decode one raw Ethernet frame.

    Returns a ParsedPacket or None if the frame is not IPv4 TCP/UDP.
    This is a direct Python port of parse_packet() from parser/packet_parser.c.
    """
    if len(data) < ETH_HDR_LEN + 20:
        return None

    # Ethernet header: dst(6) src(6) ethertype(2)
    eth_type = struct.unpack_from("!H", data, 12)[0]
    if eth_type != ETH_P_IP:
        return None

    # IPv4 header
    ip_start = ETH_HDR_LEN
    version_ihl = data[ip_start]
    version  = (version_ihl >> 4) & 0xF
    ihl      = (version_ihl & 0xF) * 4
    if version != 4 or ihl < 20:
        return None

    protocol   = data[ip_start + 9]
    ip_tot_len = struct.unpack_from("!H", data, ip_start + 2)[0]
    src_ip_raw = struct.unpack_from("!I", data, ip_start + 12)[0]
    dst_ip_raw = struct.unpack_from("!I", data, ip_start + 16)[0]

    src_ip = socket.inet_ntoa(struct.pack("!I", src_ip_raw))
    dst_ip = socket.inet_ntoa(struct.pack("!I", dst_ip_raw))

    transport_start = ip_start + ihl
    tcp_flags = 0
    src_port  = 0
    dst_port  = 0
    payload_len = 0

    if protocol == IPPROTO_TCP:
        if len(data) < transport_start + 20:
            return None
        src_port  = struct.unpack_from("!H", data, transport_start)[0]
        dst_port  = struct.unpack_from("!H", data, transport_start + 2)[0]
        data_off  = (data[transport_start + 12] >> 4) * 4
        tcp_flags = data[transport_start + 13]
        payload_len = max(0, ip_tot_len - ihl - data_off)

    elif protocol == IPPROTO_UDP:
        if len(data) < transport_start + 8:
            return None
        src_port  = struct.unpack_from("!H", data, transport_start)[0]
        dst_port  = struct.unpack_from("!H", data, transport_start + 2)[0]
        udp_len   = struct.unpack_from("!H", data, transport_start + 4)[0]
        payload_len = max(0, udp_len - 8)

    else:
        return None

    return ParsedPacket(
        ts_ns        = ts_ns,
        src_ip       = src_ip,
        dst_ip       = dst_ip,
        protocol     = protocol,
        src_port     = src_port,
        dst_port     = dst_port,
        payload_len  = payload_len,
        tcp_flags    = tcp_flags,
        ip_total_len = ip_tot_len,
    )


# ── Capture thread ────────────────────────────────────────────────────────────

class PacketCapture:
    """
    Captures raw Ethernet frames on the specified interface and parses
    them into ParsedPacket objects pushed onto `out_queue`.

    Usage:
        cap = PacketCapture("lo", out_queue)
        cap.start()
        ...
        cap.stop()
    """

    def __init__(self, interface: str, out_queue: stdlib_queue.Queue,
                 rcvbuf_mb: int = 16):
        self.interface  = interface
        self.out_queue  = out_queue
        self.rcvbuf_mb  = rcvbuf_mb
        self._stop_flag = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True,
                                           name="capture")
        # Statistics
        self.frames_received = 0
        self.frames_parsed   = 0
        self.frames_dropped  = 0   # queue full

    def start(self) -> None:
        self._thread.start()
        log.info("[capture] Started on interface '%s'", self.interface)

    def stop(self) -> None:
        self._stop_flag.set()
        self._thread.join(timeout=3.0)
        log.info("[capture] Stopped. received=%d parsed=%d dropped=%d",
                 self.frames_received, self.frames_parsed, self.frames_dropped)

    def _open_socket(self) -> socket.socket:
        """Open and bind an AF_PACKET raw socket."""
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW,
                             socket.htons(0x0003))  # ETH_P_ALL
        sock.bind((self.interface, 0))
        # Increase kernel receive buffer to reduce drops
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUFFORCE,
                            self.rcvbuf_mb * 1024 * 1024)
        except OSError:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                            self.rcvbuf_mb * 1024 * 1024)
        sock.settimeout(1.0)   # Allow periodic stop checks
        return sock

    def _run(self) -> None:
        try:
            sock = self._open_socket()
        except OSError as exc:
            log.error("[capture] Cannot open socket: %s  (need root/CAP_NET_RAW)",
                      exc)
            return

        log.info("[capture] Socket open on '%s'", self.interface)

        while not self._stop_flag.is_set():
            try:
                data = sock.recv(65535)
            except socket.timeout:
                continue
            except OSError as exc:
                log.warning("[capture] recv error: %s", exc)
                continue

            self.frames_received += 1
            ts_ns = time.time_ns()

            pkt = _parse_frame(data, ts_ns)
            if pkt is None:
                continue

            self.frames_parsed += 1
            try:
                self.out_queue.put_nowait(pkt)
            except stdlib_queue.Full:
                self.frames_dropped += 1

        sock.close()