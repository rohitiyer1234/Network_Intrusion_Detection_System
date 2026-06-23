# NETWORK04 — FPGA-NIDS: Network Intrusion Detection System
### PYNQ-Z2 Zynq-7020 · MLP Classifier · Real-Time Flow Analysis

> This README reflects the **cleaned and consolidated** project layout.
> Architecture and technologies are unchanged from the original design —
> this pass fixed wiring bugs, removed duplicate/dead files, and reduced
> the run sequence to the minimum number of commands.

---

## Overview

NETWORK04 is a real-time Network Intrusion Detection System (NIDS) that runs bidirectional flow reconstruction and MLP-based traffic classification on a **PYNQ-Z2 FPGA board** (Zynq-7020, ARM Cortex-A9 PS). The system classifies network flows into three classes:

| Class | Description |
|-------|-------------|
| `normal` | Legitimate web, DNS, video streaming traffic |
| `ddos` | UDP flood, TCP SYN flood attacks |
| `botnet` | Command-and-control beaconing and data exfiltration |

> **This iteration** migrated the model from `normal/ddos/portscan` to
> **`normal/ddos/botnet`** (trained on CSE-CIC-IDS2018). PYNQ inference now loads the
> BN-folded float model **`model_fp.npz`** (pure NumPy, ~99.5%), and every classification
> carries a **per-flow inference time (ms)** surfaced on the dashboard and CLI.

**Architecture at a glance (unchanged):**

```
LAPTOP                                     PYNQ-Z2 (192.168.2.99)
──────────────────────────────────         ───────────────────────────────────────────
advanced_traffic_generator.py              receiver.py
  └─ 9 concurrent traffic threads  ──UDP──▶  UDPReceiver (port 5555)
     (Normal, DNS, Video,                        │  ParsedPacket
      DDoS flood, SYN flood,                     ▼
      Botnet C2, Botnet exfil,              FlowEngine  (bidirectional, FIN/RST + timeout)
      V-PortScan, H-PortScan*)                   │  FlowRecord
                                                 ▼
                                           FeatureExtractor  (20 CICIDS-style features)
                                                 │  ExtractedFlow
                                                 ▼
                                           NumpyMLP (FC→ReLU × 2 → Softmax; BN folded)
                                                 │  JSON result (+ inference_ms)
                                                 ▼
dashboard_server.py  ◀──TCP (port 9000)──  ResultSender
(* port-scan threads are out-of-distribution — the model has no portscan class)
  └─ Flask + SSE bridge, served on :8080
        │
        ▼
  Browser → http://localhost:8080
```

---

## Repository Structure (consolidated)

```
NETWORK04/
├── LAPTOP/
│   ├── advanced_traffic_generator.py   # 9-pattern concurrent traffic generator (production)
│   ├── dashboard_server.py             # TCP:9000 result sink + Flask/SSE dashboard on :8080
│   ├── export_model.py                 # One-time .pth → model_fp.npz conversion (BN folded)
│   └── tests/
│       ├── test_connectivity.py        # Minimal single-flow smoke test
│       └── run_classification_demo.py  # Normal → DDoS → Botnet → mixed demo, one command
│
├── PYNQ/
│   ├── receiver.py                     # Main PYNQ pipeline entry point
│   ├── capture.py                      # ParsedPacket type + TCP flag constants
│   ├── flow_engine.py                  # Bidirectional flow table (FIN/RST/timeout)
│   ├── extractor.py                    # 20-feature CICIDS extractor
│   ├── numpy_model.py                  # NumPy MLP inference engine (runtime model, BN folded)
│   ├── constants.py                    # Feature order, class names, normalization ranges
│   └── model_fp.npz                    # BN-folded float NumPy weights (runtime model)
│
├── tools/                              # Optional, NOT required to run the demo
│   ├── model_reference.py              # PyTorch architecture reference / retraining doc
│   ├── validate_numpy_model.py         # Offline sanity check for model_fp.npz
│   ├── evaluate_model.py               # Full metrics suite (needs pandas/sklearn + your own CSV)
│   └── classification_receiver_cli.py  # Console-only alternative to dashboard_server.py
│
├── requirements.txt
└── README.md
```

### What changed vs. the original layout
- `dashboard_server.py` moved into `LAPTOP/` (it's a laptop-side component).
- `LAPTOP/test_ddos.py`, `test_normal.py`, `test_portscan.py` **removed** — duplicated by `advanced_traffic_generator.py` and by the consolidated `tests/run_classification_demo.py`.
- `LAPTOP/integration_test.py` → `LAPTOP/tests/run_classification_demo.py` (now the canonical demo script).
- `LAPTOP/verified_flow_test.py` → `LAPTOP/tests/test_connectivity.py` (canonical smoke test).
- `LAPTOP/classification_receiver.py` → `tools/classification_receiver_cli.py` — this duplicated `dashboard_server.py`'s TCP:9000 role; kept only as an optional console-only fallback, clearly labeled as not part of the run path.
- `PYNQ/model.py` → `tools/model_reference.py` — not used at runtime (PYNQ inference is pure NumPy); kept as architecture/retraining documentation.
- `PYNQ/validate_numpy_model.py` → `tools/validate_numpy_model.py`.
- `PYNQ/evaluate_mode.py` → `tools/evaluate_model.py` — optional, requires `pandas`/`scikit-learn` and a labeled dataset CSV not included in this repo.
- All `__pycache__/` directories removed.

---

## Bugs Fixed in This Pass

| File | Bug | Fix |
|---|---|---|
| `PYNQ/receiver.py` | `//jugaad time display section` — invalid JS-style comment caused a Python `SyntaxError`. **The pipeline could not start at all.** | Removed; the surrounding verbose per-flow debug dump is now gated behind `--quiet`/debug instead of always printing. |
| `PYNQ/flow_engine.py` | Unconditional `print()` on every TCP packet and every completed flow — would flood the console and slow inference under DDoS/portscan load | Gated behind a new `debug` flag, wired from `receiver.py`'s existing `--quiet` flag |
| `LAPTOP/export_model.py` | Hardcoded relative paths assumed the `.pth` and output `.npz` were in the same directory as the script; the `.pth` actually lives in `PYNQ/` | Rewritten to resolve `PYNQ/balanced_dataset_mlp_model.pth` → `PYNQ/mlp_weights.npz` relative to the script's own location, regardless of caller's working directory |
| `LAPTOP/dashboard_server.py` | The "Avg confidence" line in the **Flow Count & Confidence by Class** bar chart never updated — `setClassConf(prev => ({...prev}))` just copied the (always-empty) previous state forever | Added real per-class confidence accumulation server-side (`class_conf_sums`/`class_conf_counts`), exposed as `class_avg_conf` in `/metrics` and `/stream`, and the frontend now reads it directly |
| `requirements.txt` | Listed `torch` as a **PYNQ** dependency (wrong — PYNQ inference is pure NumPy) and referenced a non-existent `websocket_client.py`; missing `flask` | Rewritten to accurately reflect what each component needs |
| `tools/validate_numpy_model.py`, `tools/evaluate_model.py`, `tools/model_reference.py` | All used import-relative / cwd-relative paths that broke once moved out of `PYNQ/` | Each now resolves `PYNQ/` and its own weight/checkpoint files relative to its own file location |

All `.py` files in the repository now compile cleanly (`python3 -m py_compile`), and the full pipeline was verified end-to-end on loopback (traffic generator → UDP:5555 → flow engine → feature extractor → MLP → TCP:9000 → dashboard `/metrics`), confirming Normal and DDoS classes report correctly with populated `features{}` and `class_avg_conf{}`.

### Note on port-scan traffic (out of distribution)
The model is trained on three classes — `normal`, `ddos`, `botnet` — and has **no portscan
class**. The generator still includes vertical/horizontal port-scan threads as
*out-of-distribution* traffic to show robustness; each scanned port becomes its own
single-packet 5-tuple flow and is classified into whichever of the three classes is nearest
(typically `normal`). This is expected, not a bug. Botnet C2/exfil traffic — which the model
*is* trained on — classifies correctly.

---

## Network Configuration

| Device | Interface | Static IP | Role |
|--------|-----------|-----------|------|
| Laptop | `eth0` / `enp*` | `192.168.2.1` | Traffic generator + dashboard backend |
| PYNQ-Z2 | `eth0` | `192.168.2.99` | Flow engine + MLP inference |

### Laptop — set static IP

```bash
sudo ip addr add 192.168.2.1/24 dev enp3s0
sudo ip link set enp3s0 up
```

### PYNQ-Z2 — set static IP (via SSH)

```bash
ssh xilinx@<pynq-dhcp-ip>   # default password: xilinx
sudo ip addr add 192.168.2.99/24 dev eth0
sudo ip link set eth0 up
```

Verify connectivity both ways: `ping 192.168.2.99` (from laptop) and `ping 192.168.2.1` (from PYNQ).

---

## Prerequisites

```bash
# Laptop
pip install flask          # dashboard_server.py
pip install torch numpy    # export_model.py only (one-time)

# PYNQ-Z2
pip install numpy          # receiver.py inference is pure NumPy — no torch needed on the board
```

See `requirements.txt` for the full breakdown, including optional tools.

---

## Setup — One-Time Steps

```bash
# 1. Export model weights (run on laptop; paths are resolved automatically).
#    Folds BatchNorm into the linear layers → self-contained NumPy model.
python3 LAPTOP/export_model.py
#   reads:  ../IDPModeltraining/model_fp.pth
#   writes: PYNQ/model_fp.npz

# 2. Copy PYNQ files to the board
scp -r PYNQ/ xilinx@192.168.2.99:~/nids/
```

(`model_fp.npz` is already included in this repo, so step 1 is only needed if you retrain the model.)

---

## Final Run Sequence (3 terminals, 3 commands)

### Terminal 1 — PYNQ board

```bash
ssh xilinx@192.168.2.99
cd ~/nids
python3 receiver.py --model model_fp.npz --laptop_ip 192.168.2.1
```

### Terminal 2 — Laptop: dashboard backend

```bash
python3 LAPTOP/dashboard_server.py
```

Open **http://localhost:8080** in your browser.

### Terminal 3 — Laptop: generate traffic (pick one)

For the full concurrent 9-pattern stress test (normal / ddos / botnet + OOD scans):
```bash
python3 LAPTOP/advanced_traffic_generator.py --target_ip 192.168.2.99
```

For a clean, easy-to-watch sequential demo (Normal → DDoS → Botnet → mixed finale, one command, finishes on its own):
```bash
python3 LAPTOP/tests/run_classification_demo.py --target_ip 192.168.2.99
```

For a minimal connectivity check before either of the above:
```bash
python3 LAPTOP/tests/test_connectivity.py --target_ip 192.168.2.99
```

That's it — 3 terminals, 3 commands, dashboard updates within 1–2 seconds.

---

## Dashboard Features

Served from the laptop on port 8080, updating via SSE at ~1 Hz:

- **Live classification donut** — percentage split of Normal / DDoS / Botnet flows
- **Flows/sec timeline** — rolling throughput chart
- **Per-class bar chart with confidence overlay** — cumulative flow counts + real per-class average confidence (`class_avg_conf`)
- **Avg Inference card** — mean NumPy inference time per flow (ms), streamed from the PYNQ receiver (`inference_ms` → `avg_inference_ms`)
- **Top attack source IPs** — ranked table of highest-volume non-normal sources
- **Recent flows feed** — live table of the last 20 classified flows with confidence scores
- **System health** — PYNQ TCP connection status, total flows, elapsed time

`GET /metrics` returns the same snapshot as JSON for scripting/automation; `GET /stream` is the raw SSE feed.

---

## Configuration Reference

### `receiver.py` flags (PYNQ)

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `model_fp.npz` | Path to NumPy weight file (BN-folded float model) |
| `--udp_port` | `5555` | UDP port to receive traffic from laptop |
| `--laptop_ip` | `192.168.2.1` | Laptop IP for streaming results back |
| `--laptop_port` | `9000` | Laptop TCP port |
| `--flow_timeout` | `1.0` | Idle flow eviction timeout (seconds) — lower to `0.5` for synthetic traffic, which rarely sends FIN/RST |
| `--quiet` | off | Suppress per-flow/per-packet debug output (recommended under heavy load) |

### `advanced_traffic_generator.py` / `tests/*.py` flags (Laptop)

| Flag | Default | Description |
|------|---------|-------------|
| `--target_ip` | `192.168.2.99` | PYNQ IP address |
| `--target_port` | `5555` | PYNQ UDP port |
| `--rate_scale` | `1.0` | (generator only) global rate multiplier |

### `dashboard_server.py` flags (Laptop)

| Flag | Default | Description |
|------|---------|-------------|
| `--tcp_port` | `9000` | TCP port to receive PYNQ classification results |
| `--http_port` | `8080` | HTTP port for the browser dashboard |
| `--host` | `0.0.0.0` | Bind address |

---

## MLP Model Architecture

```
Input: 20 CICIDS-style flow features (min-max normalized to [0, 1])

FC(20→64) → ReLU          (BatchNorm folded into the weights)
FC(64→32) → ReLU          (BatchNorm folded into the weights)
FC(32→3)  → Softmax

Output: P(normal), P(ddos), P(botnet)
```

Trained on a balanced **CSE-CIC-IDS2018** dataset using PyTorch (see
`../IDPModeltraining/`). The runtime model `PYNQ/model_fp.npz` has BatchNorm **folded into
the linear layers**, so the PYNQ forward is plain FC→ReLU — dependency-free NumPy inference
on the ARM Cortex-A9 (`PYNQ/numpy_model.py`), ~99.5% accuracy, ~0.02 ms/flow.
An INT8 quantized variant + FPGA `.mem` files (`../IDPModeltraining/model_qat.npz`,
~97% accuracy) are also available for the hardware datapath.

---

## 20 Feature Vector

| # | Feature | Description |
|---|---------|-------------|
| 0 | `flow_duration` | Total flow duration (s) |
| 1 | `total_fwd_packets` | Forward packet count |
| 2 | `total_bwd_packets` | Backward packet count |
| 3 | `total_len_fwd_pkts` | Forward payload bytes |
| 4 | `total_len_bwd_pkts` | Backward payload bytes |
| 5 | `fwd_pkt_len_max` | Max forward packet length |
| 6 | `fwd_pkt_len_min` | Min forward packet length |
| 7 | `bwd_pkt_len_max` | Max backward packet length |
| 8 | `bwd_pkt_len_min` | Min backward packet length |
| 9 | `flow_bytes_per_sec` | Total bytes / duration |
| 10 | `flow_packets_per_sec` | Total packets / duration |
| 11 | `flow_iat_mean` | Mean inter-arrival time (s) |
| 12 | `flow_iat_std` | Std dev of inter-arrival times |
| 13 | `fwd_iat_mean` | Mean forward IAT (s) |
| 14 | `bwd_iat_mean` | Mean backward IAT (s) |
| 15 | `fwd_psh_flags` | Forward PSH flag count |
| 16 | `bwd_psh_flags` | Backward PSH flag count |
| 17 | `syn_flag_count` | SYN flag count |
| 18 | `rst_flag_count` | RST flag count |
| 19 | `avg_fwd_segment_size` | Average forward payload length |

---

## Optional Tools (`tools/`)

These are **not required** to run the live demo:

- `tools/validate_numpy_model.py` — quick sanity check of `model_fp.npz` against three hand-built feature vectors (normal/ddos/botnet) plus a 100-sample random stress/latency test.
  ```bash
  python3 tools/validate_numpy_model.py
  ```
- `tools/evaluate_model.py` — full accuracy/precision/recall/F1/confusion-matrix report. Requires `pip install pandas scikit-learn` and your own labeled CSV (columns = the 20 features + `label`):
  ```bash
  python3 tools/evaluate_model.py --dataset /path/to/labeled_flows.csv
  ```
- `tools/model_reference.py` — the original PyTorch `TrafficClassifier` definition, useful if retraining. Requires `pip install torch`.
- `tools/classification_receiver_cli.py` — console-only colour dashboard, an alternative to `dashboard_server.py` for headless machines. **Do not run at the same time as `dashboard_server.py`** — both bind TCP:9000 by default.

---

## Troubleshooting

**PYNQ cannot receive UDP frames:**
- Confirm `ping 192.168.2.1` works from PYNQ
- Check UDP port is not blocked: `sudo ufw allow 5555/udp`

**Dashboard shows no data:**
- Confirm `dashboard_server.py` is running before opening the browser
- Confirm PYNQ's `ResultSender` connected: look for `[sender] Connected to 192.168.2.1:9000` in the PYNQ logs, and `[tcp] PYNQ connected from ...` in the dashboard logs

**Flow timeout too slow / too many incomplete flows:**
- Lower `--flow_timeout` to `0.5` for synthetic traffic (which rarely sends FIN/RST)

**Port scans classify as "normal" (or another class):**
- Expected — portscan is **out of distribution** (the model has no portscan class). See the note above. Normal, DDoS and Botnet traffic classify correctly.

**Model file not found:**
- Run `python3 LAPTOP/export_model.py` first (it resolves paths automatically regardless of your working directory), then `scp PYNQ/model_fp.npz` to the board if you regenerated it.

---

## Authors

RV College of Engineering — Electronics and Communication Engineering, VI Semester
SUSTAIN-ATHON 2025-26 | Mentor: Dr. Ravishankar Holla
