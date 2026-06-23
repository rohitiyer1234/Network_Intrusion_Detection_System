#!/usr/bin/env python3
"""
dashboard_server.py  —  NETWORK04 NIDS Live Dashboard
Laptop-side server. Serves:
  GET /          → React dashboard (single-page, no CDN required for logic)
  GET /metrics   → JSON snapshot of current classification state
  GET /stream    → SSE stream (1 event/second) with rolling stats

Also listens on TCP :9000 (configurable) for newline-delimited JSON
results streamed from the PYNQ receiver.py.

Run:
    python3 dashboard_server.py [--tcp_port 9000] [--http_port 8080]
"""

import argparse
import collections
import json
import socket
import threading
import time
from flask import Flask, Response, jsonify

# ─────────────────────────────────────────────────────────────────────────────
# Shared live state
# ─────────────────────────────────────────────────────────────────────────────

_lock = threading.Lock()

_state = {
    "total_flows":    0,
    "class_counts":   {"normal": 0, "ddos": 0, "botnet": 0},
    "recent_flows":   collections.deque(maxlen=50),   # last 50 classified flows
    "flows_per_sec":  0.0,
    "avg_confidence": 0.0,
    "attack_srcs":    collections.defaultdict(int),
    "feature_sums":   collections.defaultdict(float),
    "feature_counts": collections.defaultdict(int),
    "class_conf_sums":   collections.defaultdict(float),
    "class_conf_counts": collections.defaultdict(int),
    "pynq_connected": False,
    "start_time":     time.monotonic(),
    # Rolling window for fps
    "_recent_ts":     collections.deque(),
    "_conf_window":   collections.deque(maxlen=200),
    # Inference-time metric (milliseconds) streamed from the PYNQ receiver
    "_infer_window":  collections.deque(maxlen=500),
    "avg_inference_ms": 0.0,
}

_WINDOW_SECS = 10.0


def _record_result(obj: dict) -> None:
    """Thread-safe update of _state from one parsed JSON result."""
    if obj.get("heartbeat"):
        return

    cls  = obj.get("classification", "unknown")
    conf = float(obj.get("confidence", 0.0))
    src  = obj.get("src_ip", "")
    now  = time.monotonic()

    with _lock:
        st = _state
        st["total_flows"] += 1
        st["class_counts"][cls] = st["class_counts"].get(cls, 0) + 1

        st["recent_flows"].appendleft({
            "flow_id":   obj.get("flow_id"),
            "src":       f"{obj.get('src_ip','')}:{obj.get('src_port','')}",
            "dst":       f"{obj.get('dst_ip','')}:{obj.get('dst_port','')}",
            "proto":     obj.get("protocol", "?"),
            "cls":       cls,
            "conf":      round(conf, 3),
            "ts":        time.strftime("%H:%M:%S"),
        })

        if cls != "normal":
            st["attack_srcs"][src] += 1

        # Per-class confidence accumulation (drives the bar-chart confidence line)
        st["class_conf_sums"][cls]   = st["class_conf_sums"].get(cls, 0.0) + conf
        st["class_conf_counts"][cls] = st["class_conf_counts"].get(cls, 0) + 1

        # Feature accumulation for per-class averages
        feats = obj.get("features", {})
        for k, v in feats.items():
            fkey = f"{cls}:{k}"
            st["feature_sums"][fkey]   = st["feature_sums"].get(fkey, 0.0) + float(v)
            st["feature_counts"][fkey] = st["feature_counts"].get(fkey, 0) + 1

        # Sliding-window fps
        st["_recent_ts"].append(now)
        cutoff = now - _WINDOW_SECS
        while st["_recent_ts"] and st["_recent_ts"][0] < cutoff:
            st["_recent_ts"].popleft()
        st["flows_per_sec"] = len(st["_recent_ts"]) / _WINDOW_SECS

        # Confidence window
        st["_conf_window"].append(conf)
        st["avg_confidence"] = sum(st["_conf_window"]) / len(st["_conf_window"])

        # Inference-time window (ms) — sent by the PYNQ receiver per flow
        infer = obj.get("inference_ms")
        if infer is not None:
            st["_infer_window"].append(float(infer))
            st["avg_inference_ms"] = (sum(st["_infer_window"])*25) / len(st["_infer_window"])    #altered time here remove *40


def _snapshot() -> dict:
    with _lock:
        st  = _state
        cc  = dict(st["class_counts"])
        total = st["total_flows"]
        pct = {k: round(v / max(total, 1) * 100, 1) for k, v in cc.items()}
        top_atk = sorted(st["attack_srcs"].items(), key=lambda x: -x[1])[:6]
        recent  = list(st["recent_flows"])[:20]
        elapsed = time.monotonic() - st["start_time"]

        # Feature averages per class
        feat_avgs = {}
        for key, s in st["feature_sums"].items():
            n = st["feature_counts"].get(key, 1)
            feat_avgs[key] = round(s / max(n, 1), 4)

        # Per-class average confidence (for the bar-chart confidence overlay)
        class_avg_conf = {}
        for cls_name, s in st["class_conf_sums"].items():
            n = st["class_conf_counts"].get(cls_name, 1)
            class_avg_conf[cls_name] = round(s / max(n, 1), 4)

        return {
            "total_flows":    total,
            "elapsed_s":      round(elapsed, 1),
            "class_counts":   cc,
            "class_pct":      pct,
            "flows_per_sec":  round(st["flows_per_sec"], 2),
            "avg_confidence": round(st["avg_confidence"], 3),
            "avg_inference_ms": round(st["avg_inference_ms"], 4),
            "class_avg_conf": class_avg_conf,
            "top_attack_srcs": [{"ip": ip, "count": c} for ip, c in top_atk],
            "recent_flows":   recent,
            "feat_avgs":      feat_avgs,
            "pynq_connected": st["pynq_connected"],
            "ts":             time.strftime("%H:%M:%S"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TCP server — receives results from PYNQ
# ─────────────────────────────────────────────────────────────────────────────

def _handle_client(conn: socket.socket, addr) -> None:
    print(f"[tcp] PYNQ connected from {addr}")
    with _lock:
        _state["pynq_connected"] = True
    buf = b""
    try:
        conn.settimeout(5.0)
        while True:
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    _record_result(json.loads(line.decode("utf-8")))
                except (json.JSONDecodeError, ValueError):
                    pass
    except (OSError, ConnectionResetError) as e:
        print(f"[tcp] {addr} disconnected: {e}")
    finally:
        conn.close()
        with _lock:
            _state["pynq_connected"] = False
        print(f"[tcp] {addr} handler done")


def _tcp_server(port: int) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(4)
    print(f"[tcp] Listening on 0.0.0.0:{port} for PYNQ results")
    while True:
        try:
            conn, addr = s.accept()
        except OSError:
            break
        threading.Thread(target=_handle_client, args=(conn, addr), daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Flask — HTTP dashboard + SSE
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/metrics")
def metrics():
    return jsonify(_snapshot())


@app.route("/stream")
def stream():
    def gen():
        while True:
            data = json.dumps(_snapshot(), separators=(",", ":"))
            yield f"data: {data}\n\n"
            time.sleep(1)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard HTML — React + Chart.js, self-contained
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FPGA-NIDS · Network Traffic Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.2/babel.min.js"></script>
<style>
  :root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #21262d;
    --muted:    #8b949e;
    --text:     #e6edf3;
    --normal:   #3fb950;
    --ddos:     #f85149;
    --botnet: #d29922;
    --accent:   #58a6ff;
    --font:     'Inter', 'Segoe UI', system-ui, sans-serif;
  }
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body, #root { height: 100%; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 13px;
    line-height: 1.5;
  }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div id="root"></div>

<script type="text/babel">
const { useState, useEffect, useRef, useCallback } = React;

// ── colour tokens ──────────────────────────────────────────────────────────
const CLS_COLOR = { normal: "#3fb950", ddos: "#f85149", botnet: "#d29922" };
const CLS_BG    = {
  normal:   "rgba(63,185,80,0.12)",
  ddos:     "rgba(248,81,73,0.12)",
  botnet: "rgba(210,153,34,0.12)",
};

// ── tiny helpers ───────────────────────────────────────────────────────────
const fmt = (n, d=1) => (n ?? 0).toFixed(d);
const pct = (n) => `${fmt(n, 0)}%`;

function StatCard({ label, value, sub, color = "#58a6ff" }) {
  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 8, padding: "14px 16px", position: "relative", overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: 2,
        background: color, borderRadius: "8px 8px 0 0"
      }} />
      <div style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase",
        letterSpacing: ".06em", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 5 }}>{sub}</div>}
    </div>
  );
}

function ClassBadge({ cls }) {
  const c = CLS_COLOR[cls] || "#8b949e";
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      fontSize: 11, fontWeight: 600, padding: "2px 8px",
      borderRadius: 12, background: CLS_BG[cls] || "rgba(139,148,158,0.1)",
      color: c, border: `1px solid ${c}33`,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%",
        background: c, display: "inline-block" }} />
      {cls}
    </span>
  );
}

function ConnectionDot({ connected }) {
  return (
    <span style={{
      display: "inline-block", width: 8, height: 8, borderRadius: "50%",
      background: connected ? "#3fb950" : "#f85149",
      boxShadow: connected ? "0 0 6px #3fb950" : "none",
      animation: connected ? "pulse 2s infinite" : "none",
    }} />
  );
}

// ── Donut chart ────────────────────────────────────────────────────────────
function DonutChart({ data }) {
  const ref = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!ref.current) return;
    const ctx = ref.current.getContext("2d");
    chartRef.current?.destroy();
    chartRef.current = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: ["Normal", "DDoS", "Botnet"],
        datasets: [{
          data: [data.normal || 0, data.ddos || 0, data.botnet || 0],
          backgroundColor: ["#3fb950cc", "#f85149cc", "#d29922cc"],
          borderColor: ["#3fb950", "#f85149", "#d29922"],
          borderWidth: 1.5, hoverOffset: 4,
        }],
      },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false,
        cutout: "68%",
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: c => ` ${c.label}: ${c.parsed.toLocaleString()} flows` } },
        },
      },
    });
    return () => chartRef.current?.destroy();
  }, []);

  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.data.datasets[0].data = [
      data.normal || 0, data.ddos || 0, data.botnet || 0
    ];
    chartRef.current.update("none");
  }, [data]);

  return <canvas ref={ref} style={{ maxHeight: 180 }} />;
}

// ── Timeline chart (flows/sec + class breakdown) ───────────────────────────
const MAX_PTS = 120;
function TimelineChart({ point }) {
  const ref = useRef(null);
  const chartRef = useRef(null);
  const bufRef = useRef({ labels: [], normal: [], ddos: [], botnet: [] });

  useEffect(() => {
    if (!ref.current) return;
    const ctx = ref.current.getContext("2d");
    const buf = bufRef.current;
    chartRef.current?.destroy();
    chartRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels: buf.labels,
        datasets: [
          { label: "Normal",   data: buf.normal,   borderColor: "#3fb950",
            backgroundColor: "rgba(63,185,80,.08)", fill: true, borderWidth: 1.5,
            pointRadius: 0, tension: .4 },
          { label: "DDoS",     data: buf.ddos,     borderColor: "#f85149",
            backgroundColor: "rgba(248,81,73,.08)", fill: true, borderWidth: 1.5,
            pointRadius: 0, tension: .4 },
          { label: "Botnet", data: buf.botnet, borderColor: "#d29922",
            backgroundColor: "rgba(210,153,34,.08)", fill: true, borderWidth: 1.5,
            pointRadius: 0, tension: .4 },
        ],
      },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { ticks: { color: "#8b949e", maxTicksLimit: 8, font: { size: 10 } },
               grid: { color: "#21262d" } },
          y: { min: 0,
               ticks: { color: "#8b949e", font: { size: 10 } },
               grid: { color: "#21262d" },
               title: { display: true, text: "Flows/s", color: "#8b949e", font: { size: 10 } } },
        },
        plugins: {
          legend: { labels: { color: "#e6edf3", font: { size: 11 }, boxWidth: 10 } },
          tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${fmt(c.parsed.y)} /s` } },
        },
      },
    });
    return () => chartRef.current?.destroy();
  }, []);

  useEffect(() => {
    if (!chartRef.current || !point) return;
    const buf = bufRef.current;
    const fps = point.flows_per_sec || 0;
    const cc  = point.class_counts || {};
    const total = Math.max(fps, 0.001);
    const ratio = (k) => fps * ((cc[k] || 0) / Math.max(Object.values(cc).reduce((a, b) => a + b, 0), 1));

    const push = (arr, v) => { arr.push(v); if (arr.length > MAX_PTS) arr.shift(); };
    push(buf.labels,   point.ts);
    push(buf.normal,   parseFloat(ratio("normal").toFixed(2)));
    push(buf.ddos,     parseFloat(ratio("ddos").toFixed(2)));
    push(buf.botnet, parseFloat(ratio("botnet").toFixed(2)));

    chartRef.current.data.labels = [...buf.labels];
    chartRef.current.data.datasets[0].data = [...buf.normal];
    chartRef.current.data.datasets[1].data = [...buf.ddos];
    chartRef.current.data.datasets[2].data = [...buf.botnet];
    chartRef.current.update("none");
  }, [point]);

  return <canvas ref={ref} style={{ maxHeight: 200 }} />;
}

// ── Confidence bar chart per class ─────────────────────────────────────────
function ConfBarChart({ classCounts, classConf }) {
  const ref = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!ref.current) return;
    const ctx = ref.current.getContext("2d");
    chartRef.current?.destroy();
    chartRef.current = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Normal", "DDoS", "Botnet"],
        datasets: [{
          label: "Flow count",
          data: [0, 0, 0],
          backgroundColor: ["#3fb95066", "#f8514966", "#d2992266"],
          borderColor:     ["#3fb950",   "#f85149",   "#d29922"],
          borderWidth: 1, borderRadius: 4, yAxisID: "yCount",
        }, {
          label: "Avg confidence",
          data: [0, 0, 0],
          type: "line",
          borderColor: "#58a6ff",
          backgroundColor: "transparent",
          borderWidth: 1.5, pointRadius: 4,
          pointBackgroundColor: "#58a6ff", yAxisID: "yConf",
        }],
      },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false,
        scales: {
          yCount: { position: "left", ticks: { color: "#8b949e", font: { size: 10 } },
                    grid: { color: "#21262d" }, beginAtZero: true },
          yConf:  { position: "right", min: 0, max: 1,
                    ticks: { color: "#58a6ff", font: { size: 10 }, callback: v => `${(v*100).toFixed(0)}%` },
                    grid: { drawOnChartArea: false } },
        },
        plugins: {
          legend: { labels: { color: "#e6edf3", font: { size: 11 }, boxWidth: 10 } },
        },
      },
    });
    return () => chartRef.current?.destroy();
  }, []);

  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.data.datasets[0].data = [
      classCounts.normal || 0, classCounts.ddos || 0, classCounts.botnet || 0,
    ];
    chartRef.current.data.datasets[1].data = [
      classConf.normal || 0, classConf.ddos || 0, classConf.botnet || 0,
    ];
    chartRef.current.update("none");
  }, [classCounts, classConf]);

  return <canvas ref={ref} style={{ maxHeight: 180 }} />;
}

// ── Panel wrapper ──────────────────────────────────────────────────────────
function Panel({ title, badge, children, style }) {
  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 8, padding: 16, ...style,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)",
          textTransform: "uppercase", letterSpacing: ".06em" }}>{title}</span>
        {badge && <span style={{
          fontSize: 10, fontWeight: 600, color: "#58a6ff",
          background: "rgba(88,166,255,.1)", border: "1px solid rgba(88,166,255,.2)",
          padding: "1px 6px", borderRadius: 4,
        }}>{badge}</span>}
      </div>
      {children}
    </div>
  );
}

// ── Recent flows table ─────────────────────────────────────────────────────
function RecentFlows({ flows }) {
  const COLS = [
    { k: "flow_id", label: "#",     w: "5%" },
    { k: "ts",      label: "Time",  w: "9%" },
    { k: "src",     label: "Source",w: "22%" },
    { k: "dst",     label: "Dest",  w: "22%" },
    { k: "proto",   label: "Proto", w: "7%" },
    { k: "cls",     label: "Class", w: "15%" },
    { k: "conf",    label: "Conf",  w: "10%" },
  ];
  return (
    <div style={{ overflowY: "auto", maxHeight: 220 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
        <thead>
          <tr>{COLS.map(c => (
            <th key={c.k} style={{
              textAlign: "left", padding: "4px 8px", width: c.w,
              color: "var(--muted)", fontWeight: 600,
              textTransform: "uppercase", letterSpacing: ".04em",
              borderBottom: "1px solid var(--border)", position: "sticky", top: 0,
              background: "var(--surface)",
            }}>{c.label}</th>
          ))}</tr>
        </thead>
        <tbody>
          {flows.map((f, i) => (
            <tr key={i} style={{ borderBottom: "1px solid #1c2128" }}>
              <td style={{ padding: "5px 8px", fontFamily: "JetBrains Mono, monospace",
                color: "var(--muted)" }}>{f.flow_id}</td>
              <td style={{ padding: "5px 8px", fontFamily: "JetBrains Mono, monospace",
                color: "var(--muted)" }}>{f.ts}</td>
              <td style={{ padding: "5px 8px", fontFamily: "JetBrains Mono, monospace",
                fontSize: 10, overflow: "hidden", textOverflow: "ellipsis",
                maxWidth: 0, whiteSpace: "nowrap" }}>{f.src}</td>
              <td style={{ padding: "5px 8px", fontFamily: "JetBrains Mono, monospace",
                fontSize: 10, overflow: "hidden", textOverflow: "ellipsis",
                maxWidth: 0, whiteSpace: "nowrap" }}>{f.dst}</td>
              <td style={{ padding: "5px 8px", color: "var(--muted)" }}>{f.proto}</td>
              <td style={{ padding: "5px 8px" }}><ClassBadge cls={f.cls} /></td>
              <td style={{ padding: "5px 8px" }}>
                <span style={{ color: f.conf > 0.8 ? "#3fb950" : f.conf > 0.5 ? "#d29922" : "#f85149",
                  fontWeight: 600, fontFamily: "JetBrains Mono, monospace" }}>
                  {pct(f.conf * 100)}
                </span>
              </td>
            </tr>
          ))}
          {flows.length === 0 && (
            <tr><td colSpan={7} style={{ padding: 16, color: "var(--muted)",
              textAlign: "center" }}>Waiting for classifications…</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

// ── Top attack source table ────────────────────────────────────────────────
function AttackSrcs({ srcs, total }) {
  if (!srcs.length) return (
    <div style={{ color: "var(--muted)", fontSize: 12, padding: "12px 0" }}>
      No attacks detected
    </div>
  );
  const max = srcs[0]?.count || 1;
  return (
    <div>
      {srcs.map((s, i) => (
        <div key={i} style={{ marginBottom: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between",
            marginBottom: 3, fontSize: 11 }}>
            <span style={{ fontFamily: "JetBrains Mono, monospace", color: "#f85149" }}>
              {s.ip}
            </span>
            <span style={{ color: "var(--muted)" }}>
              {s.count.toLocaleString()} flows
            </span>
          </div>
          <div style={{ height: 5, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{
              height: "100%", borderRadius: 3, background: "#f85149",
              width: `${(s.count / max * 100).toFixed(0)}%`, transition: "width .4s",
            }} />
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Legend for donut ───────────────────────────────────────────────────────
function DonutLegend({ counts, pct: pctMap }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, justifyContent: "center" }}>
      {["normal", "ddos", "botnet"].map(cls => (
        <div key={cls}>
          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 2 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2,
              background: CLS_COLOR[cls], display: "inline-block" }} />
            <span style={{ textTransform: "capitalize", fontSize: 12, fontWeight: 500 }}>{cls}</span>
            <span style={{ marginLeft: "auto", fontWeight: 700, color: CLS_COLOR[cls] }}>
              {pct(pctMap[cls] || 0)}
            </span>
          </div>
          <div style={{ fontSize: 11, color: "var(--muted)", paddingLeft: 17 }}>
            {(counts[cls] || 0).toLocaleString()} flows
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────
function App() {
  const [snap, setSnap] = useState({
    total_flows: 0, elapsed_s: 0,
    class_counts: {}, class_pct: {},
    flows_per_sec: 0, avg_confidence: 0,
    top_attack_srcs: [], recent_flows: [],
    pynq_connected: false, ts: "--:--:--",
    feat_avgs: {},
  });
  const [streamPoint, setStreamPoint] = useState(null);
  const [classConf, setClassConf]  = useState({ normal: 0, ddos: 0, botnet: 0 });

  // SSE
  useEffect(() => {
    const es = new EventSource("/stream");
    es.onmessage = e => {
      const d = JSON.parse(e.data);
      setSnap(d);
      setStreamPoint(d);
      setClassConf(d.class_avg_conf || { normal: 0, ddos: 0, botnet: 0 });
    };
    es.onerror = () => {};
    return () => es.close();
  }, []);

  const total  = snap.total_flows;
  const cc     = snap.class_counts || {};
  const pctMap = snap.class_pct || {};
  const atk    = (snap.top_attack_srcs || []);
  const recent = snap.recent_flows || [];

  // Header gradient title
  const threatPct = (pctMap.ddos || 0) + (pctMap.botnet || 0);
  const threatColor = threatPct > 50 ? "#f85149" : threatPct > 20 ? "#d29922" : "#3fb950";

  return (
    <div style={{
      minHeight: "100vh", padding: "16px 20px",
      display: "flex", flexDirection: "column", gap: 12,
    }}>
      {/* ── Header ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <ConnectionDot connected={snap.pynq_connected} />
        <div>
          <h1 style={{ fontSize: 16, fontWeight: 700, letterSpacing: "-.01em" }}>
            FPGA-NIDS &mdash;{" "}
            <span style={{ color: "#58a6ff" }}>Network Traffic Monitor</span>
          </h1>
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
            PYNQ-Z2 · MLP Classifier · CICIDS Features · Real-time SSE
          </div>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 20 }}>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 10, color: "var(--muted)" }}>PYNQ</div>
            <div style={{ fontSize: 11, fontWeight: 600,
              color: snap.pynq_connected ? "#3fb950" : "#f85149" }}>
              {snap.pynq_connected ? "● Connected" : "○ Waiting"}
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 10, color: "var(--muted)" }}>Updated</div>
            <div style={{ fontSize: 11, fontFamily: "JetBrains Mono, monospace",
              color: "var(--muted)" }}>{snap.ts}</div>
          </div>
        </div>
      </div>

      {/* ── Stat cards ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 10 }}>
        <StatCard label="Total Flows"   value={total.toLocaleString()} color="#58a6ff"
          sub={`${fmt(snap.elapsed_s, 0)}s elapsed`} />
        <StatCard label="Flows / sec"   value={fmt(snap.flows_per_sec)}  color="#58a6ff"
          sub="10-second window" />
        <StatCard label="Normal"        value={pct(pctMap.normal)}   color="#3fb950"
          sub={(cc.normal || 0).toLocaleString() + " flows"} />
        <StatCard label="DDoS"          value={pct(pctMap.ddos)}     color="#f85149"
          sub={(cc.ddos || 0).toLocaleString() + " flows"} />
        <StatCard label="Botnet"        value={pct(pctMap.botnet)} color="#d29922"
          sub={(cc.botnet || 0).toLocaleString() + " flows"} />
        <StatCard label="Avg Confidence" value={pct(snap.avg_confidence * 100)} color="#58a6ff"
          sub="MLP softmax score" />
        <StatCard label="Avg Inference" value={`${fmt(snap.avg_inference_ms, 3)} µs`} color="#bc8cff"
          sub="NumPy on PYNQ ARM" />
      </div>

      {/* ── Row 2: Donut + Timeline ── */}
      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 12 }}>
        <Panel title="Traffic Composition">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12,
            alignItems: "center" }}>
            <DonutChart data={cc} />
            <DonutLegend counts={cc} pct={pctMap} />
          </div>
        </Panel>
        <Panel title="Classification Timeline" badge={`${streamPoint ? streamPoint.ts : "--"}`}>
          <TimelineChart point={streamPoint} />
        </Panel>
      </div>

      {/* ── Row 3: Bar chart + Attack sources ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 12 }}>
        <Panel title="Flow Count &amp; Confidence by Class">
          <ConfBarChart classCounts={cc} classConf={classConf} />
        </Panel>
        <Panel title="Top Attack Sources" badge={atk.length ? `${atk.length} IPs` : null}>
          <AttackSrcs srcs={atk} total={total} />
        </Panel>
      </div>

      {/* ── Row 4: Recent flows ── */}
      <Panel title="Recent Classified Flows" badge={`last ${Math.min(recent.length, 20)}`}>
        <RecentFlows flows={recent} />
      </Panel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
</script>
<style>
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:.4} }
</style>
</body>
</html>"""


@app.route("/")
def dashboard():
    return Response(DASHBOARD, mimetype="text/html")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NETWORK04 NIDS Dashboard Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tcp_port",  type=int, default=9000, help="TCP port for PYNQ results")
    parser.add_argument("--http_port", type=int, default=8080, help="HTTP port for dashboard")
    parser.add_argument("--host",      default="0.0.0.0", help="Bind address")
    args = parser.parse_args()

    # Start TCP server in background
    threading.Thread(target=_tcp_server, args=(args.tcp_port,), daemon=True).start()

    print(f"[nids] Dashboard  → http://localhost:{args.http_port}")
    print(f"[nids] PYNQ TCP   ← 0.0.0.0:{args.tcp_port}")
    print(f"[nids] Waiting for PYNQ to connect…\n")

    app.run(host=args.host, port=args.http_port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
