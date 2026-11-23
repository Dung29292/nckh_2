"""Ket noi camera dien thoai qua Wi-Fi LAN (HTTPS + WebSocket + QR Code).

Cung cap lop ``MobileVideoCapture`` tuong thich hoan toan voi giao dien
``cv2.VideoCapture`` (``isOpened``, ``read``, ``get``, ``set``, ``release``)
de dua luong frame BGR (NumPy array) tu trinh duyet dien thoai vao pipeline
xu ly anh / tach phong / quay mau ma khong can cai app tren dien thoai.
"""
from __future__ import annotations

import collections
import datetime
import hmac
import ipaddress
import json
import secrets
import socket
import ssl
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import simple_websocket
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from flask import Flask, Response, abort, request
from werkzeug.serving import WSGIRequestHandler, make_server

from . import paths


# ---------------------------------------------------------------------------
# MANG LAN & CONG KET NOI
# ---------------------------------------------------------------------------

_VIRTUAL_IFACE_KEYWORDS = (
    "radmin",
    "vpn",
    "virtual",
    "vmware",
    "vbox",
    "wsl",
    "hyper-v",
    "vethernet",
    "docker",
    "tailscale",
    "zerotier",
    "hamachi",
    "loopback",
    "teredo",
    "pseudo",
    "isatap",
)


def get_lan_ip(preferred_ip: str | None = None) -> str:
    """Tu dong tim IP LAN thuc te (uu tien Wi-Fi / Ethernet, bo qua Radmin VPN / WSL / may ao)."""
    if preferred_ip:
        return preferred_ip

    # 1. Dung psutil neu co de chon dung card mang vat ly dang hoat dong
    try:
        import psutil

        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        candidates: list[tuple[int, str]] = []
        for if_name, snic_list in addrs.items():
            st = stats.get(if_name)
            if not st or not st.isup:
                continue
            low = if_name.lower()
            if any(kw in low for kw in _VIRTUAL_IFACE_KEYWORDS):
                continue
            for sn in snic_list:
                if sn.family != socket.AF_INET:
                    continue
                ip_obj = ipaddress.ip_address(sn.address)
                if ip_obj.is_private and not ip_obj.is_loopback and not ip_obj.is_link_local:
                    # Uu tien Wi-Fi -> Ethernet -> cac card vat ly khac
                    if any(k in low for k in ("wi-fi", "wifi", "wlan", "wireless")):
                        prio = 0
                    elif "ethernet" in low:
                        prio = 1
                    else:
                        prio = 2
                    candidates.append((prio, str(ip_obj)))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
    except Exception:
        pass

    # 2. Fallback: hoi bang dinh tuyen OS qua UDP socket (khong gui goi tin ra mang)
    for target in ("192.168.1.1", "8.8.8.8"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((target, 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127.") and not ip.startswith("169.254."):
                return ip
        except OSError:
            pass
        finally:
            sock.close()

    return "127.0.0.1"


def find_available_port(host: str = "0.0.0.0", preferred_port: int = 8443) -> int:
    """Tim cong TCP kha dung, uu tien ``preferred_port`` de giu on dinh quyen Firewall."""
    candidates = [preferred_port, *range(preferred_port + 1, preferred_port + 11), 0]
    for port in candidates:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            return int(sock.getsockname()[1])
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError("Khong tim thay cong mang kha dung cho may chu camera dien thoai.")


# ---------------------------------------------------------------------------
# CHUNG CHI SSL TU KY (BAT BUOC DE DIEN THOAI CHO PHEP getUserMedia)
# ---------------------------------------------------------------------------

def ensure_ssl_cert(ip_str: str, cert_dir: Path | None = None) -> tuple[Path, Path]:
    """Tao va cache chung chi X.509 tu ky co SAN khop voi IP LAN hien tai.

    Trinh duyet di dong (Chrome Android / Safari iOS) chan ``getUserMedia`` tren
    cac dia chi IP LAN ``http://`` khong ma hoa. Chung chi tu ky cho phep
    mo ``https://<LAN_IP>:<PORT>`` va WebSocket ``wss://`` tren cung mot cong.
    """
    if cert_dir is None:
        cert_dir = paths.DATA_DIR / "raw" / "certs"
    cert_dir.mkdir(parents=True, exist_ok=True)

    safe_ip = ip_str.replace(".", "_").replace(":", "_")
    cert_path = cert_dir / f"cert_{safe_ip}.pem"
    key_path = cert_dir / f"key_{safe_ip}.pem"

    # Tai su dung cert con han (>= 7 ngay) de nguoi dung khong phai xac nhan lai moi lan chay
    if cert_path.exists() and key_path.exists():
        try:
            existing = x509.load_pem_x509_certificate(cert_path.read_bytes())
            remaining = existing.not_valid_after_utc - datetime.datetime.now(datetime.timezone.utc)
            if remaining > datetime.timedelta(days=7):
                return cert_path, key_path
        except Exception:
            pass

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NCKH_2 Local Vision Host"),
        x509.NameAttribute(NameOID.COMMON_NAME, ip_str),
    ])
    now = datetime.datetime.now(datetime.timezone.utc)

    san_entries: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    try:
        ip_obj = ipaddress.ip_address(ip_str)
        if str(ip_obj) != "127.0.0.1":
            san_entries.append(x509.IPAddress(ip_obj))
    except ValueError:
        san_entries.append(x509.DNSName(ip_str))

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_cert_sign=True,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


# ---------------------------------------------------------------------------
# TAO MA QR (OPENCV NATIVE + ASCII TERMINAL)
# ---------------------------------------------------------------------------

def generate_qr_matrix(url: str) -> np.ndarray:
    """Tao ma tran QR nhi phan (0 = den, 255 = trang) bang OpenCV."""
    encoder = cv2.QRCodeEncoder.create()
    return encoder.encode(url)


def generate_qr_image(url: str, size_px: int = 320, border_px: int = 24) -> np.ndarray:
    """Tao anh QR code BGR sac net de hien thi len cua so Desktop."""
    matrix = generate_qr_matrix(url)
    qr_scaled = cv2.resize(matrix, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
    bordered = cv2.copyMakeBorder(
        qr_scaled, border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT, value=255,
    )
    return cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)


def render_qr_terminal(url: str) -> str:
    """Ve ma QR gon gang bang ky tu Unicode half-block de quet ngay tu Terminal."""
    matrix = generate_qr_matrix(url)
    # Them quiet zone 2 o
    padded = np.pad(matrix < 128, pad_width=2, mode="constant", constant_values=False)
    h, w = padded.shape
    lines: list[str] = []
    for y in range(0, h, 2):
        top = padded[y]
        bot = padded[y + 1] if y + 1 < h else np.zeros(w, dtype=bool)
        row_chars = []
        for t, b in zip(top, bot):
            if t and b:
                row_chars.append("█")
            elif t and not b:
                row_chars.append("▀")
            elif not t and b:
                row_chars.append("▄")
            else:
                row_chars.append(" ")
        lines.append("  " + "".join(row_chars))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GIAO DIEN WEB MOBILE (HTML5 + getUserMedia + WebSocket Binary Stream)
# ---------------------------------------------------------------------------

_MOBILE_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover" />
  <title>Mobile Camera Stream - NCKH 2</title>
  <style>
    :root {
      --bg: #090d16;
      --panel: rgba(15, 23, 42, 0.85);
      --border: rgba(255, 255, 255, 0.16);
      --accent: #3b82f6;
      --danger: #ef4444;
      --success: #10b981;
      --warn: #f59e0b;
      --text: #f8fafc;
      --muted: #94a3b8;
    }
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-tap-highlight-color: transparent;
      user-select: none;
      -webkit-user-select: none;
    }
    html, body {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100dvh;
      overflow: hidden;
      touch-action: none;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .stage {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #000;
      touch-action: none;
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #000;
      pointer-events: none;
    }

    /* 3D Capture Guide Overlay */
    .guide-overlay {
      position: absolute;
      inset: 0;
      pointer-events: none;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 5;
      transition: opacity 0.25s;
    }
    .guide-overlay.hidden { opacity: 0; }
    .guide-box {
      position: relative;
      width: min(76vw, 56vh);
      height: min(76vw, 56vh);
      border: 1px dashed rgba(255, 255, 255, 0.35);
      border-radius: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .orbit-ring {
      position: absolute;
      width: 82%;
      height: 82%;
      border: 1.5px dashed rgba(96, 165, 250, 0.5);
      border-radius: 50%;
      pointer-events: none;
    }
    .level-indicator {
      position: absolute;
      top: 12px;
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 3px 10px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.15);
      font-size: 11px;
      font-weight: 700;
      color: #94a3b8;
      backdrop-filter: blur(6px);
      transition: border-color 0.15s, color 0.15s;
    }
    .level-indicator.aligned {
      border-color: #10b981;
      color: #6ee7b7;
      background: rgba(6, 78, 59, 0.6);
    }
    .level-line {
      width: 22px;
      height: 2px;
      background: #94a3b8;
      transform-origin: center;
      transition: transform 0.08s ease-out, background 0.15s;
    }
    .level-indicator.aligned .level-line {
      background: #10b981;
    }
    .corner {
      position: absolute;
      width: 20px;
      height: 20px;
      border-color: #60a5fa;
      border-style: solid;
    }
    .corner.tl { top: -2px; left: -2px; border-width: 3px 0 0 3px; border-top-left-radius: 20px; }
    .corner.tr { top: -2px; right: -2px; border-width: 3px 3px 0 0; border-top-right-radius: 20px; }
    .corner.bl { bottom: -2px; left: -2px; border-width: 0 0 3px 3px; border-bottom-left-radius: 20px; }
    .corner.br { bottom: -2px; right: -2px; border-width: 0 3px 3px 0; border-bottom-right-radius: 20px; }
    .crosshair {
      width: 22px;
      height: 22px;
      position: relative;
    }
    .crosshair::before, .crosshair::after {
      content: "";
      position: absolute;
      background: rgba(255, 255, 255, 0.65);
    }
    .crosshair::before { top: 10px; left: 0; right: 0; height: 2px; }
    .crosshair::after { left: 10px; top: 0; bottom: 0; width: 2px; }
    .guide-label {
      position: absolute;
      bottom: 10px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.5px;
      color: rgba(255, 255, 255, 0.85);
      background: rgba(15, 23, 42, 0.7);
      backdrop-filter: blur(6px);
      padding: 3px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.12);
    }

    /* Top HUD */
    .hud-top {
      position: absolute;
      top: env(safe-area-inset-top, 12px);
      left: 12px;
      right: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 8px 14px;
      border-radius: 14px;
      background: var(--panel);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      font-size: 13px;
      z-index: 10;
    }
    .status-group { display: flex; align-items: center; gap: 8px; font-weight: 600; }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--warn); }
    .dot.ok { background: var(--success); box-shadow: 0 0 8px var(--success); }
    .dot.err { background: var(--danger); }
    .rec-pill {
      display: none; align-items: center; gap: 6px;
      padding: 3px 10px; border-radius: 999px;
      background: rgba(239, 68, 68, 0.25); border: 1px solid var(--danger);
      color: #fecaca; font-weight: 700; font-size: 12px;
      animation: pulse 1.2s infinite;
    }
    .rec-pill.active { display: inline-flex; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.55; } }
    .stats { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }

    /* Pro Timer & Comparison Banner */
    .pro-timer-banner {
      position: absolute;
      top: calc(env(safe-area-inset-top, 12px) + 54px);
      left: 50%;
      transform: translateX(-50%);
      display: none;
      align-items: center;
      gap: 8px;
      padding: 6px 16px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.85);
      border: 1.5px solid var(--border);
      backdrop-filter: blur(12px);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
      z-index: 15;
      font-variant-numeric: tabular-nums;
    }
    .pro-timer-banner.recording {
      display: inline-flex;
      border-color: var(--danger);
      background: rgba(30, 10, 10, 0.92);
    }
    .pro-timer-banner.recorded-done {
      display: inline-flex;
      border-color: #3b82f6;
      background: rgba(15, 23, 42, 0.92);
    }
    .timer-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--danger);
      animation: pulse 1s infinite;
    }
    .pro-timer-banner.recorded-done .timer-dot {
      background: #10b981;
      animation: none;
    }
    .timer-digits {
      font-size: 16px;
      font-weight: 800;
      color: #fff;
      letter-spacing: 0.5px;
    }
    .timer-frames {
      font-size: 12px;
      font-weight: 600;
      color: #94a3b8;
    }

    /* Zoom Pill */
    .zoom-pill {
      position: absolute;
      bottom: calc(env(safe-area-inset-bottom, 12px) + 115px);
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      align-items: center;
      gap: 4px;
      background: rgba(15, 23, 42, 0.8);
      backdrop-filter: blur(10px);
      padding: 4px 6px;
      border-radius: 999px;
      border: 1px solid var(--border);
      z-index: 10;
    }
    .zoom-btn {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      border: none;
      background: transparent;
      color: var(--text);
      font-size: 12px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      padding: 0;
    }
    .zoom-btn.active {
      background: rgba(255, 255, 255, 0.2);
      color: #60a5fa;
    }

    /* Bottom HUD */
    .hud-bottom {
      position: absolute;
      bottom: calc(env(safe-area-inset-bottom, 12px) + 8px);
      left: 12px;
      right: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 10px 12px;
      border-radius: 18px;
      background: var(--panel);
      backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      z-index: 10;
    }
    .controls-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr 1.6fr;
      gap: 6px;
    }
    .settings-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }
    button, select {
      appearance: none;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.08);
      color: var(--text);
      padding: 10px 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      text-align: center;
      transition: transform 0.1s, background 0.15s;
    }
    button:active { transform: scale(0.96); }
    button.active { background: var(--accent); border-color: #60a5fa; }
    button.rec-btn {
      background: rgba(239, 68, 68, 0.25);
      border-color: var(--danger);
      color: #fee2e2;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      font-weight: 700;
    }
    button.rec-btn .rec-icon {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--danger);
      transition: border-radius 0.2s;
    }
    button.rec-btn.recording {
      background: var(--danger);
      color: #fff;
    }
    button.rec-btn.recording .rec-icon {
      border-radius: 2px;
      background: #fff;
    }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    select { padding: 6px 8px; font-size: 11px; text-align: left; flex: 1; max-width: 49%; }

    .banner-err {
      position: absolute;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(9, 13, 22, 0.94);
      z-index: 30;
      text-align: center;
    }
    .banner-err.show { display: flex; }
    .banner-box {
      max-width: 400px;
      padding: 20px;
      border-radius: 16px;
      background: #111827;
      border: 1px solid var(--danger);
    }
    .banner-box h3 { color: #fca5a5; margin-bottom: 8px; }
    .banner-box p { color: #cbd5e1; font-size: 14px; line-height: 1.5; margin-bottom: 14px; }
  </style>
</head>
<body>
  <div class="stage">
    <video id="video" autoplay playsinline muted></video>

    <!-- 3D Capture Guide Overlay -->
    <div id="guideOverlay" class="guide-overlay">
      <div class="guide-box">
        <div class="corner tl"></div>
        <div class="corner tr"></div>
        <div class="corner bl"></div>
        <div class="corner br"></div>
        <div class="crosshair"></div>
        <div class="orbit-ring"></div>
        <div id="levelBar" class="level-indicator">
          <div id="levelLine" class="level-line"></div>
          <span id="levelText">0°</span>
        </div>
        <span class="guide-label">GIỮ VẬT THỂ Ở TÂM & XOAY 360° ĐỀU TAY</span>
      </div>
    </div>

    <!-- Top HUD -->
    <div class="hud-top">
      <div class="status-group">
        <span id="statusDot" class="dot"></span>
        <span id="statusText">Đang kết nối...</span>
      </div>
      <div id="recBadge" class="rec-pill">● REC <span id="recTimer" style="margin-left:4px">00:00</span></div>
      <div id="statsText" class="stats">0 fps | 0 KB</div>
    </div>

    <!-- Dedicated Big Stopwatch Overlay -->
    <div id="proTimerBanner" class="pro-timer-banner">
      <div class="timer-dot"></div>
      <span id="bigTimerText" class="timer-digits">00:00.0</span>
      <span id="bigFrameText" class="timer-frames">(0 f)</span>
    </div>

    <!-- Hardware Zoom Pill -->
    <div id="zoomPill" class="zoom-pill">
      <button id="zoom1xBtn" type="button" class="zoom-btn active">1x</button>
      <button id="zoom2xBtn" type="button" class="zoom-btn">2x</button>
    </div>

    <!-- Bottom HUD -->
    <div class="hud-bottom">
      <div class="settings-row">
        <select id="resSelect" title="Độ phân giải & Tốc độ">
          <option value="1280x720@60" selected>720p (Khuyên dùng: Siêu mượt 30-60 FPS)</option>
          <option value="960x540@60">540p (Siêu tốc độ 60 FPS)</option>
          <option value="1280x720@30">720p @ 30 FPS (Chuẩn ổn định)</option>
          <option value="1920x1080@30">1080p (Rất nặng - Cần máy mạnh)</option>
        </select>
        <select id="cameraSelect" title="Chọn ống kính">
          <option value="">Mặc định: Camera sau</option>
        </select>
      </div>
      <div class="controls-row">
        <button id="switchBtn" type="button">🔄 Đổi cam</button>
        <button id="torchBtn" type="button">🔦 Flash</button>
        <button id="guideBtn" type="button" class="active">📐 Khung</button>
        <button id="recBtn" type="button" class="rec-btn">
          <span class="rec-icon"></span>
          <span id="recBtnText">Quay</span>
        </button>
      </div>
    </div>

    <div id="errBanner" class="banner-err">
      <div class="banner-box">
        <h3 id="errTitle">Không mở được Camera</h3>
        <p id="errMsg"></p>
        <button id="retryBtn" type="button" class="active" style="width:100%">Thử lại</button>
      </div>
    </div>
  </div>

  <script>
    const params = new URLSearchParams(window.location.search);
    const TOKEN = params.get("token") || "";

    const videoEl = document.getElementById("video");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const statsText = document.getElementById("statsText");
    const recBadge = document.getElementById("recBadge");
    const recTimer = document.getElementById("recTimer");
    const resSelect = document.getElementById("resSelect");
    const cameraSelect = document.getElementById("cameraSelect");
    const switchBtn = document.getElementById("switchBtn");
    const torchBtn = document.getElementById("torchBtn");
    const guideBtn = document.getElementById("guideBtn");
    const guideOverlay = document.getElementById("guideOverlay");
    const levelBar = document.getElementById("levelBar");
    const levelLine = document.getElementById("levelLine");
    const levelText = document.getElementById("levelText");
    const recBtn = document.getElementById("recBtn");
    const recBtnText = document.getElementById("recBtnText");
    const zoom1xBtn = document.getElementById("zoom1xBtn");
    const zoom2xBtn = document.getElementById("zoom2xBtn");
    const errBanner = document.getElementById("errBanner");
    const errTitle = document.getElementById("errTitle");
    const errMsg = document.getElementById("errMsg");
    const retryBtn = document.getElementById("retryBtn");
    const proTimerBanner = document.getElementById("proTimerBanner");
    const bigTimerText = document.getElementById("bigTimerText");
    const bigFrameText = document.getElementById("bigFrameText");

    // Bo dem don tinh gon & toc do cao (Zero-Overhead Single Buffer)
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d", { alpha: false });
    let inFlightBlob = false;

    let stream = null;
    let ws = null;
    let facingMode = "environment";
    let selectedDeviceId = "";
    let torchOn = false;
    let torchSupported = false;
    let streaming = true;
    let wakeLock = null;

    let zoomCaps = null;
    let currentZoom = 1.0;
    let touchStartDist = 0;
    let baseZoom = 1.0;

    let TARGET_FPS = 60;
    let TARGET_W = 1280;
    let TARGET_H = 720;
    const JPEG_QUALITY = 0.65;
    let lastSentMs = 0;
    let frameCounter = 0;
    let lastKb = 0;

    let isRecording = false;
    let recStartTime = 0;
    let recTimerInterval = null;
    let recordedFrameCount = 0;

    function setStatus(state, text) {
      statusDot.className = "dot " + (state === "ok" ? "ok" : state === "err" ? "err" : "");
      statusText.textContent = text;
    }

    function showError(title, msg) {
      errTitle.textContent = title;
      errMsg.textContent = msg;
      errBanner.classList.add("show");
    }

    function hideError() {
      errBanner.classList.remove("show");
    }

    async function requestWakeLock() {
      try {
        if ("wakeLock" in navigator) {
          wakeLock = await navigator.wakeLock.request("screen");
        }
      } catch (_) {}
    }

    function stopCamera() {
      torchOn = false;
      torchBtn.classList.remove("active");
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
      }
    }

    function applyZoom(z) {
      if (!stream || !zoomCaps) return;
      const clamped = Math.min(Math.max(z, zoomCaps.min), zoomCaps.max);
      currentZoom = clamped;
      const track = stream.getVideoTracks()[0];
      if (track && track.applyConstraints) {
        track.applyConstraints({ advanced: [{ zoom: clamped }] }).catch(() => {});
      }
      if (zoom1xBtn) zoom1xBtn.classList.toggle("active", Math.abs(currentZoom - 1.0) < 0.2);
      if (zoom2xBtn) zoom2xBtn.classList.toggle("active", Math.abs(currentZoom - 2.0) < 0.2);
    }

    async function populateCameraList() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter((d) => d.kind === "videoinput");
        const prev = cameraSelect.value;
        cameraSelect.innerHTML = "";
        videoInputs.forEach((dev, idx) => {
          const opt = document.createElement("option");
          opt.value = dev.deviceId;
          opt.textContent = dev.label || `Camera ${idx + 1}`;
          cameraSelect.appendChild(opt);
        });
        if (selectedDeviceId && videoInputs.some((d) => d.deviceId === selectedDeviceId)) {
          cameraSelect.value = selectedDeviceId;
        } else if (stream) {
          const track = stream.getVideoTracks()[0];
          const curId = track.getSettings ? track.getSettings().deviceId : "";
          if (curId && videoInputs.some((d) => d.deviceId === curId)) {
            cameraSelect.value = curId;
          } else if (prev) {
            cameraSelect.value = prev;
          }
        }
      } catch (_) {}
    }

    async function startCamera() {
      hideError();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showError(
          "Trình duyệt chặn Camera",
          "Cần truy cập bằng HTTPS (hoặc localhost). Hãy kiểm tra lại đường dẫn https:// và xác nhận chứng chỉ SSL."
        );
        return;
      }

      stopCamera();
      const [resPart, fpsPart] = resSelect.value.split("@");
      const [reqW, reqH] = resPart.split("x").map(Number);
      TARGET_FPS = fpsPart ? Number(fpsPart) : 60;
      TARGET_W = reqW;
      TARGET_H = reqH;

      // Tu dong dao chieu rong / cao theo dung huong xoay (Portrait / Landscape)
      const isPortrait = window.innerHeight > window.innerWidth;
      const idealW = isPortrait ? Math.min(reqW, reqH) : Math.max(reqW, reqH);
      const idealH = isPortrait ? Math.max(reqW, reqH) : Math.min(reqW, reqH);

      const videoConstraint = selectedDeviceId
        ? { deviceId: { exact: selectedDeviceId }, width: { ideal: idealW }, height: { ideal: idealH }, frameRate: { ideal: TARGET_FPS, min: 30 } }
        : { facingMode: { ideal: facingMode }, width: { ideal: idealW }, height: { ideal: idealH }, frameRate: { ideal: TARGET_FPS, min: 30 } };

      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: videoConstraint });
        videoEl.srcObject = stream;
        await videoEl.play();
        await populateCameraList();
        await requestWakeLock();

        const track = stream.getVideoTracks()[0];
        const caps = track && track.getCapabilities ? track.getCapabilities() : {};
        torchSupported = Boolean(caps && caps.torch);
        torchBtn.disabled = !torchSupported;
        torchBtn.textContent = torchSupported ? "🔦 Flash" : "🔦 Flash (N/A)";

        if (caps && caps.zoom) {
          zoomCaps = caps.zoom;
          currentZoom = Math.max(1.0, zoomCaps.min);
        } else {
          zoomCaps = null;
        }
      } catch (err) {
        console.error("getUserMedia error:", err);
        showError(
          "Không thể mở Camera",
          `${err.name}: ${err.message}. Hãy cấp quyền Camera cho trình duyệt và thử lại.`
        );
      }
    }

    function playBeep(freq = 880, duration = 0.12) {
      try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
      } catch (_) {}
    }

    function connectWebSocket() {
      const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProto}//${window.location.host}/ws?token=${encodeURIComponent(TOKEN)}`;
      setStatus("wait", "Đang kết nối PC...");
      ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        setStatus("ok", "Đã kết nối PC");
      };

      ws.onmessage = (ev) => {
        if (typeof ev.data === "string") {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === "status" && typeof msg.recording === "boolean") {
              const wasRecording = isRecording;
              isRecording = msg.recording;
              recBadge.classList.toggle("active", isRecording);
              recBtn.classList.toggle("recording", isRecording);
              recBtnText.textContent = isRecording ? "Dừng" : "Quay";

              if (isRecording) {
                recStartTime = Date.now();
                recordedFrameCount = 0;
                playBeep(880, 0.15);
                if ("vibrate" in navigator) navigator.vibrate([50, 40, 50]);
                if (proTimerBanner) {
                  proTimerBanner.className = "pro-timer-banner recording";
                }
                if (recTimerInterval) clearInterval(recTimerInterval);
                recTimerInterval = setInterval(() => {
                  if (!isRecording) return;
                  const elapsedMs = Date.now() - recStartTime;
                  const totalSec = Math.floor(elapsedMs / 1000);
                  const mm = String(Math.floor(totalSec / 60)).padStart(2, "0");
                  const ss = String(totalSec % 60).padStart(2, "0");
                  const timeStr = `${mm}:${ss}`;

                  if (recTimer) recTimer.textContent = `${mm}:${ss} | ${recordedFrameCount}f`;
                  if (bigTimerText) bigTimerText.textContent = timeStr;
                  if (bigFrameText) {
                    const fpsEst = elapsedMs > 500 ? (recordedFrameCount / (elapsedMs / 1000)).toFixed(1) : "0.0";
                    bigFrameText.textContent = `🔴 Đang ghi: ${recordedFrameCount} f (${fpsEst} fps)`;
                  }
                }, 250);
              } else {
                if (recTimerInterval) clearInterval(recTimerInterval);
                recTimerInterval = null;
                if (wasRecording) {
                  playBeep(587, 0.1);
                  setTimeout(() => playBeep(880, 0.18), 120);
                  if ("vibrate" in navigator) navigator.vibrate([100]);
                }
                if (proTimerBanner && recordedFrameCount > 0) {
                  proTimerBanner.className = "pro-timer-banner recorded-done";
                  const elapsedSec = ((Date.now() - recStartTime) / 1000).toFixed(1);
                  if (bigTimerText) bigTimerText.textContent = `Đã quay: ${elapsedSec}s`;
                  if (bigFrameText) bigFrameText.textContent = `✅ ${recordedFrameCount} frames -> Đã lưu trên PC`;
                } else if (proTimerBanner) {
                  proTimerBanner.className = "pro-timer-banner";
                  if (bigTimerText) bigTimerText.textContent = "00:00.0";
                  if (bigFrameText) bigFrameText.textContent = "Chờ quay (Bấm [R] trên PC hoặc [Quay])";
                }
                if (recTimer) recTimer.textContent = "00:00";
              }
            }
          } catch (_) {}
        }
      };

      ws.onclose = (ev) => {
        if (ev.code === 4003) {
          setStatus("err", "Sai Session Token");
          return;
        }
        setStatus("err", "Mất kết nối, thử lại...");
        setTimeout(connectWebSocket, 1500);
      };

      ws.onerror = () => {
        setStatus("err", "Lỗi WebSocket");
      };
    }

    function sendLoop(now) {
      requestAnimationFrame(sendLoop);
      if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
      if (inFlightBlob || ws.bufferedAmount > 524288) return;
      if (videoEl.readyState < 2 || !videoEl.videoWidth || !videoEl.videoHeight) return;
      const frameInterval = 1000 / TARGET_FPS;
      if (now - lastSentMs < frameInterval - 4) return;

      lastSentMs = now;
      if (canvas.width !== videoEl.videoWidth || canvas.height !== videoEl.videoHeight) {
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
      }
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
      inFlightBlob = true;

      canvas.toBlob(
        (blob) => {
          inFlightBlob = false;
          if (blob && ws && ws.readyState === WebSocket.OPEN && ws.bufferedAmount <= 1048576) {
            ws.send(blob);
            frameCounter++;
            if (isRecording) recordedFrameCount++;
            lastKb = Math.round(blob.size / 1024);
          }
        },
        "image/jpeg",
        JPEG_QUALITY
      );
    }

    setInterval(() => {
      const vw = videoEl.videoWidth || 0;
      const vh = videoEl.videoHeight || 0;
      let camFps = TARGET_FPS;
      if (stream) {
        const track = stream.getVideoTracks()[0];
        if (track && track.getSettings && track.getSettings().frameRate) {
          camFps = Math.round(track.getSettings().frameRate);
        }
      }
      statsText.textContent = `${vw}x${vh} | Cam: ${camFps}f | Net: ${frameCounter} fps | ${lastKb} KB`;
      frameCounter = 0;
    }, 1000);

    // Gestures: Pinch to zoom
    const stageEl = document.querySelector(".stage");
    stageEl.addEventListener("touchstart", (e) => {
      if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        touchStartDist = Math.hypot(dx, dy);
        baseZoom = currentZoom;
      }
    }, { passive: true });

    stageEl.addEventListener("touchmove", (e) => {
      if (e.touches.length === 2 && touchStartDist > 0 && zoomCaps) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.hypot(dx, dy);
        const factor = dist / touchStartDist;
        applyZoom(baseZoom * factor);
      }
    }, { passive: true });

    zoom1xBtn.addEventListener("click", () => {
      if ("vibrate" in navigator) navigator.vibrate(20);
      applyZoom(1.0);
    });

    zoom2xBtn.addEventListener("click", () => {
      if ("vibrate" in navigator) navigator.vibrate(20);
      applyZoom(2.0);
    });

    switchBtn.addEventListener("click", () => {
      if ("vibrate" in navigator) navigator.vibrate(20);
      selectedDeviceId = "";
      facingMode = facingMode === "environment" ? "user" : "environment";
      startCamera();
    });

    cameraSelect.addEventListener("change", () => {
      selectedDeviceId = cameraSelect.value;
      startCamera();
    });

    resSelect.addEventListener("change", () => {
      startCamera();
    });

    torchBtn.addEventListener("click", async () => {
      if (!stream || !torchSupported) return;
      const track = stream.getVideoTracks()[0];
      try {
        torchOn = !torchOn;
        await track.applyConstraints({ advanced: [{ torch: torchOn }] });
        torchBtn.classList.toggle("active", torchOn);
        if ("vibrate" in navigator) navigator.vibrate(20);
      } catch (err) {
        torchOn = false;
        torchBtn.classList.remove("active");
        alert("Không bật được đèn Flash trên ống kính này: " + err.message);
      }
    });

    guideBtn.addEventListener("click", () => {
      guideOverlay.classList.toggle("hidden");
      guideBtn.classList.toggle("active", !guideOverlay.classList.contains("hidden"));
      if ("vibrate" in navigator) navigator.vibrate(20);
    });

    recBtn.addEventListener("click", () => {
      if ("vibrate" in navigator) navigator.vibrate([40, 60, 40]);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "control", action: "toggle_record" }));
      }
    });

    retryBtn.addEventListener("click", startCamera);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") requestWakeLock();
    });

    startCamera();
    connectWebSocket();
    requestAnimationFrame(sendLoop);
  </script>
</body>
</html>
"""


class _MobileRequestHandler(WSGIRequestHandler):
    """Tuy bien WSGI handler de an log truy cap va khong ghi de header HTTP 200 sau khi da nang cap WebSocket."""

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        pass

    def _ws_hijacked(self) -> bool:
        env = getattr(self, "environ", None)
        return bool(env and env.get("nckh.ws_hijacked"))

    def send_response(self, code: int, message: str | None = None) -> None:
        if not self._ws_hijacked():
            super().send_response(code, message)

    def send_header(self, keyword: str, value: str) -> None:
        if not self._ws_hijacked():
            super().send_header(keyword, value)

    def end_headers(self) -> None:
        if not self._ws_hijacked():
            super().end_headers()

    def run_wsgi(self) -> None:
        try:
            super().run_wsgi()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# LOP MOBILE VIDEO CAPTURE (DROP-IN CHO cv2.VideoCapture)
# ---------------------------------------------------------------------------

class MobileVideoCapture:
    """Nhan video stream thoi gian thuc tu trinh duyet dien thoai nhu mot ``cv2.VideoCapture``.

    Cach dung doc lap voi bat ky pipeline OpenCV nao::

        with MobileVideoCapture() as cap:
            cap.wait_for_connection(show_window=True)
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                # `frame` la numpy.ndarray (H, W, 3) BGR chuan OpenCV
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        port: int = 8443,
        host_ip: str | None = None,
        use_ssl: bool = True,
    ) -> None:
        self.target_width = int(width)
        self.target_height = int(height)
        self.target_fps = float(fps)
        self.use_ssl = bool(use_ssl)

        self.local_ip = get_lan_ip(host_ip)
        self.port = find_available_port("0.0.0.0", port)
        self.token = secrets.token_urlsafe(24)

        scheme = "https" if self.use_ssl else "http"
        self.url = f"{scheme}://{self.local_ip}:{self.port}/?token={self.token}"

        self._opened = True
        self._connected = False
        self._stop_event = threading.Event()
        self._frame_cond = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._frame_seq = 0
        self._last_read_seq = 0

        self._actual_width = 0
        self._actual_height = 0
        self._frame_timestamps: collections.deque[float] = collections.deque(maxlen=30)

        self._ws_lock = threading.Lock()
        self._active_ws: simple_websocket.Server | None = None
        self._recording_state = False
        self._record_queue: collections.deque[np.ndarray] = collections.deque(maxlen=1200)
        self._cmd_queue: collections.deque[str] = collections.deque()

        self._qr_img_cache: np.ndarray | None = None
        self._server = self._create_server()
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="MobileCameraServer",
            daemon=True,
        )
        self._server_thread.start()

    def _check_token(self, candidate: str | None) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate, self.token)

    def _create_server(self):
        app = Flask(__name__)
        capture_ref = self

        @app.get("/")
        def index():
            if not capture_ref._check_token(request.args.get("token")):
                abort(403, description="Phien truy cap khong hop le hoac da het han (Sai token).")
            resp = Response(_MOBILE_HTML, mimetype="text/html; charset=utf-8")
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        @app.get("/health")
        def health():
            return {"status": "ok", "connected": capture_ref._connected}

        @app.route("/ws", websocket=True)
        def ws_stream():
            if not capture_ref._check_token(request.args.get("token")):
                abort(403)
            try:
                ws = simple_websocket.Server.accept(request.environ)
            except simple_websocket.ConnectionError:
                return "WebSocket upgrade required", 400
            request.environ["nckh.ws_hijacked"] = True
            capture_ref._handle_ws_session(ws)
            return ""

        ssl_ctx = None
        if self.use_ssl:
            cert_path, key_path = ensure_ssl_cert(self.local_ip)
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(str(cert_path), str(key_path))

        server = make_server(
            "0.0.0.0",
            self.port,
            app,
            threaded=True,
            ssl_context=ssl_ctx,
            request_handler=_MobileRequestHandler,
        )
        server.daemon_threads = True
        return server

    def _handle_ws_session(self, ws: simple_websocket.Server) -> None:
        with self._ws_lock:
            old_ws = self._active_ws
            self._active_ws = ws
            self._connected = True
        if old_ws is not None and old_ws is not ws:
            try:
                old_ws.close()
            except Exception:
                pass

        # Dong bo trang thai REC hien tai xuong dien thoai
        self.set_recording_state(self._recording_state)

        try:
            while not self._stop_event.is_set() and ws.connected:
                msg = ws.receive(timeout=1.0)
                if msg is None:
                    continue
                if isinstance(msg, bytes):
                    arr = np.frombuffer(msg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    now = time.monotonic()
                    with self._frame_cond:
                        self._actual_height, self._actual_width = frame.shape[:2]
                        self._latest_frame = frame
                        self._frame_seq += 1
                        self._frame_timestamps.append(now)
                        if self._recording_state:
                            self._record_queue.append(frame)
                        self._frame_cond.notify_all()
                elif isinstance(msg, str):
                    try:
                        payload = json.loads(msg)
                        if payload.get("type") == "control" and payload.get("action"):
                            self._cmd_queue.append(str(payload["action"]))
                    except json.JSONDecodeError:
                        pass
        except simple_websocket.ConnectionClosed:
            pass
        except Exception:
            pass
        finally:
            with self._ws_lock:
                if self._active_ws is ws:
                    self._active_ws = None
                    self._connected = False
            with self._frame_cond:
                self._frame_cond.notify_all()

    # --- Giao dien chuan tuong thich cv2.VideoCapture ---

    def isOpened(self) -> bool:
        return self._opened and not self._stop_event.is_set()

    def grab(self, timeout: float = 5.0) -> bool:
        """Cho va danh dau da co frame moi tu dien thoai (tuong thich cv2.VideoCapture.grab)."""
        if not self.isOpened():
            return False
        deadline = None if timeout is None else (time.monotonic() + timeout)
        with self._frame_cond:
            while self._frame_seq <= self._last_read_seq:
                if self._stop_event.is_set():
                    return False
                if deadline is not None:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        return False
                    self._frame_cond.wait(timeout=min(rem, 0.25))
                else:
                    self._frame_cond.wait(timeout=0.25)
            return True

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        """Tra ve frame BGR moi nhat (tuong thich cv2.VideoCapture.retrieve)."""
        with self._frame_cond:
            if self._latest_frame is None:
                return False, None
            self._last_read_seq = self._frame_seq
            return True, self._latest_frame.copy()

    def read(self, timeout: float = 5.0) -> tuple[bool, np.ndarray | None]:
        """Cho frame moi tiep theo va tra ve ``(ok, bgr_frame)`` giong ``cv2.VideoCapture.read``."""
        if not self.grab(timeout=timeout):
            return False, None
        return self.retrieve()

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._actual_width or self.target_width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._actual_height or self.target_height)
        if prop_id == cv2.CAP_PROP_FPS:
            with self._frame_cond:
                if len(self._frame_timestamps) >= 2:
                    dt = self._frame_timestamps[-1] - self._frame_timestamps[0]
                    if dt > 0:
                        return float((len(self._frame_timestamps) - 1) / dt)
            return self.target_fps
        return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self.target_width = int(value)
            return True
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self.target_height = int(value)
            return True
        if prop_id == cv2.CAP_PROP_FPS:
            self.target_fps = float(value)
            return True
        return False

    def release(self) -> None:
        """Tat WebSocket va may chu HTTPS, giai phong cong mang."""
        if not self._opened and self._stop_event.is_set():
            return
        self._opened = False
        self._stop_event.set()
        with self._ws_lock:
            ws = self._active_ws
            self._active_ws = None
            self._connected = False
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        with self._frame_cond:
            self._frame_cond.notify_all()
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass

    def __enter__(self) -> "MobileVideoCapture":
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.release()

    # --- Tien ich UI Desktop & Dieu khien 2 chieu ---

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def has_frame(self) -> bool:
        with self._frame_cond:
            return self._latest_frame is not None

    def get_live_preview(self) -> np.ndarray | None:
        """Tra ve frame thoi gian thuc moi nhat (Zero Latency) ma khong block."""
        with self._frame_cond:
            return self._latest_frame

    def pop_recorded_frames(self) -> list[np.ndarray]:
        """Lay toan bo frame da duoc ghi nhan trong hang doi khi dang quay."""
        with self._frame_cond:
            if not self._record_queue:
                return []
            frames = list(self._record_queue)
            self._record_queue.clear()
            return frames

    def set_recording_state(self, recording: bool) -> None:
        self._recording_state = bool(recording)
        if self._recording_state:
            with self._frame_cond:
                self._record_queue.clear()
        with self._ws_lock:
            ws = self._active_ws
            if ws is not None and ws.connected:
                try:
                    ws.send(json.dumps({"type": "status", "recording": self._recording_state}))
                except Exception:
                    pass

    def pop_command(self) -> str | None:
        if self._cmd_queue:
            return self._cmd_queue.popleft()
        return None

    def get_qr_card(self, card_w: int = 960, card_h: int = 580) -> np.ndarray:
        """Tao khung hinh huong dan + Ma QR quet bang dien thoai de hien tren OpenCV."""
        if self._qr_img_cache is None:
            self._qr_img_cache = generate_qr_image(self.url, size_px=310, border_px=18)

        card = np.full((card_h, card_w, 3), (24, 28, 36), dtype=np.uint8)
        qh, qw = self._qr_img_cache.shape[:2]
        qx, qy = 48, (card_h - qh) // 2
        card[qy : qy + qh, qx : qx + qw] = self._qr_img_cache

        tx = qx + qw + 42
        cv2.putText(card, "QUET MA QR BANG DIEN THOAI", (tx, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (96, 165, 250), 2)
        cv2.putText(card, "1. Ket noi dien thoai & may tinh CUNG mang Wi-Fi", (tx, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (226, 232, 240), 1)
        cv2.putText(card, "2. Mo Camera dien thoai va quet ma QR ben trai", (tx, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (226, 232, 240), 1)
        if self.use_ssl:
            cv2.putText(card, "3. Trinh duyet bao mat: chon Nang cao -> Tiep tuc", (tx, 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (251, 191, 36), 1)
            cv2.putText(card, "   (Chrome: Advanced -> Proceed | Safari: Show Details)", (tx, 265),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (148, 163, 184), 1)
        cv2.putText(card, "4. Bam 'Cho phep Camera' -> Video tu dong hien tai day", (tx, 305),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (226, 232, 240), 1)

        cv2.putText(card, f"URL: {self.url[:52]}", (tx, 370),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (52, 211, 153), 1)
        if len(self.url) > 52:
            cv2.putText(card, f"     {self.url[52:]}", (tx, 395),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (52, 211, 153), 1)

        cv2.putText(card, "Dang cho ket noi tu dien thoai...  |  [0-3] Doi Webcam  |  [Q] Huy",
                    (tx, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (148, 163, 184), 1)
        return card

    def print_connection_info(self) -> None:
        print("\n=== Ket noi Camera Dien thoai (Wi-Fi LAN) ===")
        print(f"  Dia chi LAN : {self.local_ip}:{self.port}")
        print(f"  Session URL : {self.url}")
        if self.use_ssl:
            print("  Luu y HTTPS : Trinh duyet se canh bao chung chi tu ky lan dau -> bam 'Nang cao / Tiep tuc' de mo Camera.")
        print("  Firewall    : Neu dien thoai quay mai khong vao duoc, hay mo cong TCP "
              f"{self.port} (Private network) trong Windows Defender Firewall.")
        print(render_qr_terminal(self.url))

    def wait_for_connection(
        self,
        timeout: float | None = None,
        show_window: bool = True,
        window_name: str = "Mobile Camera - Scan QR",
    ) -> bool:
        """Doi den khi nhan duoc frame dau tien tu dien thoai."""
        self.print_connection_info()
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while self.isOpened():
            if self.has_frame:
                if show_window:
                    try:
                        cv2.destroyWindow(window_name)
                    except cv2.error:
                        pass
                return True
            if show_window:
                cv2.imshow(window_name, self.get_qr_card())
                if (cv2.waitKey(30) & 0xFF) in (ord("q"), ord("Q"), 27):
                    return False
            else:
                time.sleep(0.05)
            if deadline is not None and time.monotonic() >= deadline:
                return False
        return False
