"""Ket noi camera dien thoai qua Wi-Fi LAN (HTTPS + WebSocket + QR Code)."""
from __future__ import annotations

import collections
import datetime
import hmac
import ipaddress
import json
from pathlib import Path
import secrets
import socket
import ssl
import threading
import time

import cv2
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from flask import Flask, Response, abort, request
import numpy as np
import simple_websocket
from werkzeug.serving import WSGIRequestHandler, make_server

from nckh_2.core import paths

_VIRTUAL_IFACE_KEYWORDS = (
    "radmin", "vpn", "virtual", "vmware", "vbox", "wsl", "hyper-v",
    "vethernet", "docker", "tailscale", "zerotier", "hamachi", "loopback",
    "teredo", "pseudo", "isatap",
)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_FILE = TEMPLATE_DIR / "mobile_camera.html"


def load_mobile_html() -> str:
    """Doc noi dung file HTML template cho mobile camera."""
    if TEMPLATE_FILE.exists():
        return TEMPLATE_FILE.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Khong tim thay HTML template tai {TEMPLATE_FILE}")


def get_lan_ip(preferred_ip: str | None = None) -> str:
    if preferred_ip:
        return preferred_ip
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


def ensure_ssl_cert(ip_str: str, cert_dir: Path | None = None) -> tuple[Path, Path]:
    if cert_dir is None:
        cert_dir = paths.DATA_DIR / "raw" / "certs"
    cert_dir.mkdir(parents=True, exist_ok=True)

    safe_ip = ip_str.replace(".", "_").replace(":", "_")
    cert_path = cert_dir / f"cert_{safe_ip}.pem"
    key_path = cert_dir / f"key_{safe_ip}.pem"

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
                digital_signature=True, key_encipherment=True, key_cert_sign=True,
                crl_sign=False, content_commitment=False, data_encipherment=False,
                key_agreement=False, encipher_only=False, decipher_only=False,
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


def generate_qr_matrix(url: str) -> np.ndarray:
    encoder = cv2.QRCodeEncoder.create()
    return encoder.encode(url)


def generate_qr_image(url: str, size_px: int = 320, border_px: int = 24) -> np.ndarray:
    matrix = generate_qr_matrix(url)
    qr_scaled = cv2.resize(matrix, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
    bordered = cv2.copyMakeBorder(
        qr_scaled, border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT, value=255,
    )
    return cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)


def render_qr_terminal(url: str) -> str:
    matrix = generate_qr_matrix(url)
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


class _MobileRequestHandler(WSGIRequestHandler):
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


class MobileVideoCapture:
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
            resp = Response(load_mobile_html(), mimetype="text/html; charset=utf-8")
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

    def isOpened(self) -> bool:
        return self._opened and not self._stop_event.is_set()

    def grab(self, timeout: float = 5.0) -> bool:
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
        with self._frame_cond:
            if self._latest_frame is None:
                return False, None
            self._last_read_seq = self._frame_seq
            return True, self._latest_frame.copy()

    def read(self, timeout: float = 5.0) -> tuple[bool, np.ndarray | None]:
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

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def has_frame(self) -> bool:
        with self._frame_cond:
            return self._latest_frame is not None

    def get_live_preview(self) -> np.ndarray | None:
        with self._frame_cond:
            return self._latest_frame

    def pop_recorded_frames(self) -> list[np.ndarray]:
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
