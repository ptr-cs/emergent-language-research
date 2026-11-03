"""Separate loopback dashboard and authenticated, optional LAN peer listener."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import select
import socket
import threading
import traceback
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .config import DEFAULTS, DEFAULT_GENERATION_TIMEOUT, INFERENCE_KEYS, TOPIC_PRESETS, validate_config, validate_url
from .engine import ACTIVE_STATUSES, Engine
from .ollama import OllamaClient, PeerClient


MAX_BODY = 1024 * 1024


class LabServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        # Unexpected errors belong in the local terminal, without request headers.
        import sys
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            traceback.print_exc()


def _peer_token(data_dir):
    path = data_dir / "peer-token"
    try:
        token = path.read_text(encoding="ascii").strip()
        if len(token) < 32 or len(token) > 256 or not token.isascii() or not token.isprintable():
            raise ValueError("Invalid connection token file. Remove data/peer-token to regenerate it.")
    except FileNotFoundError:
        token = secrets.token_urlsafe(32)
        import os
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as file:
            file.write(token + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return token


class LabApp:
    def __init__(self, data_dir, model="gpt-oss:20b", ollama_url="http://127.0.0.1:11434",
                 port=8765, peer_listen=None, peer_port=8766, generation_timeout=DEFAULT_GENERATION_TIMEOUT):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.port = port
        self.peer_listen = peer_listen
        self.peer_port = peer_port
        self.token = _peer_token(self.data_dir)
        self.csrf = secrets.token_urlsafe(32)
        self.ollama = OllamaClient(ollama_url, generation_timeout)
        self.peer_client = PeerClient(generation_timeout)
        self.inference_lock = threading.Lock()
        self.control_lock = threading.RLock()
        self.activity_lock = threading.Lock()
        self.peer_activity = deque(maxlen=100)
        self.peer_active = None
        self.incoming_cancel = threading.Event()
        self.engine = Engine(self.data_dir, self.generate_local, self.peer_client.generate)
        self.dashboard_server = None
        self.peer_server = None

    def generate_local(self, messages, config, on_delta, cancel):
        if not self.inference_lock.acquire(blocking=False):
            raise RuntimeError("This computer is already generating for another run. Wait for it to finish.")
        try:
            return self.ollama.generate(messages, config, on_delta, cancel)
        finally:
            self.inference_lock.release()

    def state(self):
        state = self.engine.snapshot()
        with self.activity_lock:
            state["peer_activity"] = list(self.peer_activity)
            state["peer_active"] = dict(self.peer_active) if self.peer_active else None
        return state

    def status(self):
        addresses = []
        if self.peer_listen:
            if self.peer_listen not in {"0.0.0.0", "127.0.0.1", "localhost"}:
                addresses.append(f"http://{self.peer_listen}:{self.peer_port}")
            try:
                for entry in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                    ip = entry[4][0]
                    if not ip.startswith("127."):
                        addresses.append(f"http://{ip}:{self.peer_port}")
            except OSError:
                pass
            # Hostname discovery may omit Wi-Fi; no outbound network probe is needed.
            if not addresses:
                addresses.append(f"http://<this-computer-LAN-IP>:{self.peer_port}")
        return {
            "version": __version__, "ollama": self.ollama.status(self.model),
            "peer": {"enabled": bool(self.peer_listen), "listen": self.peer_listen,
                     "port": self.peer_port, "token": self.token if self.peer_listen else "",
                     "addresses": sorted(set(addresses)), "model": self.model},
            "defaults": {**DEFAULTS, "model": self.model},
            "generation_timeout": self.ollama.timeout,
            "topic_presets": TOPIC_PRESETS,
        }

    def start_servers(self):
        try:
            self.dashboard_server = LabServer(("127.0.0.1", self.port), self.dashboard_handler())
            # Useful for model-free tests that request port 0.
            self.port = self.dashboard_server.server_address[1]
            if self.peer_listen:
                self.peer_server = LabServer((self.peer_listen, self.peer_port), self.peer_handler())
                self.peer_port = self.peer_server.server_address[1]
        except OSError:
            if self.dashboard_server:
                self.dashboard_server.server_close()
            raise
        for server in (self.dashboard_server, self.peer_server):
            if server:
                threading.Thread(target=server.serve_forever, daemon=True).start()

    def shutdown(self):
        self.incoming_cancel.set()
        self.engine.shutdown()
        for server in (self.dashboard_server, self.peer_server):
            if server:
                server.shutdown()
                server.server_close()

    def dashboard_handler(self):
        app = self
        static = Path(__file__).resolve().parent.parent / "static"

        class DashboardHandler(_Handler):
            def authorize(self, post=False):
                allowed = {f"127.0.0.1:{app.port}", f"localhost:{app.port}"}
                if self.headers.get("Host", "") not in allowed:
                    self.send_json({"error": "Dashboard is available only through localhost."}, 403)
                    return False
                origin = self.headers.get("Origin")
                if origin and origin not in {"http://" + host for host in allowed}:
                    self.send_json({"error": "Cross-origin dashboard access is forbidden."}, 403)
                    return False
                if post and not secrets.compare_digest(self.headers.get("X-Lab-Token", "").encode(), app.csrf.encode()):
                    self.send_json({"error": "Refresh the dashboard before changing a run."}, 403)
                    return False
                return True

            def do_GET(self):
                if not self.authorize():
                    return
                path = urlsplit(self.path).path
                try:
                    if path == "/api/bootstrap":
                        return self.send_json({"csrf_token": app.csrf})
                    if path == "/api/status":
                        return self.send_json(app.status())
                    if path == "/api/state":
                        return self.send_json(app.state())
                    if path == "/api/runs":
                        return self.send_json({"runs": app.engine.list_runs()})
                    if path.startswith("/api/runs/"):
                        parts = path.strip("/").split("/")
                        if len(parts) == 4 and parts[3] == "export":
                            fmt = parse_qs(urlsplit(self.path).query).get("format", ["json"])[0]
                            body, mime, filename = app.engine.export(parts[2], fmt)
                            return self.send_bytes(body, mime, attachment=filename)
                        if len(parts) == 3:
                            return self.send_json({"session": app.engine.get_run(parts[2], include_prompts=False)})
                    if path in {"/", "/index.html", "/style.css", "/app.js"}:
                        file = static / ("index.html" if path == "/" else path.lstrip("/"))
                        return self.send_bytes(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
                    self.send_json({"error": "Not found."}, 404)
                except (ValueError, FileNotFoundError) as exc:
                    self.send_json({"error": str(exc)}, 404)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                if not self.authorize(post=True):
                    return
                try:
                    payload = self.read_json()
                    path = urlsplit(self.path).path
                    if path == "/api/start":
                        config = validate_config(payload, app.model)
                        with app.control_lock:
                            if app.inference_lock.locked():
                                return self.send_json({"error": "This computer is currently generating. Wait for it to finish."}, 409)
                            active = app.engine.snapshot().get("session")
                            if active and active["status"] in ACTIVE_STATUSES:
                                return self.send_json({"error": "Stop the current run before starting another."}, 409)
                            app.ollama.ensure_model(config["model"])
                            if config["mode"] == "network":
                                health = app.peer_client.check(config["peer_url"], config["peer_token"])
                                if health.get("protocol") != 1 or not health.get("ok"):
                                    raise ValueError(health.get("error", "Peer is not ready or uses an incompatible protocol."))
                                # The worker chooses its own local model. Record it for reproducibility.
                                config["_peer_model"] = health.get("model")
                            return self.send_json(app.engine.start(config))
                    if path == "/api/check-peer":
                        url = validate_url(payload.get("peer_url", ""))
                        token = payload.get("peer_token", "")
                        if not isinstance(token, str) or not token or len(token) > 256 or not token.isascii() or not token.isprintable():
                            raise ValueError("Enter a valid peer connection token.")
                        return self.send_json(app.peer_client.check(url, token))
                    if path in {"/api/pause", "/api/resume", "/api/stop"}:
                        return self.send_json(getattr(app.engine, path.split("/")[-1])())
                    self.send_json({"error": "Not found."}, 404)
                except (ValueError, TypeError, KeyError) as exc:
                    self.send_json({"error": str(exc)}, 400)
                except RuntimeError as exc:
                    self.send_json({"error": str(exc)}, 502)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        return DashboardHandler

    def peer_handler(self):
        app = self

        class PeerHandler(_Handler):
            def authorize(self):
                if not secrets.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + app.token).encode()):
                    self.send_json({"error": "Incorrect peer token."}, 401)
                    return False
                return True

            def do_GET(self):
                if not self.authorize():
                    return
                if self.path != "/api/peer/health":
                    return self.send_json({"error": "Peer endpoints only; open the dashboard on this computer's localhost."}, 404)
                status = app.ollama.status(app.model)
                self.send_json({"ok": status["ok"] and status["model_ready"], "protocol": 1,
                                "version": __version__, "model": app.model,
                                "models": status["models"], "error": status.get("error") or
                                (None if status["model_ready"] else f"Model {app.model} is not downloaded.")})

            def do_POST(self):
                if not self.authorize():
                    return
                if self.path != "/api/peer/generate":
                    return self.send_json({"error": "Not found."}, 404)
                acquired = False
                stream_started = False
                watcher_done = threading.Event()
                try:
                    payload = self.read_json()
                    if payload.get("protocol") != 1:
                        raise ValueError("Unsupported peer protocol. Both computers need the same app version.")
                    messages = payload.get("messages")
                    if not isinstance(messages, list) or not 1 <= len(messages) <= 250:
                        raise ValueError("Invalid prompt messages.")
                    for message in messages:
                        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                            or message["role"] not in {"system", "user", "assistant"}
                            or not isinstance(message["content"], str) or len(message["content"]) > 100000):
                            raise ValueError("Invalid prompt message.")
                    options = payload.get("config", {})
                    if not isinstance(options, dict) or set(options) - set(INFERENCE_KEYS):
                        raise ValueError("Unsupported peer inference settings.")
                    config = validate_config({**options, "model": app.model})
                    context = payload.get("context", {})
                    if not isinstance(context, dict):
                        raise ValueError("Invalid peer context.")
                    context = {k: v for k, v in context.items() if k in {"session_id", "speaker", "role", "turn", "round"}
                               and (v is None or isinstance(v, (str, int)))}
                    with app.control_lock:
                        local = app.engine.snapshot().get("session")
                        if local and local["status"] in ACTIVE_STATUSES:
                            return self.send_json({"error": "The peer has a local run active. Stop that run before connecting."}, 409)
                        acquired = app.inference_lock.acquire(blocking=False)
                        if not acquired:
                            return self.send_json({"error": "The peer is already generating for another coordinator."}, 409)
                    app.ollama.ensure_model(app.model)
                    cancel = threading.Event()
                    with app.activity_lock:
                        app.peer_active = {**context, "content": "", "phase": "generating", "model": app.model}
                    self.send_response(200)
                    self.common_headers("application/x-ndjson")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    stream_started = True

                    def monitor_disconnect():
                        while not watcher_done.wait(0.2):
                            if app.incoming_cancel.is_set():
                                cancel.set()
                                return
                            try:
                                readable, _, _ = select.select([self.connection], [], [], 0)
                                if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                                    cancel.set()
                                    return
                            except (OSError, ValueError):
                                cancel.set()
                                return

                    threading.Thread(target=monitor_disconnect, daemon=True).start()

                    def delta(text):
                        if cancel.is_set():
                            return
                        with app.activity_lock:
                            app.peer_active["content"] += text
                        self.send_line({"delta": text})

                    result = app.ollama.generate(messages, config, delta, cancel)
                    if not cancel.is_set():
                        self.send_line({"result": result})
                        entry = {"at": datetime.now(timezone.utc).isoformat(), "parent_session": context.get("session_id"),
                                 **context, "model": app.model, "content": result["content"],
                                 "words": len(result["content"].split())}
                        with app.activity_lock:
                            app.peer_activity.append(entry)
                        # Persist only final content and provenance, not remote private prompts or tokens.
                        with (app.data_dir / "peer-activity.jsonl").open("a", encoding="utf-8") as file:
                            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except (ValueError, TypeError, KeyError) as exc:
                    if stream_started:
                        self.send_line({"error": str(exc)})
                    else:
                        self.send_json({"error": str(exc)}, 400)
                except (RuntimeError, OSError) as exc:
                    if stream_started:
                        try:
                            self.send_line({"error": str(exc)})
                        except OSError:
                            pass
                    else:
                        self.send_json({"error": str(exc)}, 502)
                finally:
                    watcher_done.set()
                    if acquired:
                        with app.activity_lock:
                            app.peer_active = None
                        app.inference_lock.release()

        return PeerHandler


class _Handler(BaseHTTPRequestHandler):
    server_version = "EmergentLanguageLab/" + __version__

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, fmt, *args):
        pass

    def common_headers(self, mime):
        self.send_header("Content-Type", mime + ("; charset=utf-8" if "charset=" not in mime and (mime.startswith("text/") or "json" in mime) else ""))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

    def send_bytes(self, body, mime, status=200, attachment=None):
        self.send_response(status)
        self.common_headers(mime)
        self.send_header("Content-Length", str(len(body)))
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=200):
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode(), "application/json", status)

    def send_line(self, data):
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode() + b"\n")
        self.wfile.flush()

    def read_json(self):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked request bodies are not supported.")
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise ValueError("Send Content-Type: application/json.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid content length.") from exc
        if not 0 < length <= MAX_BODY:
            raise ValueError("Request must be a nonempty JSON body smaller than 1 MiB.")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("Request must be a JSON object.")
        return data
