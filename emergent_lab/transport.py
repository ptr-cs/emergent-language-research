"""Bounded HTTP JSON/NDJSON, without dependencies or environment proxy routing."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from urllib.parse import urlsplit

from .config import validate_url


class GenerationCancelled(RuntimeError):
    pass


def _connect(base: str, timeout: float):
    parsed = urlsplit(validate_url(base))
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    return cls(parsed.hostname, parsed.port, timeout=timeout)


def _error(response):
    body = response.read(8192).decode("utf-8", errors="replace")
    try:
        data = json.loads(body)
        message = data.get("error", body)
        if isinstance(message, dict):
            message = message.get("message", str(message))
    except (ValueError, AttributeError):
        message = body
    return RuntimeError(f"HTTP {response.status}: {str(message)[:1000]}")


def request_json(base: str, path: str, body=None, headers=None, timeout=5):
    conn = _connect(base, timeout)
    try:
        encoded = None if body is None else json.dumps(body).encode()
        conn.request("GET" if body is None else "POST", path, encoded,
                     {"Content-Type": "application/json", **(headers or {})})
        response = conn.getresponse()
        if response.status != 200:
            raise _error(response)
        raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise RuntimeError("Server response was too large.")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise RuntimeError("Server returned an invalid JSON response.")
        return result
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"Cannot connect to {base}: {exc}") from exc
    finally:
        conn.close()


def stream_json(base, path, body, on_chunk, cancel, headers=None, timeout=600):
    conn = _connect(base, min(10, timeout))
    finished = threading.Event()
    expired = threading.Event()
    connection_socket = [None]
    response = None
    deadline = time.monotonic() + timeout

    def interrupt():
        while not finished.wait(0.1):
            if cancel.is_set() or time.monotonic() >= deadline:
                if not cancel.is_set():
                    expired.set()
                # HTTPConnection may relinquish its socket for a close-delimited
                # HTTP/1.0 peer response. Keep the original socket so cancellation
                # also interrupts a reply that has not emitted any content yet.
                sock = connection_socket[0] or conn.sock
                if sock:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                # The reader owns HTTPResponse/connection cleanup. Closing the
                # buffered file here races http.client's chunked-read cleanup.
                return

    watcher = threading.Thread(target=interrupt, daemon=True)
    watcher.start()
    try:
        if cancel.is_set():
            raise GenerationCancelled("Generation stopped.")
        conn.connect()
        connection_socket[0] = conn.sock
        if cancel.is_set():
            raise GenerationCancelled("Generation stopped.")
        conn.sock.settimeout(timeout)
        conn.request("POST", path, json.dumps(body).encode(),
                     {"Content-Type": "application/json", "Connection": "keep-alive", **(headers or {})})
        response = conn.getresponse()
        if response.status != 200:
            raise _error(response)
        total_bytes = 0
        while True:
            if cancel.is_set():
                raise GenerationCancelled("Generation stopped.")
            line = response.readline(1024 * 1024 + 1)
            if not line:
                break
            total_bytes += len(line)
            if len(line) > 1024 * 1024 or total_bytes > 16 * 1024 * 1024:
                raise RuntimeError("Generation exceeded the response size limit.")
            if line.strip():
                chunk = json.loads(line)
                if not isinstance(chunk, dict):
                    raise RuntimeError("Invalid generation chunk.")
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"])[:1000])
                on_chunk(chunk)
        if cancel.is_set():
            raise GenerationCancelled("Generation stopped.")
        if expired.is_set():
            raise RuntimeError(f"Generation timed out after {timeout:g} seconds.")
    except (OSError, http.client.HTTPException, ValueError) as exc:
        if cancel.is_set():
            raise GenerationCancelled("Generation stopped.") from exc
        if expired.is_set():
            raise RuntimeError(f"Generation timed out after {timeout:g} seconds.") from exc
        raise RuntimeError(f"Generation connection failed: {exc}") from exc
    finally:
        finished.set()
        if response:
            response.close()
        conn.close()
        watcher.join(timeout=0.5)
