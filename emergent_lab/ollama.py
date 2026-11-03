"""Ollama native chat API; only final content crosses the communication channel."""

from __future__ import annotations

import threading
import time

from .config import DEFAULT_GENERATION_TIMEOUT, validate_url
from .transport import request_json, stream_json


class OllamaClient:
    def __init__(self, base_url="http://127.0.0.1:11434", timeout=DEFAULT_GENERATION_TIMEOUT):
        self.base_url = validate_url(base_url)
        self.timeout = timeout
        self._models = []
        self._checked = 0
        self._lock = threading.Lock()

    def models(self, refresh=False):
        with self._lock:
            if refresh or time.monotonic() - self._checked > 15:
                data = request_json(self.base_url, "/api/tags")
                self._models = [m for m in data.get("models", []) if
                                not m.get("remote_host") and not m.get("remote_model")
                                and "cloud" not in m.get("name", "").lower()]
                self._checked = time.monotonic()
            return list(self._models)

    def ensure_model(self, model):
        models = self.models()
        if model not in {m.get("name") for m in models}:
            raise ValueError(f"Local model '{model}' is unavailable. Check 'ollama list'. The app never downloads models automatically.")

    def status(self, model):
        try:
            models = self.models(refresh=True)
            return {"ok": True, "models": models,
                    "model_ready": model in {m.get("name") for m in models},
                    "url": self.base_url}
        except (RuntimeError, ValueError) as exc:
            return {"ok": False, "models": [], "model_ready": False,
                    "url": self.base_url, "error": str(exc)}

    def generate(self, messages, config, on_delta, cancel):
        self.ensure_model(config["model"])
        body = {
            "model": config["model"], "messages": messages, "stream": True,
            "think": config["think"], "keep_alive": "10m",
            "options": {key: config[key] for key in
                        ("temperature", "top_p", "seed", "num_ctx", "num_predict")},
        }
        result = {"content": "", "model": config["model"]}
        complete = False

        def chunk(data):
            nonlocal complete
            # Do not forward or retain message.thinking, tool calls, or other fields.
            content = data.get("message", {}).get("content", "")
            if content:
                if not isinstance(content, str):
                    raise RuntimeError("Ollama returned non-text content.")
                result["content"] += content
                on_delta(content)
            if data.get("done"):
                complete = True
                for key in ("model", "done_reason", "eval_count", "prompt_eval_count",
                            "total_duration", "load_duration", "eval_duration"):
                    if key in data:
                        result[key] = data[key]

        stream_json(self.base_url, "/api/chat", body, chunk, cancel, timeout=self.timeout)
        if not complete:
            raise RuntimeError("Ollama disconnected before completing the response.")
        if not result["content"].strip():
            raise RuntimeError("The model produced no final message. Increase generation tokens (reasoning shares the token budget) or use low reasoning.")
        return result


class PeerClient:
    def __init__(self, timeout=DEFAULT_GENERATION_TIMEOUT):
        self.timeout = timeout

    def check(self, url, token):
        return request_json(url, "/api/peer/health", headers={"Authorization": "Bearer " + token}, timeout=8)

    def generate(self, messages, config, on_delta, cancel):
        from .config import INFERENCE_KEYS
        body = {"protocol": 1, "messages": messages,
                "config": {key: config[key] for key in INFERENCE_KEYS},
                "context": {key.lstrip("_"): config.get(key) for key in
                            ("_session_id", "_speaker", "_role", "_turn", "_round")}}
        result = None

        def chunk(data):
            nonlocal result
            if "delta" in data:
                on_delta(data["delta"])
            if "result" in data:
                result = data["result"]

        stream_json(config["peer_url"], "/api/peer/generate", body, chunk, cancel,
                    headers={"Authorization": "Bearer " + config["peer_token"]}, timeout=self.timeout + 15)
        if not isinstance(result, dict) or not isinstance(result.get("content"), str):
            raise RuntimeError("The peer disconnected before completing the response.")
        return result
