"""Thread-safe conversation coordinator with durable, secret-free run records."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable
import uuid

from .experiment import build_turn, limit_words, measure, score_round, word_count


ACTIVE_STATUSES = {"running", "paused", "stopping"}
GENERATION_FIELDS = {
    "model", "eval_count", "prompt_eval_count", "done_reason", "total_duration",
    "load_duration", "prompt_eval_duration", "eval_duration", "thinking_tokens",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in config.items()
            if "token" not in key.lower() and "secret" not in key.lower() and not key.startswith("_")}


def _valid_record(record: Any, run_id: str) -> bool:
    return (isinstance(record, dict) and record.get("id") == run_id
            and isinstance(record.get("created_at"), str)
            and isinstance(record.get("turns"), list)
            and isinstance(record.get("feedback"), list)
            and isinstance(record.get("config"), dict)
            and isinstance(record.get("metrics"), dict))


class Engine:
    def __init__(self, data_dir: Path, generate_local: Callable, generate_peer: Callable):
        self.data_dir = Path(data_dir)
        self.runs_dir = self.data_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.generate_local = generate_local
        self.generate_peer = generate_peer
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._session: dict[str, Any] | None = None
        self._runtime_config: dict[str, Any] = {}
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._recover()

    def _recover(self) -> None:
        newest: dict[str, Any] | None = None
        for path in self.runs_dir.glob("*.json"):
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
                if not _valid_record(session, path.stem):
                    continue
                if session.get("status") in ACTIVE_STATUSES:
                    session["status"] = "interrupted"
                    session["active"] = None
                    session["updated_at"] = _now()
                    self._write(session)
                if newest is None or session.get("created_at", "") > newest.get("created_at", ""):
                    newest = session
            except (OSError, ValueError, TypeError):
                continue
        self._session = newest

    def _write(self, session: dict[str, Any]) -> None:
        path = self.runs_dir / (session["id"] + ".json")
        temporary = path.with_suffix(".tmp")
        # In-progress streamed text is deliberately transient. Completed turns
        # and their exact prompts are committed atomically after generation.
        payload = dict(session)
        payload["active"] = None
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _save_locked(self) -> None:
        if self._session:
            self._session["updated_at"] = _now()
            self._write(self._session)

    def snapshot(self, include_prompts: bool = False) -> dict[str, Any]:
        with self._lock:
            session = copy.deepcopy(self._session)
        if session and not include_prompts:
            for turn in session["turns"]:
                turn.pop("prompts", None)
        return {"session": session}

    def start(self, config: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            if self._session and self._session["status"] in ACTIVE_STATUSES:
                raise ValueError("A run is already active. Stop it before starting another.")
            if self._thread and self._thread.is_alive():
                raise ValueError("The previous generation is still stopping. Try again shortly.")
            self._cancel = threading.Event()
            self._runtime_config = copy.deepcopy(config)
            created = _now()
            self._session = {
                "id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10],
                "status": "running",
                "created_at": created,
                "updated_at": created,
                "config": _public_config(config),
                "turns": [],
                "feedback": [],
                "metrics": measure([], []),
                "active": None,
                "error": None,
                "schema_version": 1,
                "experiment_notes": (
                    "Only final English-prompted messages cross the communication channel. "
                    "Prompts encourage English; semantic language compliance is not automatically verified. "
                    "These frozen model calls adapt in context; they do not update model weights. "
                    "Task accuracy and lexical statistics are descriptive, not evidence of a new language."
                ),
            }
            self._save_locked()
            self._thread = threading.Thread(target=self._run, name="conversation-run", daemon=True)
            self._thread.start()
            return self.snapshot()

    def pause(self) -> dict[str, Any]:
        with self._condition:
            if self._session and self._session["status"] == "running":
                self._session["status"] = "paused"
                self._save_locked()
            return self.snapshot()

    def resume(self) -> dict[str, Any]:
        with self._condition:
            if self._session and self._session["status"] == "paused":
                self._session["status"] = "running"
                self._save_locked()
                self._condition.notify_all()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._condition:
            if self._session and self._session["status"] in ACTIVE_STATUSES:
                self._cancel.set()
                self._session["status"] = "stopping"
                self._save_locked()
                self._condition.notify_all()
            return self.snapshot()

    def shutdown(self, timeout: float = 2.0) -> None:
        self.stop()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._session["status"] == "paused" and not self._cancel.is_set():
                        self._condition.wait()
                    if self._cancel.is_set():
                        break
                    session = self._session
                    config = self._runtime_config
                    number = len(session["turns"]) + 1
                    if number > int(config.get("max_turns", 40)):
                        session["status"] = "completed"
                        session["active"] = None
                        self._save_locked()
                        return
                    plan = build_turn(config, session["turns"], session["feedback"], number)
                    session["active"] = {key: plan[key] for key in
                                         ("number", "speaker", "role", "round", "budget_words")}
                    session["active"].update({"content": "", "phase": "generating"})
                    callback_config = dict(config)
                    callback_config.update({
                        "_session_id": session["id"], "_speaker": plan["speaker"],
                        "_role": plan["role"], "_turn": number, "_round": plan["round"],
                    })
                stream_content = ""

                def on_delta(delta: str) -> None:
                    nonlocal stream_content
                    with self._lock:
                        if self._cancel.is_set() or not self._session.get("active"):
                            return
                        active = self._session["active"]
                        # The live preview shows only channel-visible words.
                        stream_content += str(delta)
                        active["content"] = limit_words(stream_content, plan["budget_words"])[0]

                remote = config.get("mode") == "network" and plan["speaker"] == "B"
                generate = self.generate_peer if remote else self.generate_local
                started = time.monotonic()
                result = generate(plan["messages"], callback_config, on_delta, self._cancel)
                elapsed = time.monotonic() - started
                with self._condition:
                    # Stop is cooperative: even if an HTTP request finishes late,
                    # its response is discarded and never becomes a visible turn.
                    if self._cancel.is_set():
                        break
                    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
                        raise ValueError("The model returned an invalid response.")
                    content, clipped, raw_words = limit_words(result["content"], plan["budget_words"])
                    if not content:
                        raise ValueError("The model returned no final message. Increase the generation token limit.")
                    generation = {key: copy.deepcopy(result[key]) for key in GENERATION_FIELDS if key in result}
                    generation["wall_seconds"] = round(elapsed, 3)
                    generation["endpoint"] = "peer" if remote else "local"
                    turn = {key: plan[key] for key in ("number", "speaker", "role", "round", "budget_words")}
                    turn.update({
                        "content": content, "words": word_count(content), "raw_words": raw_words,
                        "clipped": clipped, "created_at": _now(), "generation": generation,
                        "prompts": copy.deepcopy(plan["messages"]),
                        "history_messages_dropped": plan["history_messages_dropped"],
                        "prompt_token_estimate": plan["prompt_token_estimate"],
                        "context_safety_tokens": plan["context_safety_tokens"],
                    })
                    session["turns"].append(turn)
                    if plan["role"] == "receiver":
                        session["feedback"].append(score_round(plan, content))
                    session["metrics"] = measure(session["turns"], session["feedback"])
                    session["active"] = None
                    if len(session["turns"]) >= int(config.get("max_turns", 40)):
                        session["status"] = "completed"
                    self._save_locked()
                    if session["status"] == "completed":
                        return
        except Exception as exc:
            with self._condition:
                if not self._cancel.is_set():
                    self._session["status"] = "error"
                    # Configuration secrets must not be reflected by a transport
                    # exception into persisted run data or the browser.
                    detail = str(exc)
                    token = str(self._runtime_config.get("peer_token", ""))
                    if token:
                        detail = detail.replace(token, "[redacted]")
                    self._session["error"] = detail[:2000]
                    self._session["active"] = None
                    self._save_locked()
                    return
        with self._condition:
            self._session["status"] = "stopped"
            self._session["active"] = None
            self._save_locked()

    def list_runs(self) -> list[dict[str, Any]]:
        summaries = []
        with self._lock:
            for path in self.runs_dir.glob("*.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if not _valid_record(record, path.stem):
                        continue
                    summaries.append({key: record.get(key) for key in
                                      ("id", "status", "created_at", "updated_at", "config", "metrics", "error")})
                except (ValueError, OSError, TypeError):
                    continue
        return sorted(summaries, key=lambda record: record.get("created_at") or "", reverse=True)

    def get_run(self, run_id: str, include_prompts: bool = False) -> dict[str, Any]:
        """Load a run for review without restoring or resuming its execution."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
            raise ValueError("Invalid run identifier.")
        with self._lock:
            path = self.runs_dir / (run_id + ".json")
            if not path.is_file():
                raise FileNotFoundError("Run not found.")
            record = json.loads(path.read_text(encoding="utf-8"))
            if not _valid_record(record, run_id):
                raise ValueError("The run record is damaged or has an unsupported format.")
        if not include_prompts:
            for turn in record["turns"]:
                turn.pop("prompts", None)
        return record

    def export(self, run_id: str, format: str = "json") -> tuple[bytes, str, str]:
        record = self.get_run(run_id, include_prompts=True)
        if format == "json":
            return (json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
                    "application/json; charset=utf-8", run_id + ".json")
        if format not in {"markdown", "md"}:
            raise ValueError("Export format must be json or markdown.")
        lines = ["# Emergent Language Lab conversation", "", f"Run: {run_id}",
                 f"Status: {record['status']}", f"Created: {record['created_at']}", "",
                 "## Configuration", "", "```json", json.dumps(record["config"], indent=2), "```", "",
                 "## Metrics", "", "```json", json.dumps(record["metrics"], indent=2), "```", "",
                 record.get("experiment_notes", ""), "", "## Transcript", ""]
        feedback = {entry["after_turn"]: entry for entry in record["feedback"]}
        for turn in record["turns"]:
            lines.extend([f"### Turn {turn['number']} · Agent {turn['speaker']} · {turn['role']}", "",
                          turn["content"], "", f"Words: {turn['words']}/{turn['budget_words']}" +
                          (" (channel truncated the reply)" if turn.get("clipped") else ""), ""])
            if turn["number"] in feedback:
                lines.extend([feedback[turn["number"]]["content"], ""])
        lines.extend(["## Exact model prompts (private experiment observations)", "",
                      "Private prompts below were logged for human inspection and were never relayed to the other agent.", ""])
        for turn in record["turns"]:
            lines.extend([f"### Turn {turn['number']} · Agent {turn['speaker']}", ""])
            for prompt in turn.get("prompts", []):
                lines.extend([f"**{prompt['role']}**", "", prompt["content"], ""])
        return ("\n".join(lines).encode("utf-8"), "text/markdown; charset=utf-8", run_id + ".md")
