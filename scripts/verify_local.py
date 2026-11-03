#!/usr/bin/env python3
"""Exercise both modes against an existing local model, without saving test runs.

This launches two local app instances. It tests the complete peer transport but
does not test a second physical computer, its firewall, or its network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emergent_lab.server import LabApp
from emergent_lab.transport import request_json


def wait_run(base, timeout):
    deadline = time.monotonic() + timeout
    count = -1
    while time.monotonic() < deadline:
        session = request_json(base, "/api/state")["session"]
        if len(session["turns"]) != count:
            count = len(session["turns"])
            if count:
                turn = session["turns"][-1]
                print(f"  Turn {count}: Agent {turn['speaker']} ({turn['generation']['endpoint']}) — {turn['content']}", flush=True)
        if session["status"] not in {"running", "paused", "stopping"}:
            if session["status"] != "completed":
                raise RuntimeError(f"Run {session['status']}: {session.get('error')}")
            return session
        time.sleep(0.25)
    raise RuntimeError("Verification run timed out.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    apps = []
    with tempfile.TemporaryDirectory(prefix="emergent-lab-verify-") as directory:
        try:
            coordinator = LabApp(Path(directory) / "coordinator", args.model, args.ollama_url, port=0)
            worker = LabApp(Path(directory) / "worker", args.model, args.ollama_url, port=0,
                            peer_listen="127.0.0.1", peer_port=0)
            for app in (coordinator, worker):
                app.start_servers()
                apps.append(app)
            base = f"http://127.0.0.1:{coordinator.port}"
            headers = {"X-Lab-Token": request_json(base, "/api/bootstrap")["csrf_token"]}
            status = request_json(base, "/api/status")
            if not status["ollama"]["model_ready"]:
                raise RuntimeError(status["ollama"].get("error") or "Local model unavailable.")
            print(f"Local model available: {args.model}", flush=True)
            for mode, turns in (("self", 2), ("network", 4)):
                print(f"Testing {mode} mode ({turns} real model calls)…", flush=True)
                config = {"mode": mode, "experiment": "coordination", "model": args.model, "max_turns": turns,
                          "compress": False, "message_words": 40}
                if mode == "network":
                    config.update(peer_url=f"http://127.0.0.1:{worker.peer_port}", peer_token=worker.token)
                request_json(base, "/api/start", config, headers)
                session = wait_run(base, timeout=coordinator.ollama.timeout * turns + 30)
                assert len(session["turns"]) == turns
                assert len(session["feedback"]) == turns // 2
                archived = request_json(base, "/api/runs/" + session["id"])["session"]
                assert archived["status"] == "completed"
                assert "prompts" not in archived["turns"][0]
                exported = request_json(base, f"/api/runs/{session['id']}/export?format=json")
                assert exported["turns"][0]["prompts"]
                assert worker.token not in json.dumps(exported)
                if mode == "network":
                    assert [t["generation"]["endpoint"] for t in session["turns"]] == ["local", "peer", "peer", "local"]
                    assert len(worker.state()["peer_activity"]) == 2
                print("  Completed: " + json.dumps(session["metrics"]), flush=True)
            print("PASS: real local inference, both modes, role switching, peer transport, archive and exports.", flush=True)
        finally:
            for app in reversed(apps):
                app.shutdown()


if __name__ == "__main__":
    main()
