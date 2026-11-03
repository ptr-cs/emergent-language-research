import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock

from emergent_lab.server import LabApp


class ServerTests(unittest.TestCase):
    """Exercise real loopback HTTP without requiring Ollama or a model."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.apps = []

    def tearDown(self):
        for app in reversed(self.apps):
            app.shutdown()
        self.temporary.cleanup()

    def app(self, model="gpt-oss:20b", peer=True):
        instance = LabApp(Path(self.temporary.name) / str(len(self.apps)), model=model,
                          port=0, peer_listen="127.0.0.1" if peer else None, peer_port=0)
        self.apps.append(instance)
        instance.ollama.ensure_model = Mock()
        instance.ollama.status = Mock(return_value={
            "ok": True, "model_ready": True, "models": [{"name": model}],
            "url": "http://127.0.0.1:11434",
        })

        def generate(messages, config, on_delta, cancel):
            on_delta("Hello from ")
            on_delta(model)
            return {"content": "Hello from " + model, "model": model,
                    "eval_count": 5, "done_reason": "stop"}

        instance.ollama.generate = Mock(side_effect=generate)
        instance.start_servers()
        return instance

    def request(self, app, path, body=None, *, peer=False, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", app.peer_port if peer else app.port, timeout=5)
        try:
            encoded = None if body is None else json.dumps(body).encode()
            outgoing = {"Content-Type": "application/json", **(headers or {})}
            connection.request("GET" if body is None else "POST", path, encoded, outgoing)
            response = connection.getresponse()
            raw = response.read()
            return response.status, dict(response.getheaders()), raw
        finally:
            connection.close()

    def authenticated(self, app, path, body):
        return self.request(app, path, body, headers={"X-Lab-Token": app.csrf})

    def wait_completed(self, app, seconds=3):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            status, _, body = self.request(app, "/api/state")
            self.assertEqual(status, 200)
            session = json.loads(body)["session"]
            if session and session["status"] in {"completed", "error", "stopped"}:
                self.assertEqual(session["status"], "completed", session.get("error"))
                return session
            threading.Event().wait(0.01)
        self.fail("HTTP-coordinated run did not complete")

    def test_bootstrap_csrf_and_static_files(self):
        app = self.app()
        status, headers, body = self.request(app, "/api/bootstrap")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["csrf_token"], app.csrf)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertEqual(self.request(app, "/api/pause", {})[0], 403)
        self.assertEqual(self.request(app, "/api/pause", {}, headers={"X-Lab-Token": "wrong"})[0], 403)
        self.assertEqual(self.authenticated(app, "/api/pause", {})[0], 200)
        status, _, body = self.request(app, "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Emergent Language", body)
        self.assertEqual(self.request(app, "/../README.md")[0], 404)
        self.assertEqual(self.request(app, "/api/peer/health")[0], 404)

    def test_untrusted_host_and_cross_origin_are_rejected(self):
        app = self.app()
        for headers in [{"Host": "attacker.example"}, {"Origin": "https://attacker.example"}]:
            with self.subTest(headers=headers):
                self.assertEqual(self.request(app, "/api/status", headers=headers)[0], 403)
        self.assertEqual(self.request(app, "/api/bootstrap", headers={"Origin": f"http://localhost:{app.port}"})[0], 200)
        self.assertEqual(self.request(app, "/api/pause", {}, headers={
            "Origin": "http://attacker.example", "X-Lab-Token": app.csrf})[0], 403)

    def test_peer_listener_requires_token_and_never_serves_dashboard(self):
        app = self.app()
        auth = {"Authorization": "Bearer " + app.token}
        self.assertEqual(self.request(app, "/api/peer/health", peer=True)[0], 401)
        self.assertEqual(self.request(app, "/", peer=True)[0], 401)
        self.assertEqual(self.request(app, "/", peer=True, headers=auth)[0], 404)
        self.assertEqual(self.request(app, "/api/status", peer=True, headers=auth)[0], 404)
        status, _, body = self.request(app, "/api/peer/health", peer=True, headers=auth)
        health = json.loads(body)
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])
        self.assertEqual(health["protocol"], 1)
        self.assertNotIn(app.token, body.decode())
        self.assertNotEqual(app.port, app.peer_port)

    def test_authenticated_peer_generation_streams_own_model_and_logs_no_prompts(self):
        app = self.app(model="gpt-oss:20b-32k")
        payload = {"protocol": 1, "messages": [{"role": "user", "content": "PRIVATE_PROMPT_SENTINEL"}],
                   "config": {"think": "low"},
                   "context": {"session_id": "fixture-run", "speaker": "B", "role": "receiver", "turn": 2}}
        status, headers, body = self.request(app, "/api/peer/generate", payload, peer=True,
                                           headers={"Authorization": "Bearer " + app.token})
        chunks = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(status, 200)
        self.assertIn("application/x-ndjson", headers["Content-Type"])
        self.assertEqual("".join(chunk.get("delta", "") for chunk in chunks), "Hello from gpt-oss:20b-32k")
        self.assertEqual(chunks[-1]["result"]["model"], "gpt-oss:20b-32k")
        generated_config = app.ollama.generate.call_args.args[1]
        self.assertEqual(generated_config["model"], "gpt-oss:20b-32k")
        deadline = time.monotonic() + 2
        while not app.peer_activity and time.monotonic() < deadline:
            threading.Event().wait(0.005)
        self.assertEqual(app.peer_activity[-1]["parent_session"], "fixture-run")
        log = (app.data_dir / "peer-activity.jsonl").read_text()
        self.assertNotIn("PRIVATE_PROMPT_SENTINEL", log)
        self.assertNotIn(app.token, log)
        invalid = {**payload, "config": {"model": "unwanted-model"}}
        self.assertEqual(self.request(app, "/api/peer/generate", invalid, peer=True,
                                     headers={"Authorization": "Bearer " + app.token})[0], 400)

    def test_two_apps_exchange_turns_over_real_http_and_archive_run(self):
        coordinator = self.app(peer=False)
        worker = self.app(model="gpt-oss:20b-32k")
        settings = {"mode": "network", "experiment": "dialogue", "max_turns": 4,
                    "peer_url": f"http://127.0.0.1:{worker.peer_port}", "peer_token": worker.token,
                    "compress": False, "message_words": 80}
        status, _, body = self.authenticated(coordinator, "/api/check-peer", {
            "peer_url": settings["peer_url"], "peer_token": worker.token})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])
        status, _, body = self.authenticated(coordinator, "/api/start", settings)
        self.assertEqual(status, 200, body)
        session = self.wait_completed(coordinator)
        self.assertEqual([turn["speaker"] for turn in session["turns"]], ["A", "B", "A", "B"])
        self.assertEqual([turn["generation"]["endpoint"] for turn in session["turns"]],
                         ["local", "peer", "local", "peer"])
        self.assertEqual(coordinator.ollama.generate.call_count, 2)
        self.assertEqual(worker.ollama.generate.call_count, 2)
        self.assertEqual(session["turns"][1]["generation"]["model"], worker.model)
        self.assertIn("Hello from gpt-oss:20b-32k", json.dumps(coordinator.ollama.generate.call_args.args[0]))
        status, _, body = self.request(coordinator, f"/api/runs/{session['id']}")
        self.assertEqual(status, 200)
        archived = json.loads(body)["session"]
        self.assertNotIn("prompts", archived["turns"][0])
        status, headers, body = self.request(coordinator, f"/api/runs/{session['id']}/export?format=json")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertIn("prompts", json.loads(body)["turns"][0])
        self.assertNotIn(worker.token, body.decode())


if __name__ == "__main__":
    unittest.main()
