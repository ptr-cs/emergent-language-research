import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from emergent_lab.engine import Engine
from emergent_lab.experiment import game_for


def config(**changes):
    result = {
        "mode": "self", "experiment": "coordination", "model": "gpt-oss:20b",
        "max_turns": 4, "seed": 42, "message_words": 32, "compress": False,
        "history_window": 12, "peer_token": "PEER_SECRET_SENTINEL",
        "temperature": 0.85, "top_p": 0.95, "num_ctx": 8192,
        "num_predict": 1024, "think": "low",
    }
    result.update(changes)
    return result


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        self.engines = []

    def tearDown(self):
        for engine in self.engines:
            engine.shutdown()
        self.temporary.cleanup()

    def engine(self, local, peer=None):
        instance = Engine(self.data_dir, local, peer or local)
        self.engines.append(instance)
        return instance

    def wait_for(self, engine, predicate, seconds=2):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            session = engine.snapshot()["session"]
            if predicate(session):
                return session
            threading.Event().wait(0.005)
        self.fail(f"State not reached; last state: {engine.snapshot()}")

    @staticmethod
    def generate(messages, runtime, on_delta, cancel):
        if runtime["_role"] == "receiver":
            text = f"Choice: {game_for(runtime, runtime['_round'])['correct_choice']}"
        else:
            text = "Describe the first object and then the second."
        on_delta(text)
        return {"content": text, "model": "fixture-model", "eval_count": 7,
                "thinking": "PRIVATE_REASONING_SENTINEL"}

    def test_network_routes_by_agent_and_exports_private_prompts_without_secrets(self):
        routes = []

        def local(*args):
            routes.append(("local", args[1]["_speaker"], args[1]["_role"]))
            return self.generate(*args)

        def peer(*args):
            routes.append(("peer", args[1]["_speaker"], args[1]["_role"]))
            return self.generate(*args)

        engine = self.engine(local, peer)
        engine.start(config(mode="network"))
        session = self.wait_for(engine, lambda state: state["status"] == "completed")
        self.assertEqual(routes, [("local", "A", "sender"), ("peer", "B", "receiver"),
                                  ("peer", "B", "sender"), ("local", "A", "receiver")])
        self.assertEqual(session["metrics"]["accuracy"], 1)
        self.assertEqual(session["metrics"]["rounds_completed"], 2)
        self.assertNotIn("prompts", session["turns"][0])
        body, mime, filename = engine.export(session["id"])
        record = json.loads(body)
        self.assertIn("prompts", record["turns"][0])
        self.assertNotIn("PEER_SECRET_SENTINEL", body.decode())
        self.assertNotIn("PRIVATE_REASONING_SENTINEL", body.decode())
        self.assertNotIn("peer_token", record["config"])
        self.assertEqual(record["turns"][0]["generation"]["eval_count"], 7)
        self.assertIn("application/json", mime)
        self.assertTrue(filename.endswith(".json"))
        markdown, _, filename = engine.export(session["id"], "markdown")
        self.assertIn(b"Exact model prompts", markdown)
        self.assertNotIn(b"PEER_SECRET_SENTINEL", markdown)
        self.assertTrue(filename.endswith(".md"))

    def test_pause_finishes_current_turn_and_stops_at_boundary(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def blocked(messages, runtime, on_delta, cancel):
            calls.append(runtime["_turn"])
            entered.set()
            self.assertTrue(release.wait(2))
            return self.generate(messages, runtime, on_delta, cancel)

        engine = self.engine(blocked)
        engine.start(config())
        self.assertTrue(entered.wait(2))
        self.assertEqual(engine.pause()["session"]["status"], "paused")
        release.set()
        self.wait_for(engine, lambda state: len(state["turns"]) == 1 and state["active"] is None)
        self.assertEqual(calls, [1])
        self.assertEqual(engine.snapshot()["session"]["status"], "paused")
        engine.resume()
        self.wait_for(engine, lambda state: state["status"] == "completed")
        self.assertEqual(calls, [1, 2, 3, 4])

    def test_stop_discards_late_response_and_rejects_overlapping_runs(self):
        entered, release = threading.Event(), threading.Event()

        def blocked(messages, runtime, on_delta, cancel):
            entered.set()
            release.wait(2)
            on_delta("A late response")
            return {"content": "A late response", "model": "fixture"}

        engine = self.engine(blocked)
        engine.start(config())
        self.assertTrue(entered.wait(2))
        with self.assertRaises(ValueError):
            engine.start(config())
        self.assertEqual(engine.stop()["session"]["status"], "stopping")
        release.set()
        session = self.wait_for(engine, lambda state: state["status"] == "stopped")
        self.assertEqual(session["turns"], [])
        self.assertIsNone(session["active"])

    def test_stream_preserves_spaces_between_chunks(self):
        preview, release = threading.Event(), threading.Event()

        def streamed(messages, runtime, on_delta, cancel):
            on_delta("hello ")
            on_delta("world")
            preview.set()
            release.wait(2)
            return {"content": "hello world", "model": "fixture"}

        engine = self.engine(streamed)
        engine.start(config(experiment="dialogue", max_turns=1))
        self.assertTrue(preview.wait(2))
        try:
            self.assertEqual(engine.snapshot()["session"]["active"]["content"], "hello world")
        finally:
            release.set()
        self.wait_for(engine, lambda state: state["status"] == "completed")

    def test_hard_cap_applies_to_committed_and_relayed_content(self):
        prompts = []

        def verbose(messages, runtime, on_delta, cancel):
            prompts.append(messages)
            return {"content": "one two three four five six seven eight nine ten", "model": "fixture"}

        engine = self.engine(verbose)
        engine.start(config(experiment="dialogue", max_turns=2, message_words=6))
        session = self.wait_for(engine, lambda state: state["status"] == "completed")
        self.assertEqual(session["turns"][0]["content"], "one two three four five six")
        self.assertEqual(session["turns"][0]["raw_words"], 10)
        self.assertTrue(session["turns"][0]["clipped"])
        self.assertEqual(session["metrics"]["budget_violations"], 2)
        self.assertNotIn("seven eight nine ten", json.dumps(prompts[1]))

    def test_transport_errors_are_redacted_in_saved_run(self):
        def broken(messages, runtime, on_delta, cancel):
            raise RuntimeError("authentication failed for " + runtime["peer_token"])

        engine = self.engine(broken)
        engine.start(config())
        session = self.wait_for(engine, lambda state: state["status"] == "error")
        self.assertIn("[redacted]", session["error"])
        self.assertNotIn("PEER_SECRET_SENTINEL", engine.export(session["id"])[0].decode())

    def test_restart_preserves_incomplete_run_as_interrupted(self):
        runs = self.data_dir / "runs"
        runs.mkdir()
        record = {"id": "saved-run", "status": "running", "created_at": "2026-01-01T00:00:00Z",
                  "turns": [], "feedback": [], "config": {}, "metrics": {},
                  "active": {"content": "partial"}}
        (runs / "saved-run.json").write_text(json.dumps(record))
        (runs / "corrupt.json").write_text("not JSON")
        (runs / "non-object.json").write_text("[]")
        engine = self.engine(self.generate)
        self.assertEqual(engine.snapshot()["session"]["status"], "interrupted")
        self.assertIsNone(engine.snapshot()["session"]["active"])
        self.assertEqual(json.loads((runs / "saved-run.json").read_text())["status"], "interrupted")
        self.assertEqual(len(engine.list_runs()), 1)
        with self.assertRaises(ValueError):
            engine.export("../outside")
        with self.assertRaises(FileNotFoundError):
            engine.export("missing-run")


if __name__ == "__main__":
    unittest.main()
