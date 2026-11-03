import json
import threading
import unittest
from unittest.mock import patch

from emergent_lab.config import validate_config, validate_url
from emergent_lab.ollama import OllamaClient, PeerClient


class InferenceTests(unittest.TestCase):
    def test_final_content_is_relayed_without_reasoning(self):
        client = OllamaClient()
        received, calls = [], []

        def stream(base, path, body, callback, cancel, **kwargs):
            calls.append(body)
            callback({"message": {"thinking": "PRIVATE_REASONING"}})
            callback({"message": {"content": "Hello "}})
            callback({"message": {"content": "partner."}, "done": True,
                      "eval_count": 10, "done_reason": "stop"})

        with patch.object(client, "ensure_model"), patch("emergent_lab.ollama.stream_json", stream):
            result = client.generate([{"role": "user", "content": "Talk."}],
                                     validate_config({}), received.append, threading.Event())
        self.assertEqual(received, ["Hello ", "partner."])
        self.assertEqual(result["content"], "Hello partner.")
        self.assertNotIn("PRIVATE_REASONING", json.dumps(result))
        self.assertEqual(calls[0]["think"], "high")
        self.assertEqual(calls[0]["options"]["num_predict"], 8192)

    def test_incomplete_stream_is_not_treated_as_completed_turn(self):
        client = OllamaClient()

        def stream(base, path, body, callback, cancel, **kwargs):
            callback({"message": {"content": "unfinished"}})

        with patch.object(client, "ensure_model"), patch("emergent_lab.ollama.stream_json", stream):
            with self.assertRaisesRegex(RuntimeError, "before completing"):
                client.generate([], validate_config({}), lambda text: None, threading.Event())

    def test_cloud_models_are_excluded_from_available_local_models(self):
        models = [
            {"name": "gpt-oss:20b"},
            {"name": "gpt-oss:120b-cloud"},
            {"name": "remote-aliased-model", "remote_host": "https://example.invalid"},
        ]
        with patch("emergent_lab.ollama.request_json", return_value={"models": models}):
            client = OllamaClient()
            self.assertEqual(client.models(refresh=True), [{"name": "gpt-oss:20b"}])
            client.ensure_model("gpt-oss:20b")
            with self.assertRaises(ValueError):
                client.ensure_model("remote-aliased-model")

    def test_peer_body_excludes_token_and_coordinator_model(self):
        settings = validate_config({"mode": "network", "peer_url": "http://127.0.0.1:8766",
                                    "peer_token": "PRIVATE_TOKEN", "model": "gpt-oss:20b"})
        settings.update({"_speaker": "B", "_role": "receiver", "_turn": 2,
                         "_round": 1, "_session_id": "run-fixture"})
        captured = []

        def stream(base, path, body, callback, cancel, **kwargs):
            captured.append((body, kwargs["headers"]))
            callback({"delta": "Choice: 2"})
            callback({"result": {"content": "Choice: 2", "model": "peer-model"}})

        with patch("emergent_lab.ollama.stream_json", stream):
            result = PeerClient().generate([], settings, lambda text: None, threading.Event())
        body, headers = captured[0]
        self.assertNotIn("PRIVATE_TOKEN", json.dumps(body))
        self.assertNotIn("model", body["config"])
        self.assertEqual(headers["Authorization"], "Bearer PRIVATE_TOKEN")
        self.assertEqual(body["context"]["speaker"], "B")
        self.assertEqual(result["model"], "peer-model")


class ConfigurationTests(unittest.TestCase):
    def test_invalid_numeric_and_boolean_settings_are_rejected(self):
        invalid = [{"temperature": float("nan")}, {"top_p": float("inf")},
                   {"max_turns": True}, {"experiment": "coordination", "max_turns": 3}, {"compress": "false"},
                   {"num_predict": 8192, "num_ctx": 8192}, {"history_window": -1}]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_config(raw)
        self.assertEqual(validate_config({"experiment": "dialogue", "max_turns": 3})["max_turns"], 3)

    def test_urls_reject_embedded_credentials_paths_and_newlines(self):
        for url in ["file:///tmp/model", "http://user:secret@host", "http://host/api",
                    "http://host?token=secret", "http://host#fragment", "http://host\n", "http://host:0"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_url(url)
        self.assertEqual(validate_url("https://lab.example:8766/"), "https://lab.example:8766")

    def test_network_settings_require_token_and_unknown_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({"mode": "network", "peer_url": "http://host:8766"})
        with self.assertRaises(ValueError):
            validate_config({"ollama_url": "http://another-host:11434"})
        self.assertEqual(validate_config({}, default_model="gpt-oss:20b-32k")["model"], "gpt-oss:20b-32k")


if __name__ == "__main__":
    unittest.main()
