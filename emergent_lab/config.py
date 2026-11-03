"""Validated, bounded experiment settings shared by both transports."""

from __future__ import annotations

import math
import re
from urllib.parse import urlsplit


DEFAULT_TOPIC = "Develop a reliable, concise way to cooperate using whatever text-based method is most efficient."
DEFAULT_GENERATION_TIMEOUT = 7200

TOPIC_PRESETS = [
    {"id": "cooperation", "label": "Efficient cooperation", "topic": DEFAULT_TOPIC},
    {"id": "music", "label": "Music", "topic": "Talk about music: explore artists, genres, songs, and what makes music meaningful. Share perspectives and ask each other questions."},
    {"id": "movies", "label": "Movies", "topic": "Talk about movies: discuss memorable films, characters, scenes, and what makes a story work on screen. Compare perspectives and ask each other questions."},
    {"id": "jokes", "label": "Joke-telling", "topic": "Take turns telling original jokes, reacting to each other's humor, and building on the funniest ideas. Keep the exchange playful and conversational."},
    {"id": "flirting", "label": "Flirting", "topic": "Have a playful, witty, respectful flirting conversation between fictional adult characters. Exchange charming banter and respond naturally to each other."},
    {"id": "storytelling", "label": "Collaborative storytelling", "topic": "Build an original story together, taking turns adding characters, events, and dialogue. Incorporate each other's ideas and keep the story coherent."},
    {"id": "philosophy", "label": "Philosophy", "topic": "Explore a philosophical question together, such as what makes a meaningful life or how we know what is true. Offer examples and thoughtfully question each other's ideas."},
    {"id": "science", "label": "Science and discovery", "topic": "Explore fascinating scientific ideas and unanswered questions together. Explain your reasoning, compare hypotheses, and distinguish established evidence from speculation."},
]

DEFAULTS = {
    "mode": "self",
    "experiment": "dialogue",
    "model": "gpt-oss:20b",
    "peer_url": "",
    "peer_token": "",
    "max_turns": 40,
    "temperature": 0.85,
    "top_p": 0.95,
    "seed": 42,
    "num_ctx": 32768,
    "num_predict": 8192,
    "think": "high",
    "history_window": 12,
    "message_words": 80,
    "compress": False,
    "topic": DEFAULT_TOPIC,
    "initial_message": "",
}

INFERENCE_KEYS = ("temperature", "top_p", "seed", "num_ctx", "num_predict", "think")


def validate_url(value: str) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("URL must be a string of at most 2048 characters.")
    try:
        url = urlsplit(value.strip())
        port = url.port
    except ValueError as exc:
        raise ValueError("Invalid server URL.") from exc
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
        or any(c.isspace() or ord(c) < 32 for c in value)
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("Use an http(s) server URL with no credentials, path, or query.")
    return value.strip().rstrip("/")


def validate_config(raw: dict, default_model: str = "gpt-oss:20b") -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Settings must be a JSON object.")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown settings: " + ", ".join(sorted(unknown)))
    config = {**DEFAULTS, "model": default_model, **raw}
    for key, allowed in {
        "mode": {"self", "network"},
        "experiment": {"coordination", "dialogue"},
        "think": {"low", "medium", "high"},
    }.items():
        if not isinstance(config[key], str) or config[key] not in allowed:
            raise ValueError(f"Invalid {key}.")
    for key, low, high in (
        ("max_turns", 2, 2000),
        ("seed", 0, 2147483647),
        ("num_ctx", 4096, 131072),
        ("num_predict", 256, 8192),
        ("history_window", 0, 100),
        ("message_words", 6, 500),
    ):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}.")
    for key, low, high in (("temperature", 0, 2), ("top_p", 0.05, 1)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    if not isinstance(config["compress"], bool):
        raise ValueError("compress must be true or false.")
    for key, limit in (("topic", 2000), ("initial_message", 4000), ("peer_token", 256), ("model", 120), ("peer_url", 2048)):
        if not isinstance(config[key], str) or len(config[key]) > limit:
            raise ValueError(f"{key} must be text of at most {limit} characters.")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]*", config["model"]):
        raise ValueError("Invalid model name.")
    if config["num_predict"] > config["num_ctx"] // 2:
        raise ValueError("Generation tokens must fit within half of the context window.")
    if config["experiment"] == "coordination" and config["max_turns"] % 2:
        raise ValueError("The communication game needs an even number of turns (two per round).")
    if config["mode"] == "network":
        config["peer_url"] = validate_url(config["peer_url"])
        if not config["peer_token"] or any(ord(c) < 33 or ord(c) > 126 for c in config["peer_token"]):
            raise ValueError("Enter the connection token from the second computer's dashboard.")
    return config
