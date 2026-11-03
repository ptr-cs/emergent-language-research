"""Deterministic tasks and actor-specific, English conversation prompts.

Only public final messages are used to build conversation history. Private task
observations and model reasoning are never added to the other actor's history.
"""

from __future__ import annotations

import random
import re
from typing import Any

from .config import DEFAULTS


COLORS = ("red", "blue", "green", "yellow", "purple", "orange")
SHAPES = ("circle", "square", "triangle", "star", "hexagon")
SIZES = ("small", "large")
TEXTURES = ("smooth", "striped", "dotted")
FEATURES = (SIZES, TEXTURES, COLORS, SHAPES)


def word_count(text: str) -> int:
    """The communication channel counts whitespace-separated words."""
    return len(text.split())


def limit_words(text: str, budget: int) -> tuple[str, bool, int]:
    """Preserve original formatting until the channel's hard word boundary."""
    matches = list(re.finditer(r"\S+", text))
    count = len(matches)
    if budget <= 0:
        return "", bool(count), count
    if count <= budget:
        return text.strip(), False, count
    return text[:matches[budget - 1].end()].strip(), True, count


def budget_for(config: dict[str, Any], number: int) -> int:
    initial = int(config.get("message_words", 80))
    if not config.get("compress", False):
        return initial
    if config.get("experiment", "dialogue") == "coordination":
        # Both actors in a round receive exactly the same budget.
        current = (number - 1) // 2
        final = max(1, (int(config.get("max_turns", 40)) - 1) // 2)
    else:
        current = number - 1
        final = max(1, int(config.get("max_turns", 40)) - 1)
    # Six words eventually force abbreviated descriptions of an eight-feature
    # ordered pair; a quarter of a large initial budget would exert no pressure.
    floor = min(initial, 6)
    return max(floor, round(initial - (initial - floor) * current / final))


def _describe(arrangement: tuple[tuple[str, ...], ...]) -> str:
    return "first " + " ".join(arrangement[0]) + "; then " + " ".join(arrangement[1])


def game_for(config: dict[str, Any], round_number: int) -> dict[str, Any]:
    """Produce reproducible close distractors without disclosing answer order."""
    rng = random.Random(int(config.get("seed", 42)) + round_number * 104729)
    target = tuple(tuple(rng.choice(feature) for feature in FEATURES) for _ in range(2))
    alternatives = {target}
    if target[0] != target[1]:
        alternatives.add((target[1], target[0]))
    while len(alternatives) < 4:
        mutable = [list(obj) for obj in target]
        obj_index = rng.randrange(2)
        feature_index = rng.randrange(4)
        mutable[obj_index][feature_index] = rng.choice(FEATURES[feature_index])
        alternatives.add(tuple(tuple(obj) for obj in mutable))
    # Sorting before shuffle avoids hash-randomization differences across hosts.
    candidates = sorted(alternatives)
    rng.shuffle(candidates)
    return {
        "round": round_number,
        "sender": "A" if round_number % 2 else "B",
        "target": _describe(target),
        "candidates": [_describe(candidate) for candidate in candidates],
        "correct_choice": candidates.index(target) + 1,
    }


def _public_history(
    speaker: str,
    turns: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
    window: int,
) -> list[dict[str, str]]:
    selected = turns[-window:] if window else []
    feedback_by_turn = {entry["after_turn"]: entry for entry in feedback}
    history: list[dict[str, str]] = []
    for turn in selected:
        role = "assistant" if turn["speaker"] == speaker else "user"
        content = turn["content"] if role == "assistant" else f"Agent {turn['speaker']}: {turn['content']}"
        history.append({"role": role, "content": content})
        result = feedback_by_turn.get(turn["number"])
        if result:
            history.append({"role": "user", "content": "Public task feedback: " + result["content"]})
    return history


def fit_context(messages: list[dict[str, str]], config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Trim old public history using a conservative, transparent token heuristic.

    Ollama does not expose a preflight tokenizer here. UTF-8 bytes / 2 plus
    per-message overhead usually overestimates English tokens, but is not an
    exact tokenizer or a guarantee for arbitrary user text. Reserve generation
    capacity and an additional 1,024-token safety allowance. The first system
    message and final current-task message are never shortened or discarded.
    """
    capacity = int(config.get("num_ctx", DEFAULTS["num_ctx"]))
    generation = int(config.get("num_predict", DEFAULTS["num_predict"]))
    available = capacity - generation - 1024

    def estimate(items: list[dict[str, str]]) -> int:
        byte_count = sum(len(item["content"].encode("utf-8")) for item in items)
        return (byte_count + 1) // 2 + 8 * len(items) + 4

    fitted = list(messages)
    dropped = 0
    while len(fitted) > 2 and estimate(fitted) > available:
        del fitted[1]
        dropped += 1
    token_estimate = estimate(fitted)
    if token_estimate > available:
        raise ValueError(
            "The system instructions and current task do not fit the conservative context budget "
            f"(estimated {token_estimate} prompt tokens + {generation} generation tokens + "
            f"1024 safety tokens, context {capacity}). Increase the context window, reduce the "
            "generation limit, or shorten the topic/starting message."
        )
    return fitted, {
        "history_messages_dropped": dropped,
        "prompt_token_estimate": token_estimate,
        "context_safety_tokens": 1024,
    }


def build_turn(
    config: dict[str, Any],
    turns: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
    number: int,
) -> dict[str, Any]:
    """Return the next actor's prompt and task metadata.

    The task metadata is coordinator-only and is never passed to a peer model.
    """
    coordination = config.get("experiment", "dialogue") == "coordination"
    round_number = (number + 1) // 2 if coordination else None
    game = game_for(config, round_number) if coordination else None
    if coordination:
        role = "sender" if number % 2 else "receiver"
        sender = game["sender"]
        speaker = sender if role == "sender" else ("B" if sender == "A" else "A")
    else:
        role = "dialogue"
        speaker = "A" if number % 2 else "B"
    budget = budget_for(config, number)
    system = (
        f"You are Agent {speaker}, cooperating with another AI in a conversation experiment. "
        "All conversation must be in English. You may negotiate concise English shorthand, "
        "abbreviations, and reusable conventions when they help you understand each other; "
        "explain new conventions in English. Do not generate an unrelated invented language "
        "or arbitrary gibberish. Respond directly to your partner; do not narrate a simulation "
        "of both agents. Only your final response will reach your partner; private thinking "
        "and private task observations will not. You share only the visible recent transcript. "
        f"The application imposes a HARD cap of {budget} whitespace-separated words on your "
        "entire response, including any labels. Excess words will be discarded. "
    )
    if config.get("compress", False):
        system += "The message budget gradually shrinks; preserve useful conventions in recent exchanges. "
    if coordination:
        system += (
            "Your shared objective is to identify the same ordered arrangement of two objects. "
            "Each object has a size (small/large), texture (smooth/striped/dotted), color "
            "(red/blue/green/yellow/purple/orange), and shape "
            "(circle/square/triangle/star/hexagon). Order and all features matter. "
            "The sender sees the target arrangement; the receiver sees four numbered candidate "
            "arrangements in a private, randomly shuffled order. The sender cannot know the "
            "candidate numbers. Roles alternate each round. Both agents receive public feedback "
            "after the receiver replies. Seek accurate identification using few words. "
            "A correct choice is the shared success signal; this experiment does not update model weights. "
        )
    else:
        system += (
            "Sustain a useful two-way conversation. Ask questions, test your partner's "
            "understanding, and retain useful shared conventions when helpful. "
        )
    topic = str(config.get("topic", ""))
    if topic:
        system += "Shared discussion goal: " + topic + " "
    messages = [{"role": "system", "content": system}]
    messages.extend(_public_history(speaker, turns, feedback, int(config.get("history_window", 12))))
    if coordination and role == "sender":
        instruction = (
            f"Round {round_number}. You are the SENDER. Your private target is: {game['target']}. "
            "Describe this exact ordered arrangement to your partner. They have the numbered "
            "candidates and will select a match. Do not guess candidate numbers. "
            f"Send one English message within {budget} words."
        )
    elif coordination:
        options = "\n".join(f"{i}. {candidate}" for i, candidate in enumerate(game["candidates"], 1))
        # The current sender's message is also repeated here so very small history
        # windows never remove the one observation required to solve this round.
        latest = turns[-1]["content"] if turns else "(no message received)"
        instruction = (
            f"Round {round_number}. You are the RECEIVER. Your partner's current message is:\n"
            f"{latest}\n\nYour private candidate order is:\n{options}\n\n"
            "Select the arrangement your partner described. You may give brief English feedback "
            "or negotiate a convention. End your response with exactly one standalone line "
            f"'Choice: N', replacing N with 1, 2, 3, or 4. The entire response, including Choice, "
            f"must fit within {budget} words. Your partner does not see these numbered candidates."
        )
    else:
        initial = str(config.get("initial_message", "")).strip()
        if number == 1:
            instruction = initial or (
                "Start a natural English conversation about the shared discussion goal above. "
                "Address your partner directly with a thought or question relevant to that goal "
                "and invite their response. Stay on the selected topic."
            )
        else:
            latest = turns[-1]["content"] if turns else "(no message received)"
            partner = turns[-1]["speaker"] if turns else ("B" if speaker == "A" else "A")
            instruction = (
                f"Your partner Agent {partner}'s latest message is:\n{latest}\n\n"
                f"It is your turn, Agent {speaker}. Reply to that message in "
                f"English using at most {budget} words."
            )
    if coordination and number == 1 and config.get("initial_message"):
        instruction += "\nAdditional starting context: " + str(config["initial_message"])
    messages.append({"role": "user", "content": instruction})
    messages, context_info = fit_context(messages, config)
    return {
        "number": number,
        "speaker": speaker,
        "role": role,
        "round": round_number,
        "budget_words": budget,
        "messages": messages,
        "game": game,
        **context_info,
    }


def score_round(plan: dict[str, Any], content: str) -> dict[str, Any]:
    game = plan["game"]
    # Reject multiple choice lines or a non-final choice; no permissive guesswork.
    choice_lines = re.findall(r"(?im)^\s*Choice\s*:", content)
    match = re.search(r"(?im)^\s*Choice:\s*([1-4])\s*[.!]?\s*\Z", content)
    choice = int(match.group(1)) if match and len(choice_lines) == 1 else None
    correct = choice == game["correct_choice"]
    if choice is None:
        detail = "No valid final 'Choice: N' line was received."
    else:
        detail = f"The receiver chose {choice}."
    result = "Success" if correct else "Miss"
    content = (
        f"Round {game['round']}: {result}. {detail} The correct candidate was "
        f"{game['correct_choice']}. Shared task score: {1 if correct else 0}/1. "
        "Roles switch next round. Preserve useful English conventions and adapt if needed."
    )
    return {
        "round": game["round"],
        "after_turn": plan["number"],
        "content": content,
        "correct": correct,
        "choice": choice,
        "correct_choice": game["correct_choice"],
        "budget_words": plan["budget_words"],
    }


def measure(turns: list[dict[str, Any]], feedback: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [word.lower() for turn in turns for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", turn["content"])]
    total_words = sum(turn["words"] for turn in turns)
    successes = sum(bool(entry["correct"]) for entry in feedback)
    texts = [turn["content"].strip().lower() for turn in turns]
    return {
        "turn_count": len(turns),
        "total_words": total_words,
        "mean_words": round(total_words / len(turns), 2) if turns else 0,
        "lexical_diversity": round(len(set(tokens)) / len(tokens), 4) if tokens else 0,
        "rounds_completed": len(feedback),
        "correct_rounds": successes,
        "accuracy": round(successes / len(feedback), 4) if feedback else None,
        "budget_violations": sum(bool(turn.get("clipped")) for turn in turns),
        "repeated_messages": len(texts) - len(set(texts)),
    }
