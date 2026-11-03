import json
import unittest

from emergent_lab.experiment import (
    budget_for,
    build_turn,
    fit_context,
    game_for,
    limit_words,
    measure,
    score_round,
)


def config(**changes):
    result = {
        "mode": "self", "experiment": "coordination", "model": "gpt-oss:20b",
        "max_turns": 8, "seed": 42, "message_words": 32, "compress": True,
        "history_window": 12,
    }
    result.update(changes)
    return result


class ExperimentTests(unittest.TestCase):
    def test_tasks_are_repeatable_with_unique_candidates_and_balanced_roles(self):
        settings = config()
        first = [game_for(settings, number) for number in range(1, 20)]
        self.assertEqual(first, [game_for(settings, number) for number in range(1, 20)])
        self.assertNotEqual(first[0], game_for(config(seed=100), 1))
        for number, game in enumerate(first, 1):
            self.assertEqual(len(set(game["candidates"])), 4)
            self.assertEqual(game["candidates"][game["correct_choice"] - 1], game["target"])
            self.assertEqual(game["sender"], "A" if number % 2 else "B")

    def test_compression_is_monotone_and_same_for_both_turns_of_a_round(self):
        settings = config(max_turns=20, message_words=32)
        budgets = [budget_for(settings, number) for number in range(1, 21)]
        self.assertEqual(budgets[0], 32)
        self.assertEqual(budgets[-1], 6)
        self.assertEqual(budgets, sorted(budgets, reverse=True))
        for offset in range(0, 20, 2):
            self.assertEqual(budgets[offset], budgets[offset + 1])
        self.assertEqual([budget_for(config(compress=False), n) for n in range(1, 9)], [32] * 8)

    def test_hard_budget_preserves_format_and_records_unclipped_word_count(self):
        content, clipped, raw_words = limit_words("  red\nblue   green\nyellow ", 3)
        self.assertEqual(content, "red\nblue   green")
        self.assertTrue(clipped)
        self.assertEqual(raw_words, 4)
        self.assertEqual(limit_words("Choice: 2\n", 3), ("Choice: 2", False, 2))

    def test_only_public_messages_enter_other_agent_history(self):
        settings = config()
        sender = build_turn(settings, [], [], 1)
        self.assertIn(sender["game"]["target"], sender["messages"][-1]["content"])
        self.assertNotIn("private candidate order", sender["messages"][-1]["content"])
        turns = [{
            "number": 1, "speaker": "A", "content": "PUBLIC_MESSAGE_SENTINEL",
            "prompts": [{"role": "user", "content": "PRIVATE_OBSERVATION_SENTINEL"}],
            "thinking": "PRIVATE_THINKING_SENTINEL",
        }]
        receiver = build_turn(settings, turns, [], 2)
        serialized = json.dumps(receiver["messages"])
        self.assertIn("PUBLIC_MESSAGE_SENTINEL", serialized)
        self.assertNotIn("PRIVATE_OBSERVATION_SENTINEL", serialized)
        self.assertNotIn("PRIVATE_THINKING_SENTINEL", serialized)
        self.assertNotIn("correct_choice", serialized)
        self.assertIn("private candidate order", receiver["messages"][-1]["content"])
        self.assertEqual(len(receiver["game"]["candidates"]), 4)

    def test_current_message_survives_zero_history_window(self):
        turns = [{"number": 1, "speaker": "A", "content": "CURRENT_MESSAGE"}]
        receiver = build_turn(config(history_window=0), turns, [], 2)
        self.assertIn("CURRENT_MESSAGE", receiver["messages"][-1]["content"])
        self.assertEqual(len(receiver["messages"]), 2)

    def test_dialogue_preserves_latest_partner_message_without_history(self):
        turns = [{"number": 1, "speaker": "A", "content": "CURRENT_DIALOGUE_MESSAGE"}]
        plan = build_turn(config(history_window=0, experiment="dialogue"), turns, [], 2)
        self.assertIn("CURRENT_DIALOGUE_MESSAGE", plan["messages"][-1]["content"])
        self.assertEqual(len(plan["messages"]), 2)

    def test_context_trimming_removes_oldest_history_and_preserves_current_task(self):
        messages = [
            {"role": "system", "content": "SYSTEM_INSTRUCTIONS"},
            {"role": "user", "content": "OLD_HISTORY " * 2000},
            {"role": "assistant", "content": "RECENT_HISTORY"},
            {"role": "user", "content": "CURRENT_TASK"},
        ]
        fitted, metadata = fit_context(messages, config(num_ctx=4096, num_predict=1024))
        self.assertEqual(fitted, [messages[0], messages[2], messages[3]])
        self.assertEqual(metadata["history_messages_dropped"], 1)
        self.assertLessEqual(metadata["prompt_token_estimate"] + metadata["context_safety_tokens"] + 1024, 4096)
        self.assertEqual(len(messages), 4)

    def test_context_refuses_oversized_current_task_instead_of_dropping_it(self):
        messages = [{"role": "system", "content": "English instructions"},
                    {"role": "user", "content": "VERY_LARGE_CURRENT_TASK " * 1000}]
        with self.assertRaisesRegex(ValueError, "do not fit"):
            fit_context(messages, config(num_ctx=4096, num_predict=1024))

    def test_role_perspective_treats_own_messages_as_assistant(self):
        turns = [
            {"number": 1, "speaker": "A", "content": "A's message"},
            {"number": 2, "speaker": "B", "content": "B's message"},
        ]
        plan = build_turn(config(experiment="dialogue"), turns, [], 3)
        self.assertEqual(plan["speaker"], "A")
        self.assertEqual(plan["messages"][1], {"role": "assistant", "content": "A's message"})
        self.assertEqual(plan["messages"][2], {"role": "user", "content": "Agent B: B's message"})

    def test_scoring_requires_one_final_standalone_choice(self):
        plan = build_turn(config(), [{"number": 1, "speaker": "A", "content": "target"}], [], 2)
        correct = plan["game"]["correct_choice"]
        self.assertTrue(score_round(plan, f"Understood.\nChoice: {correct}")["correct"])
        invalid = [
            f"I choose Choice: {correct}",
            f"Choice: {correct}\nMore explanation after the choice.",
            f"Choice: {correct}\nChoice: {correct}",
            "Choice: 0", "Choice: 5", "Choice: N", "I understand.",
        ]
        for text in invalid:
            with self.subTest(text=text):
                result = score_round(plan, text)
                self.assertFalse(result["correct"])
                self.assertIsNone(result["choice"])
        wrong = correct % 4 + 1
        self.assertFalse(score_round(plan, f"Choice: {wrong}")["correct"])
        clipped = limit_words(f"Too many explanatory words.\nChoice: {correct}", 4)[0]
        self.assertFalse(score_round(plan, clipped)["correct"])

    def test_measurements_are_descriptive_and_no_accuracy_without_rounds(self):
        self.assertIsNone(measure([], [])["accuracy"])
        turns = [{"content": "red red", "words": 2, "clipped": False}] * 2
        metrics = measure(turns, [{"correct": True}, {"correct": False}])
        self.assertEqual(metrics["mean_words"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["repeated_messages"], 1)
        self.assertEqual(metrics["lexical_diversity"], 0.25)


if __name__ == "__main__":
    unittest.main()
