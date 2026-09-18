"""Unit tests for the Conversation Manager."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from conversation.manager import ConversationManager
from conversation.prompt_templates import build_prompt
from memory.session_memory import SessionMemory


class TestPromptTemplates(unittest.TestCase):
    """Test prompt template generation."""

    def test_build_prompt_basic(self):
        prompt = build_prompt("Hello there")
        self.assertIn("User: Hello there", prompt)
        self.assertIn("Assistant:", prompt)

    def test_build_prompt_with_emotion(self):
        prompt = build_prompt("I'm feeling great", emotion="happy")
        self.assertIn("enthusiastic", prompt.lower())

    def test_build_prompt_with_history(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        prompt = build_prompt("How are you?", history=history)
        self.assertIn("Hi", prompt)
        self.assertIn("Hello!", prompt)


class TestConversationManager(unittest.TestCase):
    """Test conversation manager logic."""

    def setUp(self):
        self.mock_llm = MagicMock()
        self.mock_llm.generate_response.return_value = "I'm doing great!"
        self.memory = SessionMemory(max_history=10)
        self.manager = ConversationManager(
            llm=self.mock_llm,
            memory=self.memory,
        )

    def test_process_turn(self):
        response = self.manager.process_turn("Hello")
        self.assertEqual(response, "I'm doing great!")
        self.mock_llm.generate_response.assert_called_once()

    def test_memory_updated(self):
        self.manager.process_turn("Hello")
        history = self.memory.get_history()
        self.assertEqual(len(history), 2)  # user + assistant
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

    def test_start_stop(self):
        self.manager.start()
        self.assertTrue(self.manager.is_active)
        self.manager.stop()
        self.assertFalse(self.manager.is_active)

    def test_full_turn_inactive(self):
        result = self.manager.full_turn()
        self.assertIsNone(result)


class TestSessionMemory(unittest.TestCase):
    """Test session memory."""

    def setUp(self):
        self.memory = SessionMemory(max_history=5)

    def test_add_and_get(self):
        self.memory.add_message("user", "Hello")
        history = self.memory.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["content"], "Hello")

    def test_max_history(self):
        for i in range(10):
            self.memory.add_message("user", f"Message {i}")
        history = self.memory.get_history()
        self.assertEqual(len(history), 5)

    def test_clear(self):
        self.memory.add_message("user", "Hello")
        self.memory.clear()
        self.assertEqual(len(self.memory.get_history()), 0)


if __name__ == "__main__":
    unittest.main()
