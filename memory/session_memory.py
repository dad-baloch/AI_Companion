"""Session Memory — stores and retrieves conversation history."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SessionMemory:
    """Manages conversation history for the current session with optional persistence."""

    def __init__(self, max_history: int = 50, persist_path: str = None):
        self.max_history = max_history
        self.persist_path = Path(persist_path) if persist_path else None
        self.history: List[Dict[str, str]] = []
        self.session_start = datetime.now().isoformat()
        self.metadata: Dict = {
            "session_start": self.session_start,
            "turns": 0,
        }

        # Load existing session if available
        if self.persist_path and self.persist_path.exists():
            self._load()

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history.

        Args:
            role: 'user' or 'assistant'.
            content: The message text.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(message)
        self.metadata["turns"] += 1

        # Trim old messages if exceeding limit
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Auto-save if persistence is enabled
        if self.persist_path:
            self._save()

        logger.debug(f"[Memory] Added {role} message. Total turns: {self.metadata['turns']}")

    def get_history(self, last_n: int = None) -> List[Dict[str, str]]:
        """Get conversation history.

        Args:
            last_n: Number of recent messages to return. None for all.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        if last_n:
            return self.history[-last_n:]
        return self.history.copy()

    def get_context_window(self, max_tokens_estimate: int = 1000) -> List[Dict[str, str]]:
        """Get messages that fit within an estimated token budget.

        Rough estimate: 1 token ≈ 4 characters.
        """
        result = []
        char_budget = max_tokens_estimate * 4
        total_chars = 0

        for msg in reversed(self.history):
            msg_chars = len(msg["content"])
            if total_chars + msg_chars > char_budget:
                break
            result.insert(0, msg)
            total_chars += msg_chars

        return result

    def summarize(self) -> str:
        """Get a brief summary of the current session."""
        return (
            f"Session started: {self.session_start}\n"
            f"Total turns: {self.metadata['turns']}\n"
            f"Messages in memory: {len(self.history)}"
        )

    def clear(self):
        """Clear all conversation history."""
        self.history.clear()
        self.metadata["turns"] = 0
        logger.info("[Memory] Session history cleared.")

    def _save(self):
        """Save session to disk."""
        data = {
            "metadata": self.metadata,
            "history": self.history,
        }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load session from disk."""
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            self.metadata = data.get("metadata", self.metadata)
            self.history = data.get("history", [])
            logger.info(f"[Memory] Loaded session with {len(self.history)} messages.")
        except Exception as e:
            logger.warning(f"[Memory] Failed to load session: {e}")
