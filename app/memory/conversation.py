from __future__ import annotations

from collections import deque


class ConversationMemory:
    def __init__(self, max_turns: int = 12) -> None:
        self._turns: deque[tuple[str, str]] = deque(maxlen=max_turns)

    def add(self, user: str, assistant: str) -> None:
        self._turns.append((user, assistant))

    def as_text(self) -> str:
        lines: list[str] = []
        for u, a in self._turns:
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {a}")
        return "\n".join(lines)
