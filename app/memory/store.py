from __future__ import annotations

from threading import Lock

from app.memory.conversation import ConversationMemory
from app.memory.summary import SummaryMemory


class MemoryStore:
    def __init__(self) -> None:
        self.conversation = ConversationMemory()
        self.summary = SummaryMemory()

    def add_turn(self, user: str, assistant: str) -> None:
        self.conversation.add(user, assistant)
        self.summary.update(f"User: {user}\nAssistant: {assistant}")

    def history_text(self) -> str:
        return self.conversation.as_text()

    def summary_text(self) -> str:
        return self.summary.get() if self.summary.is_compressed() else ""


_sessions: dict[str, MemoryStore] = {}
_lock = Lock()


def get_session(session_id: str | None) -> MemoryStore:
    """Return a per-session MemoryStore. Falls back to a shared 'default' bucket."""
    sid = session_id or "default"
    with _lock:
        store = _sessions.get(sid)
        if store is None:
            store = MemoryStore()
            _sessions[sid] = store
        return store


def reset_session(session_id: str | None = None) -> None:
    sid = session_id or "default"
    with _lock:
        _sessions.pop(sid, None)
