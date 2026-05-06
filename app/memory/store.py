from __future__ import annotations

from threading import Lock

from app.memory.conversation import ConversationMemory
from app.memory.semantic import SemanticMemory
from app.memory.summary import SummaryMemory


class MemoryStore:
    def __init__(self) -> None:
        self.conversation = ConversationMemory()
        self.summary = SummaryMemory()
        self.semantic = SemanticMemory()

    def add_turn(self, user: str, assistant: str) -> None:
        self.conversation.add(user, assistant)
        self.summary.update(f"User: {user}\nAssistant: {assistant}")
        self.semantic.update(user, assistant)

    def history_text(self) -> str:
        return self.conversation.as_text()

    def summary_text(self) -> str:
        """Return episodic summary.

        Returns the compressed summary when available, or the raw accumulated
        text when the session hasn't crossed the compression threshold yet.
        This ensures episodic context is always surfaced, not just after turn ~20.
        """
        return self.summary.get()

    def semantic_text(self) -> str:
        return self.semantic.as_text()

    def snapshot(self) -> dict:
        """Full memory snapshot for /memory API and UI debug panel."""
        return {
            "turn_count": self.semantic._turn_count,
            "history": self.history_text(),
            "episodic_summary": self.summary.get(),
            "episodic_compressed": self.summary.is_compressed(),
            "semantic": self.semantic.as_dict(),
        }


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
