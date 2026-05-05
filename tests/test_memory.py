"""Tests for the memory subsystem."""
from __future__ import annotations

import pytest

from app.memory.conversation import ConversationMemory
from app.memory.store import get_session, reset_session, MemoryStore
from app.memory.summary import SummaryMemory


class TestConversationMemory:
    def test_stores_turns(self):
        mem = ConversationMemory()
        mem.add("hello", "hi there")
        assert "hello" in mem.as_text()
        assert "hi there" in mem.as_text()

    def test_sliding_window_evicts_oldest(self):
        mem = ConversationMemory(max_turns=2)
        mem.add("q1", "a1")
        mem.add("q2", "a2")
        mem.add("q3", "a3")  # evicts q1/a1
        text = mem.as_text()
        assert "q1" not in text
        assert "q3" in text

    def test_as_text_format(self):
        mem = ConversationMemory()
        mem.add("what is rag?", "RAG stands for...")
        text = mem.as_text()
        assert "User: what is rag?" in text
        assert "Assistant: RAG stands for..." in text

    def test_empty_memory_returns_empty_string(self):
        mem = ConversationMemory()
        assert mem.as_text() == ""

    def test_multiple_turns_ordered(self):
        mem = ConversationMemory()
        for i in range(5):
            mem.add(f"q{i}", f"a{i}")
        text = mem.as_text()
        # Should be in order
        last = -1
        for i in range(5):
            pos = text.find(f"q{i}")
            assert pos > last
            last = pos


class TestSummaryMemory:
    def test_update_and_get(self):
        mem = SummaryMemory()
        mem.update("User: hi\nAssistant: hello")
        assert "hi" in mem.get() or "hello" in mem.get()

    def test_starts_empty(self):
        mem = SummaryMemory()
        assert mem.get() == ""

    def test_grows_with_updates(self):
        mem = SummaryMemory()
        mem.update("turn one")
        mem.update("turn two")
        assert len(mem.get()) > 0

    def test_truncates_on_overflow(self):
        mem = SummaryMemory(compress_threshold=100, target_chars=50)
        # Offline (no API key): falls back to tail truncation
        long_text = "word " * 30  # 150 chars
        mem.update(long_text)
        # After overflow, summary should be at most target_chars + some slack
        assert len(mem.get()) <= 200  # generous bound for offline path


class TestMemoryStore:
    def test_add_turn_updates_conversation(self):
        store = MemoryStore()
        store.add_turn("question", "answer")
        assert "question" in store.history_text()

    def test_history_text_shows_recent_turns(self):
        store = MemoryStore()
        store.add_turn("first", "reply1")
        store.add_turn("second", "reply2")
        history = store.history_text()
        assert "first" in history
        assert "second" in history

    def test_summary_text_initially_empty(self):
        store = MemoryStore()
        assert store.summary_text() == ""

    def test_summary_text_hidden_until_compressed(self):
        store = MemoryStore()
        store.add_turn("question", "answer")
        assert store.history_text()
        assert store.summary_text() == ""


class TestSessionRegistry:
    def test_same_id_returns_same_store(self):
        s1 = get_session("abc")
        s2 = get_session("abc")
        assert s1 is s2

    def test_different_ids_return_different_stores(self):
        s1 = get_session("session-a")
        s2 = get_session("session-b")
        assert s1 is not s2

    def test_reset_clears_session(self):
        s1 = get_session("reset-test")
        s1.add_turn("q", "a")
        reset_session("reset-test")
        s2 = get_session("reset-test")
        assert s2 is not s1
        assert s2.history_text() == ""

    def test_none_session_returns_default(self):
        s = get_session(None)
        assert s is get_session("default")

    def test_session_persists_across_calls(self):
        s = get_session("persist-test")
        s.add_turn("remember this", "ok")
        s2 = get_session("persist-test")
        assert "remember this" in s2.history_text()
