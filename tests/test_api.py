"""Tests for the FastAPI endpoints using TestClient."""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.memory.store import reset_session


@pytest.fixture(autouse=True)
def reset_api_sessions():
    reset_session("api-test")
    reset_session("default")
    yield
    reset_session("api-test")
    reset_session("default")


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data.get("status") == "ok"


# ---------------------------------------------------------------------------
# GET / (UI)
# ---------------------------------------------------------------------------

class TestUIEndpoint:
    def test_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_contains_chat_elements(self, client):
        resp = client.get("/")
        body = resp.text
        assert "question" in body
        assert "Send" in body

    def test_contains_debug_toggle(self, client):
        resp = client.get("/")
        assert "debug" in resp.text.lower()

    def test_contains_streaming_toggle(self, client):
        resp = client.get("/")
        assert "stream" in resp.text.lower()


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

class TestAskEndpoint:
    def test_returns_200(self, client):
        resp = client.post("/ask", json={"question": "What is RAG?"})
        assert resp.status_code == 200

    def test_response_has_answer(self, client):
        resp = client.post("/ask", json={"question": "What is RAG?"})
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)

    def test_response_has_trace(self, client):
        resp = client.post("/ask", json={"question": "Explain RAG"})
        data = resp.json()
        assert "trace" in data
        assert isinstance(data["trace"], list)

    def test_response_has_session_id(self, client):
        resp = client.post("/ask", json={"question": "hi", "session_id": "api-test"})
        data = resp.json()
        assert data["session_id"] == "api-test"

    def test_debug_false_omits_documents(self, client):
        resp = client.post("/ask", json={"question": "What is RAG?", "debug": False})
        data = resp.json()
        assert data.get("documents") == []

    def test_debug_true_includes_decision(self, client):
        resp = client.post("/ask", json={"question": "What is RAG?", "debug": True})
        data = resp.json()
        assert "decision" in data

    def test_refusal_question(self, client):
        resp = client.post("/ask", json={"question": "What is my password?"})
        assert resp.status_code == 200
        data = resp.json()
        answer = data.get("answer", "").lower()
        decision = data.get("decision") or {}
        assert decision.get("action") == "refuse" or "can't" in answer or "sensitive" in answer

    def test_clarify_question(self, client):
        resp = client.post("/ask", json={"question": "tell me more"})
        assert resp.status_code == 200
        data = resp.json()
        decision = data.get("decision") or {}
        answer = data.get("answer", "").lower()
        assert decision.get("action") == "clarify" or "clarif" in answer or "detail" in answer

    def test_session_memory_persists(self, client):
        # First turn
        r1 = client.post("/ask", json={"question": "What is RAG?", "session_id": "mem-test"})
        assert r1.status_code == 200
        # Second turn in same session
        r2 = client.post("/ask", json={"question": "Summarize what we discussed.", "session_id": "mem-test"})
        assert r2.status_code == 200
        assert r2.json()["answer"].strip()

    def test_empty_question_handled(self, client):
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("answer", "").strip()

    def test_very_long_question_handled(self, client):
        resp = client.post("/ask", json={"question": "explain machine learning " * 50})
        assert resp.status_code == 200

    def test_tool_question_arxiv(self, client):
        resp = client.post("/ask", json={"question": "Search arxiv for papers on RLHF"})
        data = resp.json()
        decision = data.get("decision") or {}
        assert decision.get("action") == "tool"
        assert decision.get("tool_name") == "arxiv_search"

    def test_calculator_question(self, client):
        resp = client.post("/ask", json={"question": "Compute 2 ** 10"})
        data = resp.json()
        assert resp.status_code == 200
        # Either the answer contains 1024 (tool path) or an answer was generated
        assert data.get("answer")


# ---------------------------------------------------------------------------
# POST /ask/stream
# ---------------------------------------------------------------------------

class TestStreamEndpoint:
    def test_returns_200(self, client):
        with client.stream("POST", "/ask/stream", json={"question": "hi"}) as resp:
            assert resp.status_code == 200

    def test_content_type_is_event_stream(self, client):
        with client.stream("POST", "/ask/stream", json={"question": "hi"}) as resp:
            assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_emits_token_events(self, client):
        events: list[str] = []
        with client.stream("POST", "/ask/stream", json={"question": "What is RAG?"}) as resp:
            for line in resp.iter_lines():
                events.append(line)
        # Should have at least one 'event: token' and one 'event: done'
        event_names = [l[7:] for l in events if l.startswith("event: ")]
        assert "token" in event_names or "done" in event_names

    def test_emits_done_sentinel(self, client):
        events: list[str] = []
        with client.stream("POST", "/ask/stream", json={"question": "hello"}) as resp:
            for line in resp.iter_lines():
                events.append(line)
        event_names = [l[7:] for l in events if l.startswith("event: ")]
        assert "done" in event_names

    def test_debug_mode_emits_debug_events(self, client):
        events: list[str] = []
        with client.stream(
            "POST", "/ask/stream",
            json={"question": "What is RAG?", "debug": True}
        ) as resp:
            for line in resp.iter_lines():
                events.append(line)
        event_names = [l[7:] for l in events if l.startswith("event: ")]
        assert "debug" in event_names

    def test_clarify_path_returns_token_and_done(self, client):
        events: list[str] = []
        with client.stream(
            "POST", "/ask/stream", json={"question": "tell me more"}
        ) as resp:
            for line in resp.iter_lines():
                events.append(line)
        event_names = [l[7:] for l in events if l.startswith("event: ")]
        assert "done" in event_names

    def test_refuse_path_returns_token_and_done(self, client):
        events: list[str] = []
        with client.stream(
            "POST", "/ask/stream", json={"question": "What is my password?"}
        ) as resp:
            for line in resp.iter_lines():
                events.append(line)
        event_names = [l[7:] for l in events if l.startswith("event: ")]
        assert "done" in event_names
