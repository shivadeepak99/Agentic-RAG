from __future__ import annotations

from app.agent.state import AgentState
from app.observability.logger import get_logger
from app.retrieval.hybrid import hybrid_search


_logger = get_logger("agent.retrieve")


def retrieve(state: AgentState) -> dict:
    state.setdefault("trace", []).append("retrieve")

    decision = state.get("decision") or {}
    query = decision.get("query") or state.get("question", "")

    try:
        docs = hybrid_search(query)
    except Exception as exc:
        _logger.error("retrieve.failure query=%r error=%s", query, exc)
        return {"documents": []}

    _logger.info(
        "retrieve.ok query=%r n_docs=%d top_score=%.3f",
        query,
        len(docs),
        float(docs[0]["score"]) if docs else 0.0,
    )
    return {"documents": docs}
