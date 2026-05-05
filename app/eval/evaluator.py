from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agent.graph import build_graph, run_agent


REFUSAL_MARKERS = ("can't help", "cannot help", "won't", "refuse", "sensitive", "out of scope")
CLARIFY_MARKERS = ("clarif", "more detail", "could you", "what would you like", "rephrase")
DONT_KNOW_MARKERS = ("don't know", "do not know", "not in the indexed", "no information", "don't have")


@dataclass
class EvalResult:
    id: str
    question: str
    answer: str
    score: float
    expected_action: str | None = None
    actual_action: str | None = None
    action_match: bool = False
    notes: list[str] = field(default_factory=list)


def _content_score(answer: str, expected_contains: list[str]) -> float:
    if not expected_contains:
        return 1.0
    ans = answer.lower()
    hits = sum(1 for s in expected_contains if s.lower() in ans)
    return hits / max(1, len(expected_contains))


def _matches_any(text: str, markers: tuple[str, ...]) -> bool:
    t = text.lower()
    return any(m in t for m in markers)


def _score_row(row: dict, result: dict) -> EvalResult:
    answer = result.get("answer", "") or ""
    decision = result.get("decision") or {}
    actual_action = decision.get("action")
    expected_action = row.get("expected_action")

    notes: list[str] = []
    score_components: list[float] = []

    # 1) Action correctness (when expected_action is set)
    action_match = False
    if expected_action:
        action_match = (actual_action == expected_action)
        score_components.append(1.0 if action_match else 0.0)
        if not action_match:
            notes.append(f"action mismatch: got {actual_action!r} want {expected_action!r}")

    # 2) Behavior-specific assertions
    kind = row.get("kind")

    if kind == "refuse":
        ok = _matches_any(answer, REFUSAL_MARKERS) or actual_action == "refuse"
        score_components.append(1.0 if ok else 0.0)
        if not ok:
            notes.append("answer does not look like a refusal")

    elif kind == "clarify":
        ok = _matches_any(answer, CLARIFY_MARKERS) or actual_action == "clarify"
        score_components.append(1.0 if ok else 0.0)
        if not ok:
            notes.append("answer does not look like a clarification")

    elif kind == "ood":
        ok = _matches_any(answer, DONT_KNOW_MARKERS)
        score_components.append(1.0 if ok else 0.0)
        if not ok:
            notes.append("OOD: expected an 'I don't know'-style response")

    elif kind == "tool":
        expected_tool = row.get("expected_tool")
        if expected_tool:
            actual_tool = decision.get("tool_name")
            ok = actual_tool == expected_tool
            score_components.append(1.0 if ok else 0.0)
            if not ok:
                notes.append(f"tool mismatch: got {actual_tool!r} want {expected_tool!r}")

    # 3) Content keywords (only if specified)
    expected_contains = row.get("expected_contains", [])
    if expected_contains:
        content_score = _content_score(answer, expected_contains)
        score_components.append(content_score)
        if content_score < 1.0:
            missing = [s for s in expected_contains if s.lower() not in answer.lower()]
            notes.append(f"missing keywords: {missing}")

    final = sum(score_components) / max(1, len(score_components)) if score_components else 1.0

    return EvalResult(
        id=row["id"],
        question=row["question"],
        answer=answer,
        score=round(final, 3),
        expected_action=expected_action,
        actual_action=actual_action,
        action_match=action_match,
        notes=notes,
    )


def run_eval(dataset: list[dict]) -> list[EvalResult]:
    graph = build_graph()
    results: list[EvalResult] = []

    for row in dataset:
        out = run_agent(graph, row["question"])
        results.append(_score_row(row, out))

    return results


def aggregate(results: list[EvalResult]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0, "avg_score": 0.0, "action_accuracy": 0.0}

    avg = sum(r.score for r in results) / n
    matched = sum(1 for r in results if r.action_match)
    expected_actions = sum(1 for r in results if r.expected_action is not None)
    action_acc = matched / expected_actions if expected_actions else 0.0
    return {
        "n": n,
        "avg_score": round(avg, 3),
        "action_accuracy": round(action_acc, 3),
        "n_with_expected_action": expected_actions,
    }


# Backward-compatible export used by older imports.
def score_answer(answer: str, expected_contains: list[str]) -> float:
    return _content_score(answer, expected_contains)
