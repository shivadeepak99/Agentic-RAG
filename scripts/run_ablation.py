"""Ablation: compare retrieval strategies.

Run:
    python scripts/run_ablation.py

Prints side-by-side comparison tables and exits with code 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.eval.dataset import DATASET
from app.eval.evaluator import aggregate, run_eval

import app.agent.nodes.retrieve as _retrieve_mod
import app.retrieval.hybrid as _hybrid_mod


def _run_with_mode(label: str, short: str, search_fn) -> dict:
    """Monkey-patch hybrid_search and run the full eval."""
    original_hybrid = _hybrid_mod.hybrid_search
    original_node = _retrieve_mod.hybrid_search

    _hybrid_mod.hybrid_search = search_fn
    _retrieve_mod.hybrid_search = search_fn
    try:
        results = run_eval(DATASET)
    finally:
        _hybrid_mod.hybrid_search = original_hybrid
        _retrieve_mod.hybrid_search = original_node

    summary = aggregate(results)
    return {"label": label, "short": short, "results": results, "summary": summary}


def _run_dataset(label: str, short: str, dataset: list[dict], search_fn) -> dict:
    """Run one ablation mode against a specific dataset slice."""
    original_hybrid = _hybrid_mod.hybrid_search
    original_node = _retrieve_mod.hybrid_search

    _hybrid_mod.hybrid_search = search_fn
    _retrieve_mod.hybrid_search = search_fn
    try:
        results = run_eval(dataset)
    finally:
        _hybrid_mod.hybrid_search = original_hybrid
        _retrieve_mod.hybrid_search = original_node

    summary = aggregate(results)
    return {"label": label, "short": short, "results": results, "summary": summary}


def _print_comparison(title: str, left: dict, right: dict) -> None:
    print(f"=== {title} ===\n")

    for mode in (left, right):
        s = mode["summary"]
        print(f"=== {mode['label']} ===")
        print(f"  avg_score:       {s['avg_score']:.3f}")
        print(f"  action_accuracy: {s['action_accuracy']:.3f}  "
              f"({s['n_with_expected_action']} cases with expected_action)")
        print()

    lh = left["short"]
    rh = right["short"]
    col = max(len(lh), len(rh), 10)
    print(f"{'ID':<20} {lh:>{col}} {rh:>{col}}  {'delta':>8}")
    print("-" * (20 + col * 2 + 14))
    for lr, rr in zip(left["results"], right["results"]):
        delta = lr.score - rr.score
        sign = "+" if delta >= 0 else ""
        print(f"{lr.id:<20} {lr.score:>{col}.3f} {rr.score:>{col}.3f}  {sign}{delta:>7.3f}")

    ls = left["summary"]["avg_score"]
    rs = right["summary"]["avg_score"]
    lift = ls - rs
    sign = "+" if lift >= 0 else ""
    print("-" * (20 + col * 2 + 14))
    print(f"{'TOTAL':<20} {ls:>{col}.3f} {rs:>{col}.3f}  {sign}{lift:>7.3f}")
    print()
    direction = "outperforms" if lift > 0 else ("ties" if lift == 0 else "underperforms")
    print(f"{left['label']} {direction} {right['label']} by {abs(lift):.3f} avg score.")
    print()


def main() -> int:
    print("Running ablation study across retrieval strategies\n")

    modes = [
        ("vector-only",              "vec-only",   _hybrid_mod.vector_only_search),
        ("lightweight hybrid",       "lw-hybrid",  _hybrid_mod.lightweight_hybrid_search),
        ("true hybrid (no reranker)","th-no-xenc", lambda q: _hybrid_mod.true_hybrid_search(q, use_reranker=False)),
        ("true hybrid + cross-encoder", "th+xenc", lambda q: _hybrid_mod.true_hybrid_search(q, use_reranker=True)),
    ]
    full_runs = [_run_with_mode(label, short, fn) for label, short, fn in modes]

    retrieval_rows = [row for row in DATASET if row.get("expected_action") == "retrieve"]
    retrieval_runs = [_run_dataset(label, short, retrieval_rows, fn) for label, short, fn in modes]

    print("=== Full agent eval (includes tool/chat/refusal paths) ===\n")
    for mode in full_runs:
        s = mode["summary"]
        print(f"=== {mode['label']} ===")
        print(f"  avg_score:       {s['avg_score']:.3f}")
        print(f"  action_accuracy: {s['action_accuracy']:.3f}  "
              f"({s['n_with_expected_action']} cases with expected_action)")
        print()

    _print_comparison(
        "Retrieval-sensitive subset: true hybrid + cross-encoder vs. vector-only",
        retrieval_runs[-1],
        retrieval_runs[0],
    )
    _print_comparison(
        "Retrieval-sensitive subset: true hybrid + cross-encoder vs. lightweight hybrid",
        retrieval_runs[-1],
        retrieval_runs[1],
    )
    _print_comparison(
        "Retrieval-sensitive subset: true hybrid + cross-encoder vs. true hybrid without reranker",
        retrieval_runs[-1],
        retrieval_runs[2],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
