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


def _run_with_mode(label: str, search_fn) -> dict:
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
    return {"label": label, "results": results, "summary": summary}


def _run_dataset(label: str, dataset: list[dict], search_fn) -> dict:
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
    return {"label": label, "results": results, "summary": summary}


def _print_comparison(title: str, hybrid: dict, vector: dict) -> None:
    print(f"=== {title} ===\n")

    for mode in (hybrid, vector):
        s = mode["summary"]
        print(f"=== {mode['label']} ===")
        print(f"  avg_score:       {s['avg_score']:.3f}")
        print(f"  action_accuracy: {s['action_accuracy']:.3f}  "
              f"({s['n_with_expected_action']} cases with expected_action)")
        print()

    print(f"{'ID':<20} {'hybrid':>8} {'vec-only':>10}  {'delta':>8}")
    print("-" * 52)
    for hr, vr in zip(hybrid["results"], vector["results"]):
        delta = hr.score - vr.score
        sign = "+" if delta >= 0 else ""
        print(f"{hr.id:<20} {hr.score:>8.3f} {vr.score:>10.3f}  {sign}{delta:>7.3f}")

    hs = hybrid["summary"]["avg_score"]
    vs = vector["summary"]["avg_score"]
    lift = hs - vs
    sign = "+" if lift >= 0 else ""
    print("-" * 52)
    print(f"{'TOTAL':<20} {hs:>8.3f} {vs:>10.3f}  {sign}{lift:>7.3f}")
    print()
    print(f"Hybrid {'outperforms' if lift > 0 else 'underperforms'} vector-only by {abs(lift):.3f} avg score.")
    print()


def main() -> int:
    print("Running ablation study across retrieval strategies\n")

    modes = [
        ("vector-only", _hybrid_mod.vector_only_search),
        ("lightweight hybrid", _hybrid_mod.lightweight_hybrid_search),
        ("true hybrid (no reranker)", lambda q: _hybrid_mod.true_hybrid_search(q, use_reranker=False)),
        ("true hybrid + cross-encoder", _hybrid_mod.hybrid_search),
    ]
    full_runs = [_run_with_mode(label, fn) for label, fn in modes]

    retrieval_rows = [row for row in DATASET if row.get("expected_action") == "retrieve"]
    retrieval_runs = [_run_dataset(label, retrieval_rows, fn) for label, fn in modes]

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
