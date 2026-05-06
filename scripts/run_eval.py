from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.eval.dataset import DATASET
from app.eval.evaluator import aggregate, run_eval
from app.eval.results import write_results


def main() -> int:
    results = run_eval(DATASET)
    summary = aggregate(results)

    out_path = Path("data/eval/results.json")
    write_results(results, out_path)

    print(f"Wrote {len(results)} eval results to {out_path}")
    print(f"Avg score:        {summary['avg_score']:.3f}")
    print(f"Action accuracy:  {summary['action_accuracy']:.3f} "
          f"({summary['n_with_expected_action']} cases with expected_action)")

    print("\nPer-category:")
    for kind, bucket in sorted(summary["per_kind"].items()):
        print(f"  {kind:<10} avg={bucket['avg_score']:.3f} n={bucket['n']}")

    print("\nComponent averages:")
    for name, value in sorted(summary["component_averages"].items()):
        print(f"  {name:<12} {value:.3f}")

    print("\nPer-case:")
    for r in results:
        flag = "OK " if r.score >= 0.99 else "..."
        breakdown = ", ".join(
            f"{name}={value:.2f}" for name, value in r.breakdown.items() if isinstance(value, float)
        )
        print(f"  [{flag}] {r.id} ({r.kind}) score={r.score:.2f} action={r.actual_action} "
              f"expected={r.expected_action}")
        print(f"        {breakdown}")
        for note in r.notes:
            print(f"        - {note}")

    failed = summary["failed_cases"]
    if failed:
        print("\nFailed-case summary:")
        for row in failed:
            note = row["notes"][0] if row["notes"] else "no diagnostic note"
            print(
                f"  {row['id']} ({row['kind']}): score={row['score']:.2f} "
                f"action={row['actual_action']} expected={row['expected_action']} | {note}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
