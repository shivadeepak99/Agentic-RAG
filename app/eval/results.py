from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.eval.evaluator import EvalResult


def write_results(results: list[EvalResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
