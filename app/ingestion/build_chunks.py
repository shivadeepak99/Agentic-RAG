from __future__ import annotations

import json
from pathlib import Path

from app.retrieval.chunking import chunk_text


def build_chunks(text_path: Path, out_dir: Path, source: str) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = text_path.read_text(encoding="utf-8")
    chunks = chunk_text(raw)

    records = []
    for i, ch in enumerate(chunks):
        records.append({
            "id": f"{text_path.stem}:{i}",
            "text": ch,
            "source": source,
        })

    out_path = out_dir / f"{text_path.stem}.chunks.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records
