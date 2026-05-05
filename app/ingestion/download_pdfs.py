from __future__ import annotations

from pathlib import Path

import requests

from app.ingestion.fetch_arxiv import ArxivPaper


def download_pdf(paper: ArxivPaper, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{paper.arxiv_id}.pdf"

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    r = requests.get(paper.pdf_url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path
