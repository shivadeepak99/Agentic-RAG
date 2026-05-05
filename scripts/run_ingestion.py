from __future__ import annotations

import argparse
import json
import shutil

# Allow running this file directly via `python scripts/run_ingestion.py ...`
# by ensuring the repository root is on sys.path.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import settings
from app.ingestion.build_chunks import build_chunks
from app.ingestion.download_pdfs import download_pdf
from app.ingestion.fetch_arxiv import fetch_arxiv
from app.ingestion.ingest import ingest_default_chunks_dir
from app.ingestion.parse_pdfs import write_parsed_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run full ingestion pipeline")
    parser.add_argument("--query", default="cat:cs.CL", help="arXiv search query")
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        help="Delete the persisted Chroma index (embeddings) before ingesting.",
    )
    args = parser.parse_args(argv)

    if args.reset_chroma:
        # Wipe the embedding index so this run re-embeds everything cleanly.
        # (Leave raw PDFs / parsed text / chunks intact unless the user deletes them.)
        if settings.chroma_persist_dir.exists():
            shutil.rmtree(settings.chroma_persist_dir, ignore_errors=True)
        print(f"[reset] Deleted Chroma dir: {settings.chroma_persist_dir}")

    settings.raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
    settings.parsed_dir.mkdir(parents=True, exist_ok=True)
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] query={args.query!r} max_results={args.max_results}")
    papers = fetch_arxiv(query=args.query, max_results=args.max_results)
    print(f"[fetch] got {len(papers)} papers")
    print(f"[paths] raw_pdfs={settings.raw_pdfs_dir} parsed={settings.parsed_dir} chunks={settings.chunks_dir}")

    meta = []
    for i, p in enumerate(papers, start=1):
        print(f"[{i}/{len(papers)}] {p.arxiv_id} download")
        pdf_path = download_pdf(p, settings.raw_pdfs_dir)

        print(f"[{i}/{len(papers)}] {p.arxiv_id} parse")
        text_path = write_parsed_text(pdf_path, settings.parsed_dir)

        print(f"[{i}/{len(papers)}] {p.arxiv_id} chunk")
        build_chunks(text_path, settings.chunks_dir, source=f"arxiv:{p.arxiv_id}")

        meta.append({"id": p.arxiv_id, "title": p.title, "summary": p.summary, "pdf_url": p.pdf_url})

    settings.metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[meta] wrote {len(meta)} records to {settings.metadata_path}")

    count = ingest_default_chunks_dir()
    print(f"[ingest] embedded+stored {count} chunks into Chroma at {settings.chroma_persist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
