from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _has_persisted_index() -> bool:
    chroma_dir = settings.chroma_persist_dir
    if not chroma_dir.exists():
        return False
    return any(chroma_dir.rglob("*"))


def _run_ingestion_if_needed() -> None:
    bootstrap = _env_bool("BOOTSTRAP_INGEST", True)
    reset_chroma = _env_bool("RESET_CHROMA", False)

    if not bootstrap:
        print("[boot] BOOTSTRAP_INGEST=false -> skipping ingestion bootstrap")
        return

    if _has_persisted_index() and not reset_chroma:
        print(f"[boot] Found persisted index at {settings.chroma_persist_dir}; skipping ingestion")
        return

    query = os.getenv("INGEST_QUERY", "cat:cs.AI")
    max_results = os.getenv("INGEST_MAX_RESULTS", "20")

    cmd = [
        sys.executable,
        "scripts/run_ingestion.py",
        "--query",
        query,
        "--max-results",
        str(max_results),
    ]
    if reset_chroma:
        cmd.append("--reset-chroma")

    print(f"[boot] Running ingestion bootstrap: query={query!r} max_results={max_results}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _run_server() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"[boot] Starting server on http://{host}:{port}")
    os.execvp(cmd[0], cmd)


def main() -> int:
    _run_ingestion_if_needed()
    _run_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
