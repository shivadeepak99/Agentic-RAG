from __future__ import annotations

import json
import logging


def get_logger(name: str = "agentic-rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "level": record.levelname,
                "name": record.name,
                "msg": record.getMessage(),
            }
            return json.dumps(payload, ensure_ascii=False)

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger
