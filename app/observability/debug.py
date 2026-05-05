from __future__ import annotations

import pprint


def pretty(obj: object) -> str:
    return pprint.pformat(obj, width=100)
