"""Small helpers for JSON-oriented agent output and context management."""

from __future__ import annotations

import json
from typing import Any


def safe_json_loads(text: str) -> tuple[bool, Any]:
    """Parse JSON, returning ``(False, None)`` for malformed or non-string input."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, None
    return True, value


def truncate_messages(messages: list[Any], max_messages: int = 40) -> list[Any]:
    """Keep the newest messages; token-aware truncation is application-specific."""
    if max_messages < 1:
        raise ValueError("max_messages must be at least 1")
    return messages[-max_messages:]
