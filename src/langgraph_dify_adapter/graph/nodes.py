"""Reusable JSON parsing and retry nodes."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from ..utils import safe_json_loads


def parse_json_node(state: dict[str, Any]) -> dict[str, Any]:
    content = state["messages"][-1].content
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    ok, data = safe_json_loads(text)
    return {"json_ok": ok, "parsed": data}


def retry_prompt_node(state: dict[str, Any]) -> dict[str, Any]:
    del state
    return {"messages": [HumanMessage(content="JSON格式错误，请重新输出标准JSON，禁止多余文字。")]}
