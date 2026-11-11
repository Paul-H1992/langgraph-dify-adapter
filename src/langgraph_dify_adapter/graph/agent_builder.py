"""Build a small LangGraph agent using JSON tool requests from a Dify model."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from ..utils import safe_json_loads


class AgentState(MessagesState):
    retry_count: int
    parsed_action: dict[str, Any] | None
    next_step: Literal["tool", "retry", "done"]


def build_json_agent(
    llm: Any,
    tools: dict[str, Callable[..., Any]],
    system_prompt: str,
    max_retry: int = 2,
    checkpointer: Any = None,
):
    """Compile a JSON-action agent graph.

    Dify chat-messages does not expose OpenAI ``tool_calls``. This helper instead
    asks the model for ``{"action": ..., "args": {...}}`` JSON. Tools are local
    Python callables and should validate their own inputs and permissions.
    """
    if max_retry < 0:
        raise ValueError("max_retry must be non-negative")

    def call_llm(state: AgentState):
        response = llm.invoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [response]}

    def parse_and_route(state: AgentState):
        text = _content_as_text(state["messages"][-1].content)
        ok, data = safe_json_loads(text)
        valid = (
            ok
            and isinstance(data, dict)
            and data.get("action") in tools
            and isinstance(data.get("args", {}), dict)
        )
        if valid:
            return {"parsed_action": data, "next_step": "tool"}
        retries = state.get("retry_count", 0)
        if _looks_like_json(text) and retries < max_retry:
            return {"parsed_action": None, "retry_count": retries + 1, "next_step": "retry"}
        return {"parsed_action": None, "next_step": "done"}

    def retry(state: AgentState):
        return {"messages": [HumanMessage(content=(
            "输出格式无效。若需调用工具，只能输出一个JSON对象："
            '{"action":"工具名","args":{}}。若无需工具，请输出普通回答。'
        ))]}

    def run_tool(state: AgentState):
        action = state["parsed_action"]
        result = tools[action["action"]](**action.get("args", {}))
        return {
            "messages": [HumanMessage(content=f"工具 {action['action']} 的结果：{result}")],
            "parsed_action": None,
            "retry_count": 0,
        }

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("parse", parse_and_route)
    graph.add_node("retry", retry)
    graph.add_node("tool", run_tool)
    graph.add_edge(START, "llm")
    graph.add_edge("llm", "parse")
    graph.add_conditional_edges(
        "parse", _parse_route,
        {"tool": "tool", "retry": "retry", "done": END},
    )
    graph.add_edge("retry", "llm")
    graph.add_edge("tool", "llm")
    return graph.compile(checkpointer=checkpointer)


def _content_as_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _parse_route(state: AgentState) -> str:
    return state["next_step"]


def _looks_like_json(text: str) -> bool:
    return text.lstrip().startswith(("{", "["))
