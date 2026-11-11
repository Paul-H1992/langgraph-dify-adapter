"""LangChain and LangGraph integration for Dify chat applications."""

from .dify_chat_model import DifyUniversalChatModel
from .exceptions import DifyAPIError

__all__ = ["DifyAPIError", "DifyUniversalChatModel", "build_json_agent"]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "build_json_agent":
        from .graph.agent_builder import build_json_agent

        return build_json_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
