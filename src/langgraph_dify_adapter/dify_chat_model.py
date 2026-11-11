"""A LangChain chat model backed by Dify's official chat-messages API."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import SecretStr

from .exceptions import DifyAPIError


class DifyUniversalChatModel(BaseChatModel):
    """Call a Dify app through ``POST /v1/chat-messages``.

    In LangGraph mode (the default), messages are serialized into one query and
    Dify conversation history is not used. In Dify-managed mode, only the latest
    human message is sent and ``conversation_id`` must be managed per user/thread.
    """

    dify_api_key: SecretStr
    dify_base_url: str
    user_id: str = "langgraph-dify-adapter"
    dify_manage_context: bool = False
    timeout: float = 120
    inputs: dict[str, Any] | None = None
    conversation_id: str | None = None
    max_query_chars: int | None = None

    @property
    def _llm_type(self) -> str:
        return "dify-chat-messages"

    def _build_payload(self, messages: Sequence[BaseMessage], *, stream: bool = False) -> dict[str, Any]:
        if not messages:
            raise ValueError("messages must contain at least one message")
        if self.dify_manage_context:
            last_human = next((message for message in reversed(messages) if message.type == "human"), None)
            if last_human is None:
                raise ValueError("Dify context mode requires at least one human message")
            query = _text_content(last_human.content)
        else:
            role_names = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
            query = "\n\n".join(
                f"[{role_names.get(message.type, message.type)}]\n{_text_content(message.content)}"
                for message in messages
            )
        if self.max_query_chars and len(query) > self.max_query_chars:
            raise ValueError(f"serialized query exceeds max_query_chars={self.max_query_chars}")

        payload: dict[str, Any] = {
            "user": self.user_id,
            "response_mode": "streaming" if stream else "blocking",
            "inputs": self.inputs or {},
            "query": query,
        }
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.dify_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages)
        try:
            response = requests.post(
                self.dify_base_url, headers=self._headers(), json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            body = getattr(response, "text", "")
            status = getattr(response, "status_code", "")
            raise DifyAPIError(f"Dify request failed (HTTP {status}): {body or exc}") from exc
        except (ValueError, json.JSONDecodeError) as exc:
            raise DifyAPIError("Dify returned invalid JSON") from exc
        return _chat_result(data)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = self._build_payload(messages, stream=True)
        response = None
        try:
            response = requests.post(
                self.dify_base_url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
            for event in _iter_sse_events(response):
                if not event["data"]:
                    continue
                try:
                    data = json.loads(event["data"])
                except json.JSONDecodeError as exc:
                    raise DifyAPIError(f"Dify stream returned invalid JSON: {event['data']}") from exc
                event_type = data.get("event") or event["event"]
                if event_type in {"message", "agent_message"}:
                    chunk = AIMessageChunk(content=data.get("answer", ""))
                    if chunk.content:
                        yield ChatGenerationChunk(message=chunk)
                elif event_type == "error":
                    raise DifyAPIError(f"Dify stream error: {data}")
                elif event_type == "message_end":
                    usage = (data.get("metadata") or {}).get("usage")
                    if usage:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content="", usage_metadata=_usage_metadata(usage))
                        )
        except requests.RequestException as exc:
            body = getattr(response, "text", "")
            status = getattr(response, "status_code", "")
            raise DifyAPIError(f"Dify stream request failed (HTTP {status}): {body or exc}") from exc
        finally:
            if response is not None:
                response.close()


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(json.dumps(block, ensure_ascii=False, default=str))
        return "".join(parts)
    return str(content)


def _usage_metadata(usage: dict[str, Any]) -> dict[str, int]:
    result = {
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }
    if "total_tokens" in usage:
        result["total_tokens"] = int(usage["total_tokens"])
    return result


def _iter_sse_events(response: requests.Response) -> Iterator[dict[str, str]]:
    """Parse the small subset of the SSE protocol used by Dify."""
    event_type = "message"
    data_lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8")
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield {"event": event_type, "data": "\n".join(data_lines)}
            event_type, data_lines = "message", []
        elif line.startswith(":"):
            continue
        else:
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_type = value
            elif field == "data":
                data_lines.append(value)
    if data_lines:
        yield {"event": event_type, "data": "\n".join(data_lines)}


def _chat_result(data: dict[str, Any]) -> ChatResult:
    if not isinstance(data, dict) or "answer" not in data:
        raise DifyAPIError("Dify response is missing the required 'answer' field")
    metadata = {key: value for key, value in data.items() if key != "answer"}
    raw_usage = (data.get("metadata") or {}).get("usage")
    kwargs: dict[str, Any] = {"content": data["answer"], "response_metadata": metadata}
    if raw_usage:
        kwargs["usage_metadata"] = _usage_metadata(raw_usage)
    return ChatResult(generations=[ChatGeneration(message=AIMessage(**kwargs))])
