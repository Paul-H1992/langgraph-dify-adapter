from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from langgraph_dify_adapter import DifyAPIError, DifyUniversalChatModel


def make_model(**kwargs):
    return DifyUniversalChatModel(
        dify_api_key="test-key",
        dify_base_url="http://dify.local/v1/chat-messages",
        **kwargs,
    )


def test_payload_serializes_langchain_roles_and_tool_context():
    model = make_model(inputs={"tenant": "acme"})
    payload = model._build_payload(
        [SystemMessage(content="Be concise"), HumanMessage(content="hello")]
    )
    assert payload["query"] == "[system]\nBe concise\n\n[user]\nhello"
    assert payload["inputs"] == {"tenant": "acme"}
    assert "conversation_id" not in payload


def test_context_mode_sends_only_latest_human_message():
    model = make_model(dify_manage_context=True, conversation_id="conv-1")
    payload = model._build_payload(
        [HumanMessage(content="old"), HumanMessage(content="latest")]
    )
    assert payload["query"] == "latest"
    assert payload["conversation_id"] == "conv-1"


def test_invoke_uses_dify_answer_and_response_metadata():
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "answer": "hi",
        "conversation_id": "conv-1",
        "message_id": "msg-1",
        "metadata": {"usage": {"total_tokens": 2}},
    }
    with patch("langgraph_dify_adapter.dify_chat_model.requests.post", return_value=response):
        result = make_model().invoke("hello")
    assert result.content == "hi"
    assert result.response_metadata["conversation_id"] == "conv-1"
    assert result.usage_metadata["total_tokens"] == 2


def test_http_failure_is_wrapped_with_status_and_body():
    response = Mock()
    response.status_code = 401
    response.text = "invalid key"
    response.raise_for_status.side_effect = __import__("requests").HTTPError("401")
    with (
        patch("langgraph_dify_adapter.dify_chat_model.requests.post", return_value=response),
        pytest.raises(DifyAPIError, match="401.*invalid key"),
    ):
        make_model().invoke("hello")


def test_stream_returns_langchain_chunks_and_accumulates_answer():
    response = Mock()
    response.raise_for_status.return_value = None
    response.iter_lines.return_value = [
        'event: message',
        'data: {"event":"message","answer":"hello "}',
        '',
        'event: message',
        'data: {"event":"message","answer":"world"}',
        '',
        'event: message_end',
        'data: {"event":"message_end"}',
        '',
    ]
    with (
        patch("langgraph_dify_adapter.dify_chat_model.requests.post", return_value=response),
    ):
        chunks = list(make_model()._stream([HumanMessage(content="hello")]))
    assert all(isinstance(chunk.message, AIMessageChunk) for chunk in chunks)
    assert "".join(chunk.message.content for chunk in chunks) == "hello world"
