# langgraph-dify-adapter

Use a private Dify chat application as a LangChain `BaseChatModel`, then compose it with LangChain chains or LangGraph state machines. The adapter calls Dify's official `POST /v1/chat-messages` endpoint and does not require a Dify plugin.

> This project is an early, community-maintained adapter. Dify's chat API returns text; it does not expose OpenAI-compatible `tool_calls`. The included LangGraph helper uses a JSON action protocol and executes registered Python functions locally. Validate tool inputs and authorization in your own functions.

## What it supports

- Blocking and Server-Sent Events streaming responses.
- LangChain `invoke`, `stream`, prompt templates, and chains.
- Stateless message serialization for LangGraph (recommended for agent graphs).
- Optional Dify-managed conversation context when the caller manages conversation IDs per user/thread.
- JSON-routed local tools in LangGraph, with bounded retries.
- Optional SQLite checkpoints through LangGraph.

The model adapter is a transport and serialization layer. LangGraph provides branching, state, retries, tool execution, and persistence. Dify's own workflow still runs behind the chat API call.

## Requirements

- Python 3.10+
- A Dify chat application API key and a reachable Dify API endpoint.
- A published Dify app. The Dify workflow's LLM node should not duplicate system instructions that are already sent in the LangGraph messages.

## Install

```bash
git clone https://github.com/Paul-H1992/langgraph-dify-adapter.git
cd langgraph-dify-adapter
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install .
```

For development, install the project with its development tools:

```bash
pip install -e ".[dev]"
```

For the SQLite checkpoint example:

```bash
pip install -e ".[sqlite]"
```

## Quick start

The URL is the full Dify endpoint, commonly `https://dify.example.com/v1/chat-messages`.

```python
import os

from langgraph_dify_adapter import DifyUniversalChatModel

llm = DifyUniversalChatModel(
    dify_api_key=os.environ["DIFY_API_KEY"],
    dify_base_url="https://dify.example.com/v1/chat-messages",
    user_id="my-service-user-42",
    timeout=120,
)

response = llm.invoke("你好")
print(response.content)

for chunk in llm.stream("简单介绍一下 RAG"):
    print(chunk.content, end="", flush=True)
```

Keep API keys on the server and load them from environment variables or a secrets manager. Dify scopes conversations and uploaded files to the `user` value, so applications serving multiple end users should provide a stable, user-specific `user_id`.

## Context modes

By default, `dify_manage_context=False`. The adapter serializes the supplied LangChain messages into one Dify `query` and omits `conversation_id`. This is the recommended mode for LangGraph because graph state is the conversation history source.

With `dify_manage_context=True`, only the latest human message is sent. Set `conversation_id` to continue a Dify conversation and persist the response's `conversation_id` (available in `AIMessage.response_metadata`) per user/thread. Do not share one fixed conversation ID across unrelated users. System messages and earlier history are not sent in this mode, so it is usually unsuitable for agent graphs.

## LangGraph JSON tool example

```python
from langchain_core.messages import HumanMessage
from langgraph_dify_adapter import DifyUniversalChatModel, build_json_agent

llm = DifyUniversalChatModel(
    dify_api_key="app-...",
    dify_base_url="http://dify.internal/v1/chat-messages",
)

def lookup_order(order_id: str) -> str:
    # Add validation and access control appropriate to your service.
    return f"Order {order_id}: processing"

agent = build_json_agent(
    llm=llm,
    tools={"lookup_order": lookup_order},
    system_prompt=(
        'If a tool is needed, output JSON only: '
        '{"action":"lookup_order","args":{"order_id":"..."}}. '
        "Otherwise answer normally."
    ),
    max_retry=2,
)

result = agent.invoke({"messages": [HumanMessage(content="查订单 1001")], "retry_count": 0})
print(result["messages"][-1].content)
```

See [`examples/`](examples/) for chains, checkpointing, streaming, and multiple Dify apps.

## Limitations

- Dify app APIs return text, not native OpenAI `tool_calls`; JSON tool selection is prompt-based and can fail. Treat model output as untrusted input.
- The helper executes Python callables in-process; it does not provide sandboxing, authorization, or schema validation.
- The wrapper sends text messages only. Dify file/image upload and multimodal inputs require separate API handling.
- Embeddings and Dify knowledge APIs are outside this chat-model adapter. A Dify app can include retrieval in its own published workflow.
- Token-aware context trimming is application-specific; `truncate_messages` only limits message count.

## Examples

1. `01_simple_chat.py` — blocking and streaming connectivity.
2. `02_langchain_chain.py` — `ChatPromptTemplate` and a runnable chain.
3. `03_langgraph_agent.py` — JSON action routing and a local tool.
4. `04_with_checkpointer.py` — SQLite-backed LangGraph threads.
5. `05_multi_model.py` — separate Dify apps selected in graph nodes.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

## License

MIT. See [`LICENSE`](LICENSE).
