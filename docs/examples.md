# Examples

- `01_simple_chat.py`: basic blocking and streamed requests.
- `02_langchain_chain.py`: a fixed LangChain prompt and model chain.
- `03_langgraph_agent.py`: graph-managed history, JSON action selection, and local tool execution.
- `04_with_checkpointer.py`: a graph compiled with LangGraph's SQLite saver and a `thread_id`.
- `05_multi_model.py`: routing and answer nodes backed by separate Dify apps.

Set `DIFY_API_KEY` and `DIFY_BASE_URL` in the environment before running network examples. The examples deliberately avoid hard-coded working credentials.
