import os

from langgraph_dify_adapter import DifyUniversalChatModel

llm = DifyUniversalChatModel(
    dify_api_key=os.environ["DIFY_API_KEY"],
    dify_base_url=os.environ["DIFY_BASE_URL"],
    dify_manage_context=False,
    timeout=120,
)

if __name__ == "__main__":
    print(llm.invoke("你是谁？").content)
    for chunk in llm.stream("简单介绍 LangGraph"):
        print(chunk.content, end="", flush=True)
