import os

from langchain_core.messages import HumanMessage

from langgraph_dify_adapter import DifyUniversalChatModel, build_json_agent

llm = DifyUniversalChatModel(
    dify_api_key=os.environ["DIFY_API_KEY"],
    dify_base_url=os.environ["DIFY_BASE_URL"],
)


def search_db(query: str) -> str:
    return f"模拟查询结果：{query}"


agent = build_json_agent(
    llm=llm,
    tools={"search_db": search_db},
    system_prompt=(
        '需要调用工具时只输出JSON：{"action":"search_db","args":{"query":"..."}}。'
        "否则直接回答。不要附加解释或Markdown。"
    ),
    max_retry=2,
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [HumanMessage(content="帮我查询订单 1001")], "retry_count": 0}
    )
    print(result["messages"][-1].content)
