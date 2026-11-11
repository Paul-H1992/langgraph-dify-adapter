import os

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph_dify_adapter import DifyUniversalChatModel, build_json_agent

llm = DifyUniversalChatModel(
    dify_api_key=os.environ["DIFY_API_KEY"],
    dify_base_url=os.environ["DIFY_BASE_URL"],
)
agent = build_json_agent(
    llm=llm,
    tools={"search_db": lambda query: f"模拟查询结果：{query}"},
    system_prompt='工具调用输出JSON：{"action":"search_db","args":{"query":"..."}}。',
)

if __name__ == "__main__":
    with SqliteSaver.from_conn_string("checkpoint.sqlite") as checkpointer:
        graph = build_json_agent(
            llm=llm,
            tools={"search_db": lambda query: f"模拟查询结果：{query}"},
            system_prompt='工具调用输出JSON：{"action":"search_db","args":{"query":"..."}}。',
            checkpointer=checkpointer,
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="查订单 1002")], "retry_count": 0},
            config={"configurable": {"thread_id": "thread-001"}},
        )
        print(result["messages"][-1].content)
