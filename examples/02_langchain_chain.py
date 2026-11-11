import os

from langchain_core.prompts import ChatPromptTemplate
from langgraph_dify_adapter import DifyUniversalChatModel

llm = DifyUniversalChatModel(
    dify_api_key=os.environ["DIFY_API_KEY"],
    dify_base_url=os.environ["DIFY_BASE_URL"],
)
chain = ChatPromptTemplate.from_messages(
    [("system", "你是专业数据分析师，回答简洁。"), ("human", "问题：{query}")]
) | llm

if __name__ == "__main__":
    print(chain.invoke({"query": "什么是 RAG？"}).content)
