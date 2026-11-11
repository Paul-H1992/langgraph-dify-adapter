import os

from langgraph_dify_adapter import DifyUniversalChatModel

llm_light = DifyUniversalChatModel(
    dify_api_key=os.environ["DIFY_LIGHT_API_KEY"],
    dify_base_url=os.environ["DIFY_BASE_URL"],
)
llm_strong = DifyUniversalChatModel(
    dify_api_key=os.environ["DIFY_STRONG_API_KEY"],
    dify_base_url=os.environ["DIFY_BASE_URL"],
)

if __name__ == "__main__":
    category = llm_light.invoke("判断问题类型：帮我查订单").content
    print("分类结果：", category)
    print("回答：", llm_strong.invoke(f"详细解释这个请求：{category}").content)
