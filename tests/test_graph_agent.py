from langchain_core.messages import AIMessage, HumanMessage

from langgraph_dify_adapter import build_json_agent


class SequenceModel:
    def __init__(self, answers):
        self.answers = iter(answers)

    def invoke(self, messages):
        del messages
        return AIMessage(content=next(self.answers))


def test_json_agent_executes_known_tool_then_returns_final_answer():
    called = []
    model = SequenceModel(
        [
            '{"action":"lookup","args":{"key":"abc"}}',
            "Found it.",
        ]
    )
    graph = build_json_agent(
        model,
        {"lookup": lambda key: called.append(key) or f"value={key}"},
        "Use lookup when needed.",
    )

    result = graph.invoke({"messages": [HumanMessage(content="find abc")], "retry_count": 0})

    assert called == ["abc"]
    assert result["messages"][-1].content == "Found it."


def test_json_agent_retries_invalid_output_up_to_limit():
    model = SequenceModel(["not JSON", "plain response"])
    graph = build_json_agent(model, {}, "Answer.", max_retry=1)

    result = graph.invoke({"messages": [HumanMessage(content="hi")], "retry_count": 0})

    assert result["messages"][-1].content == "plain response"
    assert result["retry_count"] == 1
