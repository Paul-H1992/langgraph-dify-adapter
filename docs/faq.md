# FAQ

### Why should I avoid duplicating system instructions in Dify?

In stateless mode, the adapter includes LangChain system messages in the serialized `query`. If the Dify workflow's LLM prompt also adds independent instructions, both instructions influence the output and may conflict. Keep responsibilities clear and test the combined prompt.

### Does this provide native tool calling?

No. Dify chat-messages returns text in `answer`, not OpenAI `tool_calls`. The example agent uses prompt-constrained JSON to choose local Python functions. Model output is not trusted; validate tool arguments and permissions before performing side effects.

### Which context mode should I use?

Use `dify_manage_context=False` for LangChain chains and LangGraph agents. It sends the supplied messages and leaves history management to your application. Use Dify-managed mode for simple chat only when your application stores the returned `conversation_id` per end user and passes it back on the next call.

### Why is `conversation_id` not updated automatically?

The model adapter is stateless across calls by design. Automatically mutating a shared model instance would mix users and graph threads. In Dify-managed mode, read `AIMessage.response_metadata["conversation_id"]` and persist it in your own per-user/session store.

### How do I handle invalid JSON?

The JSON agent retries malformed or unknown actions up to `max_retry`. Use a precise system prompt and validate argument types and values inside each tool. This cannot guarantee valid model output.

### The Dify request times out. What should I check?

Set `timeout`, confirm the full API URL and network route, and check Dify logs and the reverse proxy timeout. The request timeout applies to the underlying HTTP request.

### Can I use Dify knowledge retrieval?

Configure retrieval in the published Dify chat workflow and call that app through this adapter. Direct knowledge-base APIs and embedding models are outside this package.

### Can I select multiple Dify apps?

Yes. Create separate `DifyUniversalChatModel` instances with their own API keys and URLs, and choose the desired model in each graph node.
