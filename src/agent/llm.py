from openai import OpenAI

class OpenAICompatLLM:
    def __init__(self, model="qwen3.5:9b",
                 base_url="http://localhost:11434/v1", api_key="ollama",
                 timeout=60):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def chat(self, messages, tools=None):
        kwargs = dict(model=self.model, messages=messages, temperature=0)
        if tools:
            kwargs["tools"] = tools
        r = self.client.chat.completions.create(**kwargs)
        m = r.choices[0].message
        out = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            out["tool_calls"] = [
                {"id": t.id, "type": "function",
                 "function": {"name": t.function.name,
                              "arguments": t.function.arguments}}
                for t in m.tool_calls]
        return out

if __name__ == "__main__":
    llm = OpenAICompatLLM()
    print(f"model: {llm.model}\n")

    # Test 1: plain chat
    r = llm.chat([{"role": "user", "content": "Reply with the word OK."}])
    print("1. plain chat:", repr(r["content"]))

    # Test 2: does the model emit a structured tool call?
    tools = [{
        "type": "function",
        "function": {
            "name": "load_data",
            "description": "Load daily price data for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "start": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
    }]
    r = llm.chat([{"role": "user", "content": "Load KO data since 2015."}], tools)
    print("2. tool_calls:", r.get("tool_calls"))
    print("   content:", repr(r["content"]))
    print("\nRESULT:", "PASS" if r.get("tool_calls") else "FAIL (no structured tool call)")
