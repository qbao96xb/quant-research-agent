import ollama

MODEL = "qwen2.5-coder:14b"  # must match `ollama ps` exactly

stream = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Explain what a 5-day forecast horizon means."}],
    stream=True,
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()
