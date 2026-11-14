import os
import httpx
from dotenv import load_dotenv

load_dotenv()

# Local Ollama server (temporary stand-in for Amazon Bedrock)
client = httpx.Client(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    timeout=120,  # first call loads the model into memory, which can take a while
)

# Test 1: Embeddings (mxbai-embed-large -> 1024 dims, same as Titan v2)
embedding_response = client.post(
    "/api/embed",
    json={
        "model": os.getenv("OLLAMA_EMBED_MODEL"),
        "input": "Hello Ollama",
    },
)
embedding_response.raise_for_status()

embedding = embedding_response.json()["embeddings"][0]

print("Embedding dimension:", len(embedding), f"(EMBED_DIM={os.getenv('EMBED_DIM')})")

# Test 2: Chat
chat_response = client.post(
    "/api/chat",
    json={
        "model": os.getenv("OLLAMA_CHAT_MODEL"),
        "messages": [
            {"role": "user", "content": "Explain Agentic AI in one sentence."}
        ],
        "stream": False,
        "think": False,  # skip qwen3's reasoning output; just return the answer
        "options": {"num_predict": 200},
    },
)
chat_response.raise_for_status()

print(
    "Ollama Response:",
    chat_response.json()["message"]["content"].strip()
)
