"""Ollama /api/chat wrapper for text completions (not embeddings)."""
import os
import requests


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, prompt: str) -> str:
        """Send a user prompt and return the assistant's text response."""
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def get_ollama_chat_client() -> OllamaChatClient:
    """Return OllamaChatClient configured from env vars."""
    base_url = os.getenv("WIKI_OLLAMA_BASE_URL", "http://jarvis:11434")
    model = os.getenv("WIKI_OLLAMA_MODEL", "qwen3:14b")
    return OllamaChatClient(base_url, model)
