"""Thin HTTP wrapper for Ollama embedding API."""
import requests


class OllamaClient:
    """Calls Ollama's /api/embeddings endpoint to generate text embeddings."""

    def __init__(self, base_url: str, model: str, timeout: int = 120) -> None:
        # 120s timeout: bge-m3 can be slow on first call (model warm-up)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> list:
        """Return the embedding vector for the given text."""
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
