"""Business logic layer for vector embedding and search."""
import os
from joplin_mcp.vector.qdrant_client import QdrantClient, joplin_id_to_uuid
from joplin_mcp.vector.ollama_client import OllamaClient

EMBED_BODY_MAX_CHARS = 4000


def build_embed_text(title: str, body: str) -> str:
    """Build the text to embed: title + first 4000 chars of body."""
    return f"{title}\n\n{body[:EMBED_BODY_MAX_CHARS]}"


class VectorService:
    """Combines Qdrant and Ollama to provide note embedding and search."""

    def __init__(self, qdrant: QdrantClient, ollama: OllamaClient) -> None:
        self.qdrant = qdrant
        self.ollama = ollama

    def upsert_note(
        self,
        note_id: str,
        title: str,
        body: str,
        folder_path: str,
        folder_id: str,
        updated_time: int,
    ) -> None:
        """Embed a note and upsert it into Qdrant."""
        text = build_embed_text(title, body)
        vector = self.ollama.embed(text)
        self.qdrant.upsert(
            point_id=joplin_id_to_uuid(note_id),
            vector=vector,
            payload={
                "title": title,
                "folder_path": folder_path,
                "folder_id": folder_id,
                "updated_time": updated_time,
            },
        )

    def search(self, query: str, top_k: int = 5) -> list:
        """Return top-K semantically similar notes."""
        vector = self.ollama.embed(query)
        results = self.qdrant.search(vector, top_k)
        return [
            {
                "title": r["payload"]["title"],
                "folder_path": r["payload"]["folder_path"],
                "folder_id": r["payload"]["folder_id"],
                "score": r["score"],
            }
            for r in results
        ]


def get_vector_service() -> VectorService:
    """Factory: create VectorService from environment variables."""
    qdrant = QdrantClient(
        base_url=os.getenv("QDRANT_BASE_URL", "http://localhost:6333"),
        collection=os.getenv("QDRANT_COLLECTION", "joplin_notes"),
    )
    ollama = OllamaClient(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("EMBED_MODEL", "bge-m3:latest"),
    )
    return VectorService(qdrant, ollama)
