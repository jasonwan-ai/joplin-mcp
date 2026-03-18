"""Qdrant REST API client for joplin-mcp (runs inside Docker container)."""
import uuid
import requests


def joplin_id_to_uuid(joplin_id: str) -> str:
    """Convert 32-char hex Joplin note ID to UUID string (raises ValueError if invalid)."""
    return str(uuid.UUID(joplin_id))


class QdrantClient:
    """Minimal Qdrant REST API client."""

    def __init__(self, base_url: str, collection: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection = collection
        self.timeout = timeout

    def ensure_collection(self, vector_size: int = 1024) -> None:
        """Create the collection if it doesn't exist (idempotent)."""
        url = f"{self.base_url}/collections/{self.collection}"
        resp = requests.get(url, timeout=self.timeout)
        if resp.status_code == 404:
            requests.put(
                url,
                json={"vectors": {"size": vector_size, "distance": "Cosine"}},
                timeout=self.timeout,
            ).raise_for_status()

    def upsert(self, point_id: str, vector: list, payload: dict) -> None:
        """Upsert a single point."""
        url = f"{self.base_url}/collections/{self.collection}/points"
        requests.put(
            url,
            json={"points": [{"id": point_id, "vector": vector, "payload": payload}]},
            timeout=self.timeout,
        ).raise_for_status()

    def upsert_batch(self, points: list) -> None:
        """Upsert a list of {id, vector, payload} dicts in one call."""
        if not points:
            return
        url = f"{self.base_url}/collections/{self.collection}/points"
        requests.put(url, json={"points": points}, timeout=self.timeout).raise_for_status()

    def search(self, vector: list, top_k: int = 5) -> list:
        """Return top-K nearest neighbours with score and payload."""
        url = f"{self.base_url}/collections/{self.collection}/points/search"
        resp = requests.post(
            url,
            json={"vector": vector, "limit": top_k, "with_payload": True},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("result", [])

    def scroll_all_ids(self) -> list:
        """Return all point IDs in the collection."""
        url = f"{self.base_url}/collections/{self.collection}/points/scroll"
        all_ids = []
        offset = None
        while True:
            body: dict = {"limit": 100, "with_payload": False, "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            resp = requests.post(url, json=body, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json().get("result", {})
            points = data.get("points", [])
            all_ids.extend(p["id"] for p in points)
            offset = data.get("next_page_offset")
            if not offset:
                break
        return all_ids

    def delete(self, point_ids: list) -> None:
        """Delete points by ID. No-op if list is empty."""
        if not point_ids:
            return
        url = f"{self.base_url}/collections/{self.collection}/points/delete"
        requests.post(
            url,
            json={"points": point_ids},
            timeout=self.timeout,
        ).raise_for_status()
