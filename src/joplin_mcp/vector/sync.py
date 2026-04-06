"""Fire-and-forget vector index sync helpers.

Called after each note CRUD operation to keep Qdrant in sync.
All functions swallow exceptions — the cron re-sync job is the backstop.
"""
import time

from joplin_mcp.notebook_utils import _compute_notebook_path, get_notebook_map_cached
from joplin_mcp.vector.qdrant_client import joplin_id_to_uuid
from joplin_mcp.vector.vector_service import get_vector_service


def try_sync_note(note_id: str) -> None:
    """Fetch note from Joplin and upsert into Qdrant. Swallows all errors."""
    try:
        from joplin_mcp.fastmcp_server import get_joplin_client
        client = get_joplin_client()
        note = client.get_note(note_id, fields="id,title,body,parent_id,updated_time")

        title = getattr(note, "title", "") or ""
        body = getattr(note, "body", "") or ""
        parent_id = getattr(note, "parent_id", "") or ""
        updated_time = getattr(note, "updated_time", None) or int(time.time() * 1000)

        nb_map = get_notebook_map_cached()
        folder_path = _compute_notebook_path(parent_id, nb_map, sep="/") or "Unknown"

        get_vector_service().upsert_note(
            note_id=note_id,
            title=title,
            body=body,
            folder_path=folder_path,
            folder_id=parent_id,
            updated_time=updated_time,
        )
    except Exception:
        pass  # Cron sync will recover via updated_time


def try_delete_from_index(note_id: str) -> None:
    """Remove a note from Qdrant. Swallows all errors."""
    try:
        point_id = joplin_id_to_uuid(note_id)
        get_vector_service().qdrant.delete([point_id])
    except Exception:
        pass  # Cron sync will reconcile orphaned points
