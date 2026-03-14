"""Brain dump organiser tools: notebook tree, semantic search, create-and-embed."""
import time
from typing import Annotated, Optional

from pydantic import Field

from joplin_mcp.fastmcp_server import create_tool, get_joplin_client
from joplin_mcp.notebook_utils import (
    _compute_notebook_path,
    get_notebook_id_by_name,
    get_notebook_map_cached,
)
from joplin_mcp.vector.vector_service import get_vector_service


@create_tool("get_notebook_tree", "Get notebook tree")
async def get_notebook_tree() -> str:
    """Get the complete Joplin notebook hierarchy as an indented tree.

    Returns all notebooks as an indented tree (2 spaces per level), sorted
    alphabetically at each level. Use this before classifying a note to see
    the full folder structure available.

    Returns:
        str: Indented tree of notebook titles, one per line. Example:
            Music
              Choir
              FL Studio
                Ideas
                Projects
            Work
    """
    client = get_joplin_client()
    notebooks = client.get_all_notebooks(fields="id,title,parent_id")

    # Build parent_id → [children] map
    children: dict = {}
    for nb in notebooks or []:
        parent = getattr(nb, "parent_id", None) or ""
        children.setdefault(parent, []).append(nb)

    # Sort each level alphabetically by title
    for level in children.values():
        level.sort(key=lambda nb: getattr(nb, "title", "").lower())

    lines: list = []

    def render(parent_id: str, depth: int) -> None:
        for nb in children.get(parent_id, []):
            lines.append("  " * depth + getattr(nb, "title", "Untitled"))
            render(getattr(nb, "id", ""), depth + 1)

    render("", 0)

    if not lines:
        return "No notebooks found."
    return "\n".join(lines)


@create_tool("semantic_search", "Semantic search")
async def semantic_search(
    query: Annotated[str, Field(description="Text to search for semantically — use the note title + first paragraph of body")],
    top_k: Annotated[int, Field(description="Number of results to return (default: 5)")] = 5,
) -> str:
    """Find notes in Joplin by semantic similarity using vector search.

    Embeds the query text using bge-m3 and returns the most similar notes
    from the Qdrant vector store. Each result includes the note title,
    its Joplin folder path, the folder ID (for direct use in create_note),
    and a similarity score (0–1, higher is more similar).

    Use the returned folder_id directly when calling create_and_embed_note.

    Returns:
        str: Numbered list of results with title, folder path, folder ID, and score.
    """
    service = get_vector_service()
    results = service.search(query, top_k)

    if not results:
        return "No semantically similar notes found. Use get_notebook_tree to choose a folder."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}. {r['title']}\n"
            f"   Folder: {r['folder_path']}\n"
            f"   Folder ID: {r['folder_id']}\n"
            f"   Score: {r['score']:.3f}"
        )
    return "\n\n".join(lines)


@create_tool("create_and_embed_note", "Create and embed note")
async def create_and_embed_note(
    title: Annotated[str, Field(description="Note title")],
    notebook_path: Annotated[str, Field(description="Notebook path, e.g. 'Music/FL Studio/Ideas'. Supports nested paths.")],
    body: Annotated[str, Field(description="Note body in markdown")] = "",
) -> str:
    """Create a note in Joplin and immediately index it in the vector store.

    Calls the Joplin API directly (not the create_note tool) to capture the
    note ID, then embeds the note using bge-m3 and upserts to Qdrant so it
    is searchable in semantic_search immediately.

    If the Qdrant upsert fails (e.g. Qdrant is temporarily down), the note is
    still saved to Joplin successfully — the next incremental sync will index it.

    Returns:
        str: Success message with note title, folder path, and note ID.
    """
    # Resolve folder ID from path (supports "A/B/C" paths)
    try:
        folder_id = get_notebook_id_by_name(notebook_path)
    except Exception as exc:
        return f"Error: Could not find notebook '{notebook_path}': {exc}"

    # Create note in Joplin directly to capture the raw note ID
    client = get_joplin_client()
    try:
        note_id = client.add_note(title=title, body=body, parent_id=folder_id)
    except Exception as exc:
        return f"Error: Could not create note in Joplin: {exc}"

    # Resolve human-readable folder path for Qdrant payload
    notebook_map = get_notebook_map_cached()
    folder_path = _compute_notebook_path(folder_id, notebook_map, sep="/")

    # Embed and upsert — non-fatal if Qdrant/Ollama unavailable
    try:
        service = get_vector_service()
        service.upsert_note(
            note_id=str(note_id),
            title=title,
            body=body,
            folder_path=folder_path,
            folder_id=folder_id,
            updated_time=int(time.time() * 1000),
        )
    except Exception:
        pass  # Cron sync will recover this note via updated_time

    return f"Created note '{title}' in '{folder_path}' (ID: {note_id})"
