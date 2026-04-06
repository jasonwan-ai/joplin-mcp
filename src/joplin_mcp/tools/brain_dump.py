"""Brain dump tools: notebook tree and semantic search."""
import os
from typing import Annotated

import requests
from pydantic import Field

from joplin_mcp.fastmcp_server import create_tool
from joplin_mcp.vector.vector_service import get_vector_service


def _joplin_base_url() -> str:
    host = os.getenv("JOPLIN_HOST", "joplin-data-api")
    port = os.getenv("JOPLIN_PORT", "41185")
    return f"http://{host}:{port}"


def _joplin_get(path: str, **params) -> dict:
    url = f"{_joplin_base_url()}{path}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _get_all_notebooks() -> list:
    """Fetch all notebooks from Joplin via raw HTTP (paginated)."""
    results = []
    page = 1
    while True:
        data = _joplin_get("/folders", fields="id,title,parent_id", limit=100, page=page)
        results.extend(data.get("items", []))
        if not data.get("has_more"):
            break
        page += 1
    return results


@create_tool("get_notebook_tree", "Get notebook tree")
async def get_notebook_tree() -> str:
    """Get notebook hierarchy as an indented tree, alphabetically sorted."""
    notebooks = _get_all_notebooks()

    if not notebooks:
        return "No notebooks found."

    children: dict = {}
    for nb in notebooks:
        parent = nb.get("parent_id") or ""
        children.setdefault(parent, []).append(nb)

    for level in children.values():
        level.sort(key=lambda nb: nb.get("title", "").lower())

    lines: list = []

    def render(parent_id: str, depth: int) -> None:
        for nb in children.get(parent_id, []):
            lines.append("  " * depth + nb.get("title", "Untitled"))
            render(nb["id"], depth + 1)

    render("", 0)
    return "\n".join(lines) if lines else "No notebooks found."


@create_tool("semantic_search", "Semantic search")
async def semantic_search(
    query: Annotated[str, Field(description="Text to search for semantically")],
    top_k: Annotated[int, Field(description="Number of results to return (default: 5)")] = 5,
) -> str:
    """Embedding-based similarity search (not keyword). Returns title, folder path, folder_id, and score."""
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
