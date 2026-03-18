"""Brain dump organiser tools: notebook tree, semantic search, create-and-embed.

Uses raw HTTP requests to the Joplin Data API — the joplin-data-api container
(nginx proxy at :41185) injects the auth token automatically, so no token is
needed here.
"""
import os
import time
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


def _joplin_post(path: str, body: dict) -> dict:
    url = f"{_joplin_base_url()}{path}"
    resp = requests.post(url, json=body, timeout=30)
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
    return results  # list of dicts with id, title, parent_id


def _build_path_map(notebooks: list) -> dict:
    """Build {id: (full_path, id)} for all notebooks (dicts)."""
    nb_map = {nb["id"]: nb for nb in notebooks if nb.get("id")}

    def resolve(nb_id: str, seen: set) -> str:
        if nb_id in seen:
            return nb_map.get(nb_id, {}).get("title", "Unknown")
        seen.add(nb_id)
        info = nb_map.get(nb_id, {})
        parent_id = info.get("parent_id") or ""
        title = info.get("title", "Untitled")
        if parent_id and parent_id in nb_map:
            return resolve(parent_id, seen) + "/" + title
        return title

    return {nb_id: (resolve(nb_id, set()), nb_id) for nb_id in nb_map}


def _notebook_id_by_path(path: str, notebooks: list) -> str:
    """Resolve a notebook path like 'A/B/C' to its ID."""
    parts = [p.strip() for p in path.split("/") if p.strip()]
    if not parts:
        raise ValueError("Empty notebook path")

    # Build parent_id -> children map (dicts)
    by_parent: dict = {}
    for nb in notebooks:
        parent = nb.get("parent_id") or ""
        by_parent.setdefault(parent, []).append(nb)

    current_parent = ""
    current_id = None
    for part in parts:
        candidates = [
            nb for nb in by_parent.get(current_parent, [])
            if nb.get("title", "").lower() == part.lower()
        ]
        if not candidates:
            raise ValueError(f"Notebook '{part}' not found in '{path}'")
        if len(candidates) > 1:
            raise ValueError(f"Multiple notebooks named '{part}' in '{path}'")
        current_id = candidates[0]["id"]
        current_parent = current_id

    if current_id is None:
        raise ValueError(f"Could not resolve notebook path '{path}'")
    return current_id


@create_tool("get_notebook_tree", "Get notebook tree")
async def get_notebook_tree() -> str:
    """Get the complete Joplin notebook hierarchy as an indented tree.

    Returns all notebooks as an indented tree (2 spaces per level), sorted
    alphabetically at each level. Use this before classifying a note to see
    the full folder structure available.
    """
    notebooks = _get_all_notebooks()

    if not notebooks:
        return "No notebooks found."

    # Build parent -> children map
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
    """Find notes in Joplin by semantic similarity using vector search.

    Returns top-K most similar notes with title, folder path, folder ID, and score.
    Use the folder_id directly when calling create_and_embed_note.
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
    notebook_path: Annotated[str, Field(description="Notebook path, e.g. 'Music/FL Studio/Ideas'")],
    body: Annotated[str, Field(description="Note body in markdown")] = "",
) -> str:
    """Create a note in Joplin and immediately index it in the vector store.

    If Qdrant upsert fails, the note is still saved — cron sync will index it later.
    """
    try:
        notebooks = _get_all_notebooks()
        folder_id = _notebook_id_by_path(notebook_path, notebooks)
    except Exception as exc:
        return f"Error: Could not find notebook '{notebook_path}': {exc}"

    try:
        result = _joplin_post("/notes", {"title": title, "body": body, "parent_id": folder_id})
        note_id = result.get("id", "")
    except Exception as exc:
        return f"Error: Could not create note in Joplin: {exc}"

    # Resolve full path for Qdrant payload
    path_map = _build_path_map(notebooks)
    folder_path, _ = path_map.get(folder_id, (notebook_path, folder_id))

    try:
        service = get_vector_service()
        service.upsert_note(
            note_id=note_id,
            title=title,
            body=body,
            folder_path=folder_path,
            folder_id=folder_id,
            updated_time=int(time.time() * 1000),
        )
    except Exception:
        pass  # Cron sync will recover via updated_time

    return f"Created note '{title}' in '{folder_path}' (ID: {note_id})"


@create_tool("reindex_note", "Re-embed a note in the vector store")
async def reindex_note(
    note_id: Annotated[str, Field(description="Joplin note ID to re-index")],
) -> str:
    """Re-embed an existing Joplin note in Qdrant after its content has been updated.

    Fetches the note's current title, body, and parent folder from Joplin,
    then upserts the fresh embedding into Qdrant. Use this after updating a
    note via the REST API (PUT /notes/:id) to keep the vector index in sync.
    """
    try:
        note = _joplin_get(f"/notes/{note_id}", fields="id,title,body,parent_id,updated_time")
    except Exception as exc:
        return f"Error: Could not fetch note {note_id}: {exc}"

    title = note.get("title", "")
    body = note.get("body", "")
    parent_id = note.get("parent_id", "")
    updated_time = note.get("updated_time", int(time.time() * 1000))

    # Resolve folder path
    notebooks = _get_all_notebooks()
    path_map = _build_path_map(notebooks)
    folder_path, folder_id = path_map.get(parent_id, ("Unknown", parent_id))

    try:
        service = get_vector_service()
        service.upsert_note(
            note_id=note_id,
            title=title,
            body=body,
            folder_path=folder_path,
            folder_id=folder_id,
            updated_time=updated_time,
        )
    except Exception as exc:
        return f"Error: Note exists but Qdrant upsert failed: {exc}"

    return f"Re-indexed '{title}' (ID: {note_id}) in folder '{folder_path}'"
