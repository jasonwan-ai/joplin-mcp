"""Core wiki ingest: classify note with Ollama, create/merge concept note, update index + schema."""
import json
import os
import re
from typing import Any, Dict, Optional

from joplin_mcp.fastmcp_server import get_joplin_client
from joplin_mcp.notebook_utils import get_notebook_id_by_name
from joplin_mcp.wiki.ollama_chat import get_ollama_chat_client
from joplin_mcp.wiki.schema import update_schema

REQUIRED_FIELDS = {"domain", "concept_title", "action", "concept_body"}


def build_classification_prompt(title: str, body: str) -> str:
    return f"""You are a wiki curator. Classify the following note into a knowledge domain and extract a clean concept.

Note title: {title}
Note body:
{body}

Respond with ONLY valid JSON matching this schema:
{{
  "domain": "string (e.g. AI/ML, Theology, Philosophy, Economics)",
  "concept_title": "string — short, noun-phrase title for the concept",
  "action": "new | update",
  "concept_body": "string — well-formatted markdown with ## Summary, ## Key Points, ## Related sections"
}}"""


def parse_classification_response(raw: str) -> Dict[str, Any]:
    """Extract and validate JSON from Ollama response. Handles ```json``` fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    json_str = fenced.group(1) if fenced else raw.strip()
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from Ollama response: {e}") from e
    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    return data


def _get_or_create_notebook(client, title: str, parent_id: str) -> str:
    all_folders = client.get_all_folders()
    for folder in all_folders:
        if getattr(folder, "title", "") == title and getattr(folder, "parent_id", "") == parent_id:
            return str(folder.id)
    new_id = client.add_folder(title=title, parent_id=parent_id)
    return str(new_id)


def _update_domain_index(client, domain_notebook_id: str, concept_title: str, concept_note_id: str) -> None:
    all_notes = client.get_all_notes(notebook_id=domain_notebook_id, fields="id,title,body")
    index_note = next((n for n in all_notes if getattr(n, "title", "") == "_index"), None)
    link_line = f"- [{concept_title}](:{concept_note_id})"
    if index_note:
        new_body = (index_note.body or "") + "\n" + link_line
        client.modify_note(str(index_note.id), body=new_body)
    else:
        client.add_note(title="_index", body=link_line, parent_id=domain_notebook_id)


def ingest_note(note_id: str) -> Dict[str, Any]:
    """Ingest one note into the wiki. Returns result dict with domain, concept_title, action."""
    client = get_joplin_client()
    ollama = get_ollama_chat_client()

    wiki_notebook_name = os.getenv("WIKI_NOTEBOOK_NAME", "📚 Wiki")

    note = client.get_note(note_id, fields="id,title,body,parent_id")
    title = note.title or ""
    body = note.body or ""

    prompt = build_classification_prompt(title, body)
    raw_response = ollama.chat(prompt)
    classification = parse_classification_response(raw_response)

    domain = classification["domain"]
    concept_title = classification["concept_title"]
    action = classification["action"]
    concept_body = classification["concept_body"]

    wiki_id = get_notebook_id_by_name(wiki_notebook_name)
    domain_notebook_id = _get_or_create_notebook(client, domain, wiki_id)

    # Always assign concept_note_id before use; default to creating a new note
    concept_note_id: Optional[str] = None

    if action == "update":
        all_notes = client.get_all_notes(notebook_id=domain_notebook_id, fields="id,title,body")
        existing = next((n for n in all_notes if getattr(n, "title", "") == concept_title), None)
        if existing:
            merged_body = (existing.body or "") + "\n\n---\n\n" + concept_body
            client.modify_note(str(existing.id), body=merged_body)
            concept_note_id = str(existing.id)
        else:
            # Fall through to "new" if no existing note found
            action = "new"

    if action == "new" or concept_note_id is None:
        concept_note_id = str(client.add_note(
            title=concept_title, body=concept_body, parent_id=domain_notebook_id
        ))
        update_schema(domain, +1)
        action = "new"

    _update_domain_index(client, domain_notebook_id, concept_title, concept_note_id)

    sources_id = _get_or_create_notebook(client, "_sources", wiki_id)
    client.modify_note(note_id, parent_id=sources_id)

    return {"domain": domain, "concept_title": concept_title, "action": action, "concept_note_id": concept_note_id}


def ingest_batch(limit: int = 10) -> list:
    """Ingest up to `limit` notes from the Unsorted notebook."""
    client = get_joplin_client()
    unsorted_name = os.getenv("WIKI_UNSORTED_NOTEBOOK", "Unsorted")
    unsorted_id = get_notebook_id_by_name(unsorted_name)
    notes = client.get_all_notes(notebook_id=unsorted_id, fields="id,title", limit=limit)
    results = []
    for note in list(notes)[:limit]:
        try:
            result = ingest_note(str(note.id))
            results.append({"note_id": str(note.id), **result})
        except Exception as e:
            results.append({"note_id": str(note.id), "error": str(e)})
    return results
