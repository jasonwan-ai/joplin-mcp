"""Reads and writes the 📚 Wiki/_schema concept-count index note."""
import os
import re
from typing import Dict

from joplin_mcp.fastmcp_server import get_joplin_client
from joplin_mcp.notebook_utils import get_notebook_id_by_name

SCHEMA_NOTE_TITLE = "_schema"


def parse_schema_body(body: str) -> Dict[str, int]:
    """Parse the schema note body into a domain -> concept count mapping."""
    result: Dict[str, int] = {}
    for match in re.finditer(r"^## (.+)\nconcepts: (\d+)", body, re.MULTILINE):
        result[match.group(1)] = int(match.group(2))
    return result


def render_schema_body(data: Dict[str, int]) -> str:
    """Render a domain -> concept count mapping into schema note body."""
    if not data:
        return ""
    lines = []
    for domain in sorted(data.keys()):
        lines.append(f"## {domain}\nconcepts: {data[domain]}")
    return "\n\n".join(lines) + "\n"


def _get_wiki_notebook_id() -> str:
    """Get the ID of the 📚 Wiki notebook."""
    notebook_name = os.getenv("WIKI_NOTEBOOK_NAME", "📚 Wiki")
    return get_notebook_id_by_name(notebook_name)


def _find_schema_note_id(client, wiki_notebook_id: str):
    """Find the _schema note ID within the wiki notebook."""
    results = client.search(query=f'title:"{SCHEMA_NOTE_TITLE}"', fields="id,title,parent_id", limit=10)
    for note in results:
        if getattr(note, "parent_id", None) == wiki_notebook_id and getattr(note, "title", "") == SCHEMA_NOTE_TITLE:
            return str(note.id)
    return None


def get_schema() -> Dict[str, int]:
    """Load the current schema (domain -> concept count) from Joplin."""
    client = get_joplin_client()
    wiki_id = _get_wiki_notebook_id()
    note_id = _find_schema_note_id(client, wiki_id)
    if not note_id:
        return {}
    note = client.get_note(note_id, fields="body")
    return parse_schema_body(note.body or "")


def update_schema(domain: str, delta: int) -> None:
    """Update the schema by incrementing/decrementing a domain's concept count."""
    client = get_joplin_client()
    wiki_id = _get_wiki_notebook_id()
    note_id = _find_schema_note_id(client, wiki_id)
    if note_id:
        note = client.get_note(note_id, fields="body")
        data = parse_schema_body(note.body or "")
    else:
        data = {}
    data[domain] = max(0, data.get(domain, 0) + delta)
    new_body = render_schema_body(data)
    if note_id:
        client.modify_note(note_id, body=new_body)
    else:
        client.add_note(title=SCHEMA_NOTE_TITLE, body=new_body, parent_id=wiki_id)
