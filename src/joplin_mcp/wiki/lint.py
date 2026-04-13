"""Wiki health report: orphaned pages, domains missing _index."""
import os
import re
from typing import List

from joplin_mcp.fastmcp_server import get_joplin_client
from joplin_mcp.notebook_utils import get_notebook_id_by_name


def is_index_note(title: str) -> bool:
    """Check if a note title is an index note (starts with underscore)."""
    return title.startswith("_")


def extract_outgoing_links(body: str) -> List[str]:
    """Return list of Joplin note IDs (32-char hex) and [[wiki links]] from body."""
    ids = re.findall(r":/([a-f0-9]{32})", body)
    wiki = re.findall(r"\[\[(.+?)\]\]", body)
    return ids + wiki


def lint_wiki() -> str:
    """Scan all wiki concept notes and return a markdown health report."""
    client = get_joplin_client()
    wiki_name = os.getenv("WIKI_NOTEBOOK_NAME", "📚 Wiki")
    wiki_id = get_notebook_id_by_name(wiki_name)

    all_folders = client.get_all_folders()
    domain_folders = [
        f for f in all_folders
        if getattr(f, "parent_id", "") == wiki_id
        and not getattr(f, "title", "").startswith("_")
    ]

    orphans = []
    domains_missing_index = []

    for folder in domain_folders:
        folder_id = str(folder.id)
        notes = client.get_all_notes(notebook_id=folder_id, fields="id,title,body")
        note_list = list(notes)
        titles = {getattr(n, "title", "") for n in note_list}

        if "_index" not in titles:
            domains_missing_index.append(getattr(folder, "title", folder_id))

        for note in note_list:
            title = getattr(note, "title", "")
            if is_index_note(title):
                continue
            body = note.body or ""
            outgoing = extract_outgoing_links(body)
            if not outgoing:
                orphans.append(f"- [{title}](:{note.id}) (no outgoing links)")

    lines = ["# Wiki Health Report\n"]
    lines.append(f"## Orphaned Concept Notes ({len(orphans)})")
    lines += orphans if orphans else ["_None found_"]
    lines.append(f"\n## Domains Missing _index ({len(domains_missing_index)})")
    if domains_missing_index:
        lines += [f"- {d}" for d in domains_missing_index]
    else:
        lines.append("_None_")

    return "\n".join(lines)
