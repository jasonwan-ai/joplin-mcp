"""MCP tool wrappers for the wiki module."""
from typing import Annotated

from pydantic import Field

from joplin_mcp.fastmcp_server import create_tool, JoplinIdType
from joplin_mcp.wiki import ingest, lint, schema


@create_tool("wiki_ingest", "Ingest one note into wiki")
async def _wiki_ingest(
    note_id: Annotated[JoplinIdType, Field(description="Note ID from Unsorted to ingest")],
) -> str:
    """Ingest one note into wiki. Classifies via Ollama, creates/merges concept note, moves source."""
    result = ingest.ingest_note(note_id)
    return (
        f"Ingested as **{result['concept_title']}** in domain **{result['domain']}** "
        f"(action: {result['action']}, note: {result['concept_note_id']})"
    )


@create_tool("wiki_ingest_batch", "Batch ingest notes from Unsorted into wiki")
async def _wiki_ingest_batch(
    limit: Annotated[int, Field(description="Max notes to ingest (default: 10)")] = 10,
) -> str:
    """Ingest up to N notes from Unsorted. Returns summary."""
    results = ingest.ingest_batch(limit)
    lines = [f"Processed {len(results)} notes:"]
    for r in results:
        if "error" in r:
            lines.append(f"- {r['note_id']}: ERROR — {r['error']}")
        else:
            lines.append(f"- {r['concept_title']} → {r['domain']} ({r['action']})")
    return "\n".join(lines)


@create_tool("wiki_lint", "Run wiki health check")
async def _wiki_lint() -> str:
    """Return markdown health report: orphaned pages, domains missing _index."""
    return lint.lint_wiki()


@create_tool("wiki_get_schema", "Get wiki domain map")
async def _wiki_get_schema() -> str:
    """Return current domain → concept count map from _schema note."""
    data = schema.get_schema()
    if not data:
        return "Wiki schema is empty — no domains indexed yet."
    lines = ["| Domain | Concepts |", "|---|---|"]
    for domain in sorted(data):
        lines.append(f"| {domain} | {data[domain]} |")
    return "\n".join(lines)
