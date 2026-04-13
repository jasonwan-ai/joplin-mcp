# LLM Wiki — Design Spec
_Date: 2026-04-12 | Repo: joplin-mcp_

## Overview

Implement the Karpathy LLM Wiki pattern inside the existing joplin-mcp stack. Joplin becomes a persistent, LLM-maintained wiki that compounds over time — replacing ad-hoc note accumulation with a structured, cross-linked knowledge base.

---

## Architecture

### Joplin Structure

```
📚 Wiki/
  _schema              ← auto-maintained: domain list + concept count per domain
  _sources/            ← ingested raw notes, moved here post-ingest (immutable)
  <Domain>/
    _index             ← links to all concept notes in this domain + 1-line summaries
    <Concept>          ← atomic concept note
```

**Concept note format:**
```markdown
## Summary
2-3 sentence definition.

## Key Points
- ...

## Related
- [[Other Concept]]
- [[Cross-domain Concept]]
```

### Ingest Flow

```
Unsorted (Joplin notebook)
    │
    ▼  wiki_ingest_batch (nightly cron or on-demand)
    ├── 1. Fetch note from Unsorted
    ├── 2. Ollama @ jarvis classifies: domain + concept title + new/update?
    ├── 3. Read _schema to check for existing close match
    ├── 4a. New concept → create_note in 📚 Wiki/<Domain>/<Concept>
    ├── 4b. Existing → edit_note to merge + add cross-links
    ├── 5. Update 📚 Wiki/<Domain>/_index
    ├── 6. Update 📚 Wiki/_schema
    └── 7. move_note source to 📚 Wiki/_sources/
```

---

## New Module: `joplin-mcp/src/joplin_mcp/wiki/`

| File | Responsibility |
|---|---|
| `ollama_chat.py` | Wraps Ollama `/api/chat` for completions. Reads `WIKI_OLLAMA_BASE_URL` env var (default `http://jarvis:11434`). Separate from existing `vector/ollama_client.py` which only does embeddings. |
| `schema.py` | Reads/writes `📚 Wiki/_schema`. Provides `get_schema()` and `update_schema(domain, concepts)`. Schema note is plain markdown — human-readable and LLM-readable. |
| `ingest.py` | Core ingest logic. Fetches note via Joplin REST, calls Ollama to classify, creates/merges wiki note, updates index + schema, moves source. |
| `lint.py` | Scans all notes under `📚 Wiki/`, asks Ollama to flag: orphaned pages (no links in or out), contradictions between concept notes, domains with no index, concepts that should be merged. Returns a markdown report. |

---

## New MCP Tools: `joplin-mcp/src/joplin_mcp/tools/wiki.py`

Thin wrappers over the wiki module. Registered via `tools/__init__.py`.

| Tool | Args | What it does |
|---|---|---|
| `wiki_ingest` | `note_id: str` | Ingest one note into wiki |
| `wiki_ingest_batch` | `limit: int = 10` | Ingest up to N notes from Unsorted |
| `wiki_lint` | — | Return markdown health report |
| `wiki_get_schema` | — | Return current domain map |

---

## Cron Integration (joplin project)

New files in `/home/jasonwan/docker/joplin`:
- `src/wiki_scripts.py` — imports `joplin_mcp.wiki.ingest` directly (same machine, no HTTP needed)
- `cron_scripts/wiki_cron.sh` — time-guarded wrapper, runs at 01:00 AEST nightly (after daily cron at 00:10)

---

## LLM Stack

| Operation | Model | Where |
|---|---|---|
| Ingest classification + concept generation | Ollama (strong 16GB model e.g. `qwen3:14b`) | `http://jarvis:11434` |
| Lint analysis | Ollama | `http://jarvis:11434` |
| Deep synthesis (manual, on-demand) | GPT-5.3 via GitHub Copilot + joplin-mcp | Copilot chat |
| Future: lint/self-improvement loop | Hermes (Nous Research) | Wire in after core is stable |

**Env vars added to joplin-mcp config:**
```
WIKI_OLLAMA_BASE_URL=http://jarvis:11434
WIKI_OLLAMA_MODEL=qwen3:14b
WIKI_NOTEBOOK_NAME=📚 Wiki
WIKI_UNSORTED_NOTEBOOK=Unsorted
```

---

## Copilot Deep Synthesis Prompt

Stored as Joplin note `📚 Wiki/_copilot_deep_synthesis`. Used manually when running domain-level migration or cross-domain synthesis passes. See full prompt in that note (created during migration phase).

**Supports:**
- One domain per invocation
- Subagent-per-subfolder for large notebooks (HF Papers, GitHub Treasures)
- Pre-populated skip list: `_Joplin`, `_Tasks and Lists`, `📚 Wiki`, `TODO_TODAY`, `Week Kanban`, `Daily Checklist`, `Weekly Schedule`, meeting minute notebooks

---

## Scope Boundaries

**In scope:**
- `wiki/` module (ollama_chat, schema, ingest, lint)
- `tools/wiki.py` MCP tools
- `tools/__init__.py` update
- `wiki_scripts.py` + `wiki_cron.sh` in joplin project
- Env var additions
- Copilot prompt note in Joplin

**Explicitly out of scope:**
- Migration of existing notes (done manually via Copilot prompt)
- Hermes wiring (after core is stable)
- Vector/Qdrant changes (wiki pages auto-indexed by existing sync)
- Any changes to existing MCP tools

---

## Success Criteria

- `wiki_ingest(note_id)` creates a well-formed concept note in the correct domain
- `wiki_ingest_batch(10)` processes 10 Unsorted notes without error
- `wiki_lint()` returns a readable report identifying at least orphaned pages
- `wiki_get_schema()` returns an accurate domain + concept count map
- Nightly cron runs cleanly, logs to `logs/wiki_cron.log`
- Copilot prompt successfully migrates one test notebook (e.g. `Theology`) into wiki format
