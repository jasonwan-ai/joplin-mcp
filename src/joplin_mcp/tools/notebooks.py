"""Notebook tools for Joplin MCP."""
from typing import Annotated, Optional

from pydantic import Field

from joplin_mcp.fastmcp_server import (
    ItemType,
    JoplinIdType,
    RequiredStringType,
    _compute_notebook_path,
    create_tool,
    format_creation_success,
    format_delete_success,
    format_item_list,
    format_update_success,
    get_joplin_client,
    get_notebook_id_by_name,
    get_notebook_map_cached,
    invalidate_notebook_map_cache,
)


# === NOTEBOOK TOOLS ===


@create_tool("list_notebooks", "List notebooks")
async def list_notebooks() -> str:
    """Returns flat list of all notebooks with IDs, titles, parent_id, and creation dates."""
    client = get_joplin_client()
    fields_list = "id,title,created_time,updated_time,parent_id"
    notebooks = client.get_all_notebooks(fields=fields_list)
    return format_item_list(notebooks, ItemType.notebook)


@create_tool("create_notebook", "Create notebook")
async def create_notebook(
    title: Annotated[RequiredStringType, Field(description="Notebook title")],
    parent_id: Annotated[
        Optional[str], Field(description="Parent notebook ID (optional)")
    ] = None,
) -> str:
    """Create a notebook. Use parent_id for sub-notebooks."""

    client = get_joplin_client()
    notebook_kwargs = {"title": title}
    if parent_id:
        notebook_kwargs["parent_id"] = parent_id.strip()

    notebook = client.add_notebook(**notebook_kwargs)
    # Invalidate notebook path cache to reflect new structure immediately
    invalidate_notebook_map_cache()
    return format_creation_success(ItemType.notebook, title, str(notebook))


@create_tool("update_notebook", "Update notebook")
async def update_notebook(
    notebook_id: Annotated[JoplinIdType, Field(description="Notebook ID to update")],
    title: Annotated[RequiredStringType, Field(description="New notebook title")],
) -> str:
    """Update notebook title."""
    client = get_joplin_client()
    client.modify_notebook(notebook_id, title=title)
    # Invalidate cache in case the notebook moved/renamed
    invalidate_notebook_map_cache()
    return format_update_success(ItemType.notebook, notebook_id)


@create_tool("delete_notebook", "Delete notebook")
async def delete_notebook(
    notebook_id: Annotated[JoplinIdType, Field(description="Notebook ID to delete")],
) -> str:
    """Permanent. Deletes the notebook and all notes inside it."""
    client = get_joplin_client()
    client.delete_notebook(notebook_id)
    # Invalidate cache since structure changed
    invalidate_notebook_map_cache()
    return format_delete_success(ItemType.notebook, notebook_id)


@create_tool("get_folder_id", "Get folder ID")
async def get_folder_id(
    name: Annotated[
        RequiredStringType,
        Field(
            description=(
                "Notebook name or hierarchical path to look up. "
                "Use a plain name for an exact title match (e.g. 'Work'), "
                "or a '/'-separated path for nested notebooks "
                "(e.g. 'Professional & Knowledge/Claude Code')."
            )
        ),
    ],
) -> str:
    """Resolve a notebook name or path to its ID (e.g. 'Projects/Work/Tasks')."""
    notebook_id = get_notebook_id_by_name(name)
    nb_map = get_notebook_map_cached()
    full_path = _compute_notebook_path(notebook_id, nb_map)
    title = (nb_map.get(notebook_id) or {}).get("title") or name

    return (
        f"Notebook found:\n"
        f"  ID:    {notebook_id}\n"
        f"  Title: {title}\n"
        f"  Path:  {full_path}"
    )
