"""Tests for tools/brain_dump.py"""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest


def _get_tool_fn(tool):
    """Unwrap FastMCP tool decorator to get underlying function."""
    if hasattr(tool, "fn"):
        return tool.fn
    return tool


# === get_notebook_tree ===

class TestGetNotebookTree:
    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump.get_joplin_client")
    async def test_returns_indented_tree(self, mock_get_client):
        from joplin_mcp.tools.brain_dump import get_notebook_tree

        mock_notebooks = [
            MagicMock(id="root1", title="Music", parent_id=None),
            MagicMock(id="child1", title="FL Studio", parent_id="root1"),
            MagicMock(id="child2", title="Choir", parent_id="root1"),
            MagicMock(id="grandchild1", title="Projects", parent_id="child1"),
            MagicMock(id="root2", title="Work", parent_id=None),
        ]
        mock_client = MagicMock()
        mock_client.get_all_notebooks.return_value = mock_notebooks
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(get_notebook_tree)
        result = await fn()

        lines = result.split("\n")
        assert "Music" in lines
        assert "  FL Studio" in lines
        assert "  Choir" in lines
        assert "    Projects" in lines
        assert "Work" in lines

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump.get_joplin_client")
    async def test_returns_message_when_no_notebooks(self, mock_get_client):
        from joplin_mcp.tools.brain_dump import get_notebook_tree

        mock_client = MagicMock()
        mock_client.get_all_notebooks.return_value = []
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(get_notebook_tree)
        result = await fn()
        assert "No notebooks found" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump.get_joplin_client")
    async def test_children_sorted_alphabetically(self, mock_get_client):
        from joplin_mcp.tools.brain_dump import get_notebook_tree

        mock_notebooks = [
            MagicMock(id="root", title="Music", parent_id=None),
            MagicMock(id="c2", title="Violin", parent_id="root"),
            MagicMock(id="c1", title="Choir", parent_id="root"),
        ]
        mock_client = MagicMock()
        mock_client.get_all_notebooks.return_value = mock_notebooks
        mock_get_client.return_value = mock_client

        fn = _get_tool_fn(get_notebook_tree)
        result = await fn()
        lines = result.split("\n")
        choir_idx = lines.index("  Choir")
        violin_idx = lines.index("  Violin")
        assert choir_idx < violin_idx


# === semantic_search ===

class TestSemanticSearch:
    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump.get_vector_service")
    async def test_returns_formatted_results(self, mock_get_service):
        from joplin_mcp.tools.brain_dump import semantic_search

        mock_service = MagicMock()
        mock_service.search.return_value = [
            {"title": "FL Studio reverb", "folder_path": "Music/FL Studio", "folder_id": "fid1", "score": 0.92},
            {"title": "Guitar chords", "folder_path": "Music/Guitar", "folder_id": "fid2", "score": 0.80},
        ]
        mock_get_service.return_value = mock_service

        fn = _get_tool_fn(semantic_search)
        result = await fn(query="reverb pedal ideas", top_k=5)

        assert "FL Studio reverb" in result
        assert "Music/FL Studio" in result
        assert "fid1" in result
        assert "0.920" in result
        assert "Guitar chords" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump.get_vector_service")
    async def test_returns_no_results_message_when_empty(self, mock_get_service):
        from joplin_mcp.tools.brain_dump import semantic_search

        mock_service = MagicMock()
        mock_service.search.return_value = []
        mock_get_service.return_value = mock_service

        fn = _get_tool_fn(semantic_search)
        result = await fn(query="something totally new")
        assert "No" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump.get_vector_service")
    async def test_passes_query_and_top_k_to_service(self, mock_get_service):
        from joplin_mcp.tools.brain_dump import semantic_search

        mock_service = MagicMock()
        mock_service.search.return_value = []
        mock_get_service.return_value = mock_service

        fn = _get_tool_fn(semantic_search)
        await fn(query="test query", top_k=3)
        mock_service.search.assert_called_once_with("test query", 3)


# === create_and_embed_note ===

class TestCreateAndEmbedNote:
    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump.get_vector_service")
    @patch("joplin_mcp.tools.brain_dump.get_joplin_client")
    @patch("joplin_mcp.tools.brain_dump.get_notebook_id_by_name")
    @patch("joplin_mcp.tools.brain_dump.get_notebook_map_cached")
    @patch("joplin_mcp.tools.brain_dump._compute_notebook_path")
    async def test_creates_note_and_embeds(
        self, mock_compute_path, mock_get_map, mock_get_id, mock_get_client, mock_get_service
    ):
        from joplin_mcp.tools.brain_dump import create_and_embed_note

        mock_get_id.return_value = "folder_id_123"
        mock_client = MagicMock()
        mock_client.add_note.return_value = "new_note_id_456"
        mock_get_client.return_value = mock_client
        mock_get_map.return_value = {}
        mock_compute_path.return_value = "Music/FL Studio/Ideas"
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        fn = _get_tool_fn(create_and_embed_note)
        result = await fn(title="FL Studio reverb ideas", notebook_path="Music/FL Studio/Ideas", body="Some content")

        mock_client.add_note.assert_called_once_with(
            title="FL Studio reverb ideas",
            body="Some content",
            parent_id="folder_id_123",
        )
        mock_service.upsert_note.assert_called_once()
        upsert_kwargs = mock_service.upsert_note.call_args[1]
        assert upsert_kwargs["note_id"] == "new_note_id_456"
        assert upsert_kwargs["folder_path"] == "Music/FL Studio/Ideas"
        assert "FL Studio reverb ideas" in result
        assert "new_note_id_456" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump.get_notebook_id_by_name")
    async def test_returns_error_when_notebook_not_found(self, mock_get_id):
        from joplin_mcp.tools.brain_dump import create_and_embed_note

        mock_get_id.side_effect = Exception("Notebook not found")

        fn = _get_tool_fn(create_and_embed_note)
        result = await fn(title="Test", notebook_path="Nonexistent/Path", body="body")
        assert "Error" in result
        assert "Nonexistent/Path" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump.get_vector_service")
    @patch("joplin_mcp.tools.brain_dump.get_joplin_client")
    @patch("joplin_mcp.tools.brain_dump.get_notebook_id_by_name")
    @patch("joplin_mcp.tools.brain_dump.get_notebook_map_cached")
    @patch("joplin_mcp.tools.brain_dump._compute_notebook_path")
    async def test_note_saved_even_if_qdrant_fails(
        self, mock_compute_path, mock_get_map, mock_get_id, mock_get_client, mock_get_service
    ):
        """Joplin write must succeed even if Qdrant upsert fails."""
        from joplin_mcp.tools.brain_dump import create_and_embed_note

        mock_get_id.return_value = "folder_id"
        mock_client = MagicMock()
        mock_client.add_note.return_value = "note_id"
        mock_get_client.return_value = mock_client
        mock_get_map.return_value = {}
        mock_compute_path.return_value = "Music"
        mock_service = MagicMock()
        mock_service.upsert_note.side_effect = Exception("Qdrant is down")
        mock_get_service.return_value = mock_service

        fn = _get_tool_fn(create_and_embed_note)
        result = await fn(title="Test note", notebook_path="Music", body="body")

        # Note was saved
        mock_client.add_note.assert_called_once()
        # Result indicates success (upsert failure is silent)
        assert "note_id" in result
