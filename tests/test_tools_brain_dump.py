"""Tests for tools/brain_dump.py"""
from unittest.mock import MagicMock, patch
import pytest


def _get_tool_fn(tool):
    if hasattr(tool, "fn"):
        return tool.fn
    return tool


# === get_notebook_tree ===

class TestGetNotebookTree:
    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump._get_all_notebooks")
    async def test_returns_indented_tree(self, mock_get_nbs):
        from joplin_mcp.tools.brain_dump import get_notebook_tree

        mock_get_nbs.return_value = [
            {"id": "root1", "title": "Music", "parent_id": ""},
            {"id": "child1", "title": "FL Studio", "parent_id": "root1"},
            {"id": "child2", "title": "Choir", "parent_id": "root1"},
            {"id": "grandchild1", "title": "Projects", "parent_id": "child1"},
            {"id": "root2", "title": "Work", "parent_id": ""},
        ]

        fn = _get_tool_fn(get_notebook_tree)
        result = await fn()
        lines = result.split("\n")
        assert "Music" in lines
        assert "  FL Studio" in lines
        assert "  Choir" in lines
        assert "    Projects" in lines
        assert "Work" in lines

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump._get_all_notebooks")
    async def test_returns_message_when_no_notebooks(self, mock_get_nbs):
        from joplin_mcp.tools.brain_dump import get_notebook_tree

        mock_get_nbs.return_value = []
        fn = _get_tool_fn(get_notebook_tree)
        result = await fn()
        assert "No notebooks found" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump._get_all_notebooks")
    async def test_children_sorted_alphabetically(self, mock_get_nbs):
        from joplin_mcp.tools.brain_dump import get_notebook_tree

        mock_get_nbs.return_value = [
            {"id": "root", "title": "Music", "parent_id": ""},
            {"id": "c2", "title": "Violin", "parent_id": "root"},
            {"id": "c1", "title": "Choir", "parent_id": "root"},
        ]
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
    @patch("joplin_mcp.tools.brain_dump._joplin_post")
    @patch("joplin_mcp.tools.brain_dump._get_all_notebooks")
    async def test_creates_note_and_embeds(self, mock_get_nbs, mock_post, mock_get_service):
        from joplin_mcp.tools.brain_dump import create_and_embed_note

        mock_get_nbs.return_value = [
            {"id": "nb1", "title": "Music", "parent_id": ""},
            {"id": "nb2", "title": "FL Studio", "parent_id": "nb1"},
            {"id": "nb3", "title": "Ideas", "parent_id": "nb2"},
        ]
        mock_post.return_value = {"id": "new_note_id_456"}
        mock_service = MagicMock()
        mock_get_service.return_value = mock_service

        fn = _get_tool_fn(create_and_embed_note)
        result = await fn(title="FL Studio reverb ideas", notebook_path="Music/FL Studio/Ideas", body="Some content")

        mock_post.assert_called_once_with("/notes", {"title": "FL Studio reverb ideas", "body": "Some content", "parent_id": "nb3"})
        mock_service.upsert_note.assert_called_once()
        upsert_kwargs = mock_service.upsert_note.call_args[1]
        assert upsert_kwargs["note_id"] == "new_note_id_456"
        assert upsert_kwargs["folder_path"] == "Music/FL Studio/Ideas"
        assert "FL Studio reverb ideas" in result
        assert "new_note_id_456" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump._get_all_notebooks")
    async def test_returns_error_when_notebook_not_found(self, mock_get_nbs):
        from joplin_mcp.tools.brain_dump import create_and_embed_note

        mock_get_nbs.return_value = []

        fn = _get_tool_fn(create_and_embed_note)
        result = await fn(title="Test", notebook_path="Nonexistent/Path", body="body")
        assert "Error" in result
        assert "Nonexistent/Path" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.tools.brain_dump.get_vector_service")
    @patch("joplin_mcp.tools.brain_dump._joplin_post")
    @patch("joplin_mcp.tools.brain_dump._get_all_notebooks")
    async def test_note_saved_even_if_qdrant_fails(self, mock_get_nbs, mock_post, mock_get_service):
        from joplin_mcp.tools.brain_dump import create_and_embed_note

        mock_get_nbs.return_value = [{"id": "nb1", "title": "Music", "parent_id": ""}]
        mock_post.return_value = {"id": "note_id"}
        mock_service = MagicMock()
        mock_service.upsert_note.side_effect = Exception("Qdrant is down")
        mock_get_service.return_value = mock_service

        fn = _get_tool_fn(create_and_embed_note)
        result = await fn(title="Test note", notebook_path="Music", body="body")

        mock_post.assert_called_once()
        assert "note_id" in result
