"""Tests for tools/wiki.py"""
from unittest.mock import patch
import pytest


def _get_tool_fn(tool):
    if hasattr(tool, "fn"):
        return tool.fn
    return tool


class TestWikiIngestTool:
    @pytest.mark.asyncio
    @patch("joplin_mcp.wiki.ingest.ingest_note")
    async def test_wiki_ingest_returns_formatted_result(self, mock_ingest):
        from joplin_mcp.tools.wiki import _wiki_ingest

        mock_ingest.return_value = {
            "domain": "AI/ML",
            "concept_title": "Attention",
            "action": "new",
            "concept_note_id": "a" * 32,
        }
        fn = _get_tool_fn(_wiki_ingest)
        result = await fn("b" * 32)
        assert "AI/ML" in result
        assert "Attention" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.wiki.schema.get_schema")
    async def test_wiki_get_schema_returns_table(self, mock_schema):
        from joplin_mcp.tools.wiki import _wiki_get_schema

        mock_schema.return_value = {"AI/ML": 14, "Theology": 3}
        fn = _get_tool_fn(_wiki_get_schema)
        result = await fn()
        assert "AI/ML" in result
        assert "14" in result

    @pytest.mark.asyncio
    @patch("joplin_mcp.wiki.schema.get_schema")
    async def test_wiki_get_schema_empty(self, mock_schema):
        from joplin_mcp.tools.wiki import _wiki_get_schema

        mock_schema.return_value = {}
        fn = _get_tool_fn(_wiki_get_schema)
        result = await fn()
        assert "empty" in result.lower()
