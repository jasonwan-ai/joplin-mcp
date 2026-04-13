"""Tests for wiki schema parsing and rendering."""
import pytest

from joplin_mcp.wiki.schema import parse_schema_body, render_schema_body


class TestParseSchemaBody:
    def test_parses_domains_and_counts(self):
        body = "## AI/ML\nconcepts: 14\n\n## Theology\nconcepts: 3\n"
        result = parse_schema_body(body)
        assert result == {"AI/ML": 14, "Theology": 3}

    def test_empty_body_returns_empty_dict(self):
        assert parse_schema_body("") == {}

    def test_domain_with_zero_concepts(self):
        body = "## NewDomain\nconcepts: 0\n"
        assert parse_schema_body(body) == {"NewDomain": 0}


class TestRenderSchemaBody:
    def test_renders_sorted_domains(self):
        data = {"Theology": 3, "AI/ML": 14}
        body = render_schema_body(data)
        assert body == "## AI/ML\nconcepts: 14\n\n## Theology\nconcepts: 3\n"

    def test_empty_dict_renders_empty_string(self):
        assert render_schema_body({}) == ""
