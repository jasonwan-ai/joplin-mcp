import json
import pytest
from joplin_mcp.wiki.ingest import build_classification_prompt, parse_classification_response


class TestBuildClassificationPrompt:
    def test_includes_title_and_body(self):
        prompt = build_classification_prompt("Attention Mechanism", "Self-attention allows...")
        assert "Attention Mechanism" in prompt
        assert "Self-attention allows" in prompt

    def test_includes_json_format_instruction(self):
        prompt = build_classification_prompt("X", "Y")
        assert "domain" in prompt
        assert "concept_title" in prompt
        assert "action" in prompt


class TestParseClassificationResponse:
    def test_parses_valid_json(self):
        raw = json.dumps({
            "domain": "AI/ML",
            "concept_title": "Attention Mechanism",
            "action": "new",
            "concept_body": "## Summary\nAttention is..."
        })
        result = parse_classification_response(raw)
        assert result["domain"] == "AI/ML"
        assert result["action"] == "new"

    def test_extracts_json_from_fenced_block(self):
        raw = 'Here is:\n```json\n{"domain": "Theology", "concept_title": "Grace", "action": "new", "concept_body": "## Summary\\nGrace is..."}\n```'
        result = parse_classification_response(raw)
        assert result["domain"] == "Theology"

    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_classification_response("not json at all")

    def test_raises_on_missing_required_fields(self):
        raw = json.dumps({"domain": "AI/ML"})
        with pytest.raises(ValueError, match="Missing required fields"):
            parse_classification_response(raw)
