"""Tests for wiki health report."""
import pytest

from joplin_mcp.wiki.lint import is_index_note, extract_outgoing_links


class TestIsIndexNote:
    def test_index_note_detected(self):
        assert is_index_note("_index") is True
        assert is_index_note("_schema") is True
        assert is_index_note("_sources") is True

    def test_concept_note_not_index(self):
        assert is_index_note("Attention Mechanism") is False
        assert is_index_note("Grace") is False


class TestExtractOutgoingLinks:
    def test_finds_joplin_links(self):
        body = "See also [Transformer](:/abc123def456abc123def456abc12345)"
        links = extract_outgoing_links(body)
        assert "abc123def456abc123def456abc12345" in links

    def test_finds_wiki_links(self):
        body = "Related: [[Attention Mechanism]]"
        links = extract_outgoing_links(body)
        assert "Attention Mechanism" in links

    def test_empty_body_returns_empty_list(self):
        assert extract_outgoing_links("") == []

    def test_finds_multiple_joplin_links(self):
        body = "See [A](:/abc123def456abc123def456abc12345) and [B](:/def456abc123def456abc123def45678)"
        links = extract_outgoing_links(body)
        assert "abc123def456abc123def456abc12345" in links
        assert "def456abc123def456abc123def45678" in links
        assert len(links) == 2

    def test_finds_multiple_wiki_links(self):
        body = "See [[First]] and [[Second]]"
        links = extract_outgoing_links(body)
        assert "First" in links
        assert "Second" in links
        assert len(links) == 2

    def test_mixed_joplin_and_wiki_links(self):
        body = "[Note](:/abc123def456abc123def456abc12345) and [[WikiPage]]"
        links = extract_outgoing_links(body)
        assert "abc123def456abc123def456abc12345" in links
        assert "WikiPage" in links
        assert len(links) == 2

    def test_ignores_invalid_joplin_links(self):
        body = "Invalid link [A](:/toolort) and valid [B](:/abc123def456abc123def456abc12345)"
        links = extract_outgoing_links(body)
        assert "abc123def456abc123def456abc12345" in links
        # Should not include the short one
        assert "toolort" not in links

    def test_wiki_links_with_spaces(self):
        body = "See [[Multi Word Page]]"
        links = extract_outgoing_links(body)
        assert "Multi Word Page" in links

    def test_wiki_links_with_special_chars(self):
        body = "See [[Page (v2)]] and [[Item/Subitem]]"
        links = extract_outgoing_links(body)
        assert "Page (v2)" in links
        assert "Item/Subitem" in links
