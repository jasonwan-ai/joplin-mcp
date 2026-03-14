"""Tests for vector/vector_service.py"""
from unittest.mock import MagicMock, patch
import pytest
from joplin_mcp.vector.vector_service import VectorService, build_embed_text


class TestBuildEmbedText:
    def test_combines_title_and_body(self):
        result = build_embed_text("My Title", "Body content here")
        assert result == "My Title\n\nBody content here"

    def test_truncates_body_at_4000_chars(self):
        long_body = "x" * 5000
        result = build_embed_text("Title", long_body)
        assert result == "Title\n\n" + "x" * 4000

    def test_handles_empty_body(self):
        result = build_embed_text("Just a title", "")
        assert result == "Just a title\n\n"


class TestVectorServiceUpsertNote:
    def test_embeds_and_upserts(self):
        mock_qdrant = MagicMock()
        mock_ollama = MagicMock()
        mock_ollama.embed.return_value = [0.1] * 1024
        service = VectorService(mock_qdrant, mock_ollama)

        service.upsert_note(
            note_id="06020c71bb2f45c3a93a58318cb4cb99",
            title="FL Studio ideas",
            body="some reverb settings",
            folder_path="Music/FL Studio/Ideas",
            folder_id="folder123",
            updated_time=1710000000,
        )

        # Should embed title + body
        mock_ollama.embed.assert_called_once_with("FL Studio ideas\n\nsome reverb settings")

        # Should upsert with UUID-formatted ID
        mock_qdrant.upsert.assert_called_once()
        call_kwargs = mock_qdrant.upsert.call_args[1]
        assert call_kwargs["point_id"] == "06020c71-bb2f-45c3-a93a-58318cb4cb99"
        assert call_kwargs["vector"] == [0.1] * 1024
        assert call_kwargs["payload"]["title"] == "FL Studio ideas"
        assert call_kwargs["payload"]["folder_path"] == "Music/FL Studio/Ideas"
        assert call_kwargs["payload"]["folder_id"] == "folder123"
        assert call_kwargs["payload"]["updated_time"] == 1710000000


class TestVectorServiceSearch:
    def test_returns_formatted_results(self):
        mock_qdrant = MagicMock()
        mock_ollama = MagicMock()
        mock_ollama.embed.return_value = [0.5] * 1024
        mock_qdrant.search.return_value = [
            {"score": 0.92, "payload": {"title": "FL Studio reverb", "folder_path": "Music/FL Studio", "folder_id": "f1", "updated_time": 1000}},
            {"score": 0.80, "payload": {"title": "Guitar recording", "folder_path": "Music/Guitar", "folder_id": "f2", "updated_time": 2000}},
        ]
        service = VectorService(mock_qdrant, mock_ollama)

        results = service.search("reverb pedal ideas", top_k=5)

        mock_ollama.embed.assert_called_once_with("reverb pedal ideas")
        mock_qdrant.search.assert_called_once_with([0.5] * 1024, 5)

        assert len(results) == 2
        assert results[0]["title"] == "FL Studio reverb"
        assert results[0]["folder_path"] == "Music/FL Studio"
        assert results[0]["folder_id"] == "f1"
        assert results[0]["score"] == 0.92

    def test_returns_empty_list_when_no_results(self):
        mock_qdrant = MagicMock()
        mock_ollama = MagicMock()
        mock_ollama.embed.return_value = [0.0] * 1024
        mock_qdrant.search.return_value = []
        service = VectorService(mock_qdrant, mock_ollama)
        results = service.search("completely new topic")
        assert results == []


class TestGetVectorService:
    def test_creates_service_from_env(self):
        import os
        from joplin_mcp.vector.vector_service import get_vector_service
        with patch.dict(os.environ, {
            "QDRANT_BASE_URL": "http://test-qdrant:6333",
            "QDRANT_COLLECTION": "test_notes",
            "OLLAMA_BASE_URL": "http://test-ollama:11434",
            "EMBED_MODEL": "bge-m3:latest",
        }):
            service = get_vector_service()
            assert service.qdrant.base_url == "http://test-qdrant:6333"
            assert service.qdrant.collection == "test_notes"
            assert service.ollama.base_url == "http://test-ollama:11434"
            assert service.ollama.model == "bge-m3:latest"
