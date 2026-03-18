"""Tests for vector/ollama_client.py"""
from unittest.mock import MagicMock, patch
import pytest
from joplin_mcp.vector.ollama_client import OllamaClient


class TestOllamaClientEmbed:
    def test_returns_embedding_vector(self):
        client = OllamaClient("http://ollama:11434", "bge-m3:latest")
        fake_embedding = [0.1] * 1024
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"embedding": fake_embedding}
            mock_post.return_value = mock_resp
            result = client.embed("test text")
            assert result == fake_embedding
            assert len(result) == 1024

    def test_sends_correct_model_and_prompt(self):
        client = OllamaClient("http://ollama:11434", "bge-m3:latest")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"embedding": [0.0] * 1024}
            mock_post.return_value = mock_resp
            client.embed("my brain dump")
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "bge-m3:latest"
            assert call_kwargs["json"]["prompt"] == "my brain dump"

    def test_calls_correct_endpoint(self):
        client = OllamaClient("http://ollama:11434", "bge-m3:latest")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"embedding": [0.0] * 1024}
            mock_post.return_value = mock_resp
            client.embed("test")
            url = mock_post.call_args[0][0]
            assert url == "http://ollama:11434/api/embeddings"

    def test_raises_on_http_error(self):
        client = OllamaClient("http://ollama:11434", "bge-m3:latest")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("500 Server Error")
            mock_post.return_value = mock_resp
            with pytest.raises(Exception, match="500 Server Error"):
                client.embed("test")
