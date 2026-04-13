from unittest.mock import MagicMock, patch
import pytest
from joplin_mcp.wiki.ollama_chat import OllamaChatClient


class TestOllamaChatClient:
    def test_returns_text_response(self):
        client = OllamaChatClient("http://jarvis:11434", "qwen3:14b")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"message": {"content": "Hello from Ollama"}}
            mock_post.return_value = mock_resp
            result = client.chat("Say hello")
            assert result == "Hello from Ollama"

    def test_sends_correct_endpoint_and_model(self):
        client = OllamaChatClient("http://jarvis:11434", "qwen3:14b")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"message": {"content": "ok"}}
            mock_post.return_value = mock_resp
            client.chat("test prompt")
            url = mock_post.call_args[0][0]
            payload = mock_post.call_args[1]["json"]
            assert url == "http://jarvis:11434/api/chat"
            assert payload["model"] == "qwen3:14b"
            assert payload["stream"] is False

    def test_formats_user_message(self):
        client = OllamaChatClient("http://jarvis:11434", "qwen3:14b")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"message": {"content": "ok"}}
            mock_post.return_value = mock_resp
            client.chat("classify this note")
            payload = mock_post.call_args[1]["json"]
            assert payload["messages"] == [{"role": "user", "content": "classify this note"}]

    def test_raises_on_http_error(self):
        client = OllamaChatClient("http://jarvis:11434", "qwen3:14b")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("503 Service Unavailable")
            mock_post.return_value = mock_resp
            with pytest.raises(Exception, match="503 Service Unavailable"):
                client.chat("test")
