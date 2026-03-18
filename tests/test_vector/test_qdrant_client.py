"""Tests for vector/qdrant_client.py"""
from unittest.mock import MagicMock, patch
import pytest
from joplin_mcp.vector.qdrant_client import QdrantClient, joplin_id_to_uuid


class TestJoplinIdToUuid:
    def test_converts_32_char_hex_to_uuid(self):
        joplin_id = "06020c71bb2f45c3a93a58318cb4cb99"
        result = joplin_id_to_uuid(joplin_id)
        assert result == "06020c71-bb2f-45c3-a93a-58318cb4cb99"

    def test_raises_on_invalid_hex(self):
        with pytest.raises(ValueError):
            joplin_id_to_uuid("not-a-valid-id")


class TestQdrantClientEnsureCollection:
    def test_creates_collection_when_not_exists(self):
        client = QdrantClient("http://qdrant:6333", "joplin_notes")
        with patch("requests.get") as mock_get, patch("requests.put") as mock_put:
            mock_get.return_value = MagicMock(status_code=404)
            mock_put.return_value = MagicMock(status_code=200)
            mock_put.return_value.raise_for_status = MagicMock()
            client.ensure_collection(vector_size=1024)
            mock_put.assert_called_once()
            call_json = mock_put.call_args[1]["json"]
            assert call_json["vectors"]["size"] == 1024
            assert call_json["vectors"]["distance"] == "Cosine"

    def test_skips_creation_when_collection_exists(self):
        client = QdrantClient("http://qdrant:6333", "joplin_notes")
        with patch("requests.get") as mock_get, patch("requests.put") as mock_put:
            mock_get.return_value = MagicMock(status_code=200)
            client.ensure_collection()
            mock_put.assert_not_called()


class TestQdrantClientUpsert:
    def test_upserts_single_point(self):
        client = QdrantClient("http://qdrant:6333", "joplin_notes")
        vector = [0.1] * 1024
        payload = {"title": "Test", "folder_path": "Music/FL Studio", "folder_id": "abc", "updated_time": 1000}
        with patch("requests.put") as mock_put:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_put.return_value = mock_resp
            client.upsert("test-uuid-1234", vector, payload)
            mock_put.assert_called_once()
            body = mock_put.call_args[1]["json"]
            assert len(body["points"]) == 1
            assert body["points"][0]["id"] == "test-uuid-1234"
            assert body["points"][0]["vector"] == vector
            assert body["points"][0]["payload"] == payload


class TestQdrantClientSearch:
    def test_returns_results_with_score(self):
        client = QdrantClient("http://qdrant:6333", "joplin_notes")
        fake_results = [
            {"id": "uuid1", "score": 0.92, "payload": {"title": "FL Studio reverb", "folder_path": "Music/FL Studio", "folder_id": "f1", "updated_time": 1000}},
            {"id": "uuid2", "score": 0.87, "payload": {"title": "Guitar chords", "folder_path": "Music/Guitar", "folder_id": "f2", "updated_time": 2000}},
        ]
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"result": fake_results}
            mock_post.return_value = mock_resp
            results = client.search([0.1] * 1024, top_k=5)
            assert len(results) == 2
            assert results[0]["score"] == 0.92
            assert results[0]["payload"]["title"] == "FL Studio reverb"


class TestQdrantClientScrollAllIds:
    def test_scrolls_single_page(self):
        client = QdrantClient("http://qdrant:6333", "joplin_notes")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "result": {
                    "points": [{"id": "uuid1"}, {"id": "uuid2"}],
                    "next_page_offset": None,
                }
            }
            mock_post.return_value = mock_resp
            ids = client.scroll_all_ids()
            assert ids == ["uuid1", "uuid2"]

    def test_scrolls_multiple_pages(self):
        client = QdrantClient("http://qdrant:6333", "joplin_notes")
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if call_count == 1:
                resp.json.return_value = {
                    "result": {"points": [{"id": "uuid1"}], "next_page_offset": "cursor1"}
                }
            else:
                resp.json.return_value = {
                    "result": {"points": [{"id": "uuid2"}], "next_page_offset": None}
                }
            return resp

        with patch("requests.post", side_effect=side_effect):
            ids = client.scroll_all_ids()
            assert ids == ["uuid1", "uuid2"]


class TestQdrantClientDelete:
    def test_deletes_points(self):
        client = QdrantClient("http://qdrant:6333", "joplin_notes")
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp
            client.delete(["uuid1", "uuid2"])
            mock_post.assert_called_once()
            body = mock_post.call_args[1]["json"]
            assert body["points"] == ["uuid1", "uuid2"]

    def test_skips_empty_list(self):
        client = QdrantClient("http://qdrant:6333", "joplin_notes")
        with patch("requests.post") as mock_post:
            client.delete([])
            mock_post.assert_not_called()
