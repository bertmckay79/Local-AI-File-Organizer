"""
Unit tests for the Ollama client.
"""

import pytest
import json
from unittest.mock import patch, MagicMock

from core.ollama_client import OllamaClient


@pytest.fixture
def client():
    return OllamaClient(
        server_url="http://localhost:11434",
        model="llama3.1",
        timeout=30,
        temperature=0.1,
        max_tokens=100,
    )


class TestOllamaClient:

    def test_init(self, client):
        """Test client initialization."""
        assert client.server_url == "http://localhost:11434"
        assert client.model == "llama3.1"
        assert client.timeout == 30

    def test_update_settings(self, client):
        """Test updating client settings."""
        client.update_settings(model="mistral", temperature=0.5)
        assert client.model == "mistral"
        assert client.temperature == 0.5

    @patch("core.ollama_client.requests.get")
    def test_is_available_true(self, mock_get, client):
        """Test server availability check (available)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert client.is_available() is True

    @patch("core.ollama_client.requests.get")
    def test_is_available_false(self, mock_get, client):
        """Test server availability check (unavailable)."""
        import requests
        mock_get.side_effect = requests.ConnectionError()

        assert client.is_available() is False

    @patch("core.ollama_client.requests.get")
    def test_list_models(self, mock_get, client):
        """Test listing models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.1"},
                {"name": "mistral"},
            ]
        }
        mock_get.return_value = mock_response

        models = client.list_models()
        assert "llama3.1" in models
        assert "mistral" in models

    @patch("core.ollama_client.requests.post")
    def test_chat_success(self, mock_post, client):
        """Test successful chat request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": '{"category": "Documents"}'}
        }
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        result = client.chat(messages)
        assert result == '{"category": "Documents"}'

    @patch("core.ollama_client.requests.post")
    def test_classify_file(self, mock_post, client):
        """Test file classification."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": json.dumps({"category": "Documents"})}
        }
        mock_post.return_value = mock_response

        file_info = {"file_name": "report.pdf", "extension": ".pdf"}
        categories = ["Documents", "Pictures", "Miscellaneous"]

        result = client.classify_file(file_info, categories)
        assert result == "Documents"

    @patch("core.ollama_client.requests.post")
    def test_classify_file_rejects_unknown_category(self, mock_post, client):
        """Unknown AI categories return None for the review workflow to handle."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": json.dumps({"category": "UnknownCategory"})}
        }
        mock_post.return_value = mock_response

        file_info = {"file_name": "unknown.xyz", "extension": ".xyz"}
        categories = ["Documents", "Pictures", "Miscellaneous"]

        result = client.classify_file(file_info, categories)
        assert result is None

    @patch("core.ollama_client.requests.post")
    def test_classify_file_case_insensitive(self, mock_post, client):
        """Test case-insensitive category matching."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": json.dumps({"category": "documents"})}
        }
        mock_post.return_value = mock_response

        file_info = {"file_name": "file.pdf", "extension": ".pdf"}
        categories = ["Documents", "Pictures", "Miscellaneous"]

        result = client.classify_file(file_info, categories)
        assert result == "Documents"


class TestFreeTextChat:
    """The Settings chat diagnostic should not request classification JSON."""

    @patch("core.ollama_client.requests.post")
    def test_qwen3_chat_uses_plain_text_and_disables_thinking(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Hello!"}}
        mock_post.return_value = mock_response
        client = OllamaClient(model="qwen3.5:4b", max_tokens=100)

        assert client.test_chat("Hello") == "Hello!"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "qwen3.5:4b"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert "format" not in payload
        assert payload["think"] is False
        assert payload["options"]["num_predict"] >= 768

    @patch("core.ollama_client.requests.post")
    def test_nonreasoning_model_does_not_receive_thinking_override(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Ready"}}
        mock_post.return_value = mock_response
        client = OllamaClient(model="qwen2.5:3b")

        assert client.test_chat("Are you ready?") == "Ready"
        payload = mock_post.call_args.kwargs["json"]
        assert "format" not in payload
        assert "think" not in payload

    def test_cloud_chat_uses_plain_text_without_json_response_format(self):
        client = OllamaClient(
            ai_provider="cloud", cloud_api_key="dummy-token",
            cloud_model="example-model"
        )
        with patch("core.ollama_client.requests.post") as mock_post:
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {
                "choices": [{"message": {"content": "Cloud ready"}}]
            }
            mock_post.return_value = response
            assert client.test_chat("Hello") == "Cloud ready"
            payload = mock_post.call_args.kwargs["json"]
            assert "response_format" not in payload
            assert payload["max_tokens"] >= 768
