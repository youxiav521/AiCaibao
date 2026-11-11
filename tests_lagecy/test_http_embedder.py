import pytest
from unittest.mock import MagicMock, patch
import requests
# Adjust the import path if necessary based on your project root
from src.applications.rag.adapters.embeddings.http_embedder import HttpEmbedder

class TestHttpEmbedder:
    @pytest.fixture
    def embedder(self):
        """Fixture to create an HttpEmbedder instance for tests."""
        # The class currently hardcodes the URL in __init__, ignoring the argument
        return HttpEmbedder(url="http://test-url/embed", timeout=5)

    def test_initialization(self):
        """Test initialization parameters."""
        embedder = HttpEmbedder(url="http://custom-url", timeout=10)
        assert embedder.url == "http://custom-url"
        assert embedder.timeout == 10

        # Test default
        embedder_default = HttpEmbedder()
        # The default depends on env var or hardcoded default in class
        # We assume no env var is set in test env for this specific check
        # or we can check if it's one of the expected defaults
        assert embedder_default.url in ["http://localhost:8002/embed", "http://10.192.72.11:8002/embed"]

    @patch("src.applications.rag.adapters.embeddings.http_embedder.requests.post")
    def test_embed_query_success(self, mock_post, embedder):
        """Test successful single query embedding."""
        # Mock response
        mock_response = MagicMock()
        expected_vector = [0.1, 0.2, 0.3]
        
        # Case 1: Direct list response
        mock_response.json.return_value = expected_vector
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = embedder.embed_query("hello")
        assert result == expected_vector
        mock_post.assert_called_with(
            embedder.url, 
            json={"text": "hello"}, 
            timeout=embedder.timeout
        )

        # Case 2: Nested list [[0.1, ...]]
        mock_response.json.return_value = [expected_vector]
        result = embedder.embed_query("hello")
        assert result == expected_vector

        # Case 3: Dict with 'vector'
        mock_response.json.return_value = {"vector": expected_vector}
        result = embedder.embed_query("hello")
        assert result == expected_vector

    @patch("src.applications.rag.adapters.embeddings.http_embedder.requests.post")
    def test_embed_texts_success(self, mock_post, embedder):
        """Test successful batch embedding."""
        mock_response = MagicMock()
        expected_vectors = [[0.1, 0.1], [0.2, 0.2]]
        
        # Case 1: Direct list response
        mock_response.json.return_value = expected_vectors
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = embedder.embed_texts(["t1", "t2"])
        assert result == expected_vectors
        mock_post.assert_called_with(
            embedder.url, 
            json={"text": ["t1", "t2"]}, 
            timeout=embedder.timeout
        )

        # Case 2: Dict with 'vectors'
        mock_response.json.return_value = {"vectors": expected_vectors}
        result = embedder.embed_texts(["t1", "t2"])
        assert result == expected_vectors

        # Case 3: Dict with 'data'
        mock_response.json.return_value = {"data": expected_vectors}
        result = embedder.embed_texts(["t1", "t2"])
        assert result == expected_vectors

    def test_embed_texts_empty(self, embedder):
        """Test embedding empty list returns empty result without making request."""
        with patch("src.applications.rag.adapters.embeddings.http_embedder.requests.post") as mock_post:
            result = embedder.embed_texts([])
            assert result == []
            mock_post.assert_not_called()

    @patch("src.applications.rag.adapters.embeddings.http_embedder.requests.post")
    def test_api_error_handling(self, mock_post, embedder):
        """Test handling of HTTP errors or connection exceptions."""
        
        # Simulate HTTP Error (e.g. 500)
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_post.return_value = mock_response

        result = embedder.embed_query("test")
        assert result == []  # Expect empty list on error based on your code

        # Simulate Connection Error
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        result = embedder.embed_texts(["test"])
        assert result == []

    @patch("src.applications.rag.adapters.embeddings.http_embedder.requests.post")
    def test_unexpected_response_format(self, mock_post, embedder):
        """Test handling of unexpected JSON structure."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Unexpected dict keys
        mock_response.json.return_value = {"unknown_key": "some_value"}
        
        result = embedder.embed_query("test")
        assert result == []
        
        result = embedder.embed_texts(["test"])
        assert result == []