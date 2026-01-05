"""Integration tests for OpenAI API compatible endpoints."""

from fastapi.testclient import TestClient


class TestOpenAIModelsEndpoint:
    """Tests for /v1/models endpoint."""

    def test_list_models_returns_openai_format(self, test_client: TestClient):
        """Test that /v1/models returns OpenAI format."""
        response = test_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()

        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_list_models_includes_model_details(self, test_client: TestClient):
        """Test that model details match OpenAI format."""
        response = test_client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()

        if data["data"]:  # If there are models
            model = data["data"][0]
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"
            assert "created" in model
            assert "owned_by" in model


class TestOpenAIModelEndpoint:
    """Tests for /v1/models/{model_name} endpoint."""

    def test_get_specific_model(self, test_client: TestClient):
        """Test getting a specific model in OpenAI format."""
        response = test_client.get("/v1/models/test-model")
        assert response.status_code == 200
        data = response.json()

        assert data["id"] == "test-model"
        assert data["object"] == "model"
        assert "created" in data
        assert data["owned_by"] == "rkllama"

    def test_get_nonexistent_model(self, test_client: TestClient):
        """Test getting a nonexistent model."""
        response = test_client.get("/v1/models/nonexistent-model")
        assert response.status_code == 404


class TestOpenAIChatCompletionsEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    def test_chat_completions_validates_model(self, test_client: TestClient):
        """Test that chat completions requires a model."""
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        # Should fail validation without model
        assert response.status_code == 422

    def test_chat_completions_validates_messages(self, test_client: TestClient):
        """Test that chat completions requires messages."""
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model"
            }
        )
        # Should fail validation without messages
        assert response.status_code == 422


class TestOpenAICompletionsEndpoint:
    """Tests for /v1/completions endpoint."""

    def test_completions_validates_model(self, test_client: TestClient):
        """Test that completions requires a model."""
        response = test_client.post(
            "/v1/completions",
            json={
                "prompt": "Hello"
            }
        )
        # Should fail validation without model
        assert response.status_code == 422

    def test_completions_validates_prompt(self, test_client: TestClient):
        """Test that completions requires a prompt."""
        response = test_client.post(
            "/v1/completions",
            json={
                "model": "test-model"
            }
        )
        # Should fail validation without prompt
        assert response.status_code == 422


class TestOpenAIEmbeddingsEndpoint:
    """Tests for /v1/embeddings endpoint."""

    def test_embeddings_validates_model(self, test_client: TestClient):
        """Test that embeddings requires a model."""
        response = test_client.post(
            "/v1/embeddings",
            json={
                "input": "Hello world"
            }
        )
        # Should fail validation without model
        assert response.status_code == 422

    def test_embeddings_validates_input(self, test_client: TestClient):
        """Test that embeddings requires input."""
        response = test_client.post(
            "/v1/embeddings",
            json={
                "model": "test-model"
            }
        )
        # Should fail validation without input
        assert response.status_code == 422
