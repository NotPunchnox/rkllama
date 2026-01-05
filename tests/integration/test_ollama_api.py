"""Integration tests for Ollama API endpoints."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestOllamaTagsEndpoint:
    """Tests for /api/tags endpoint."""

    def test_tags_returns_models_list(self, test_client: TestClient):
        """Test that /api/tags returns a list of models."""
        response = test_client.get("/api/tags")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_tags_includes_model_details(self, test_client: TestClient):
        """Test that model details are included in response."""
        response = test_client.get("/api/tags")
        assert response.status_code == 200
        data = response.json()

        if data["models"]:  # If there are models
            model = data["models"][0]
            assert "name" in model
            assert "model" in model
            assert "details" in model


class TestOllamaPsEndpoint:
    """Tests for /api/ps endpoint."""

    def test_ps_returns_running_models(self, test_client: TestClient):
        """Test that /api/ps returns running models."""
        response = test_client.get("/api/ps")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)


class TestOllamaVersionEndpoint:
    """Tests for /api/version endpoint."""

    def test_version_returns_version_string(self, test_client: TestClient):
        """Test that /api/version returns a version."""
        response = test_client.get("/api/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)


class TestOllamaLoadUnloadEndpoints:
    """Tests for /api/load and /api/unload endpoints."""

    def test_load_model_success(self, test_client: TestClient, mock_worker_manager: MagicMock):
        """Test loading a model via /api/load."""
        mock_worker_manager.exists_model_loaded.return_value = False

        with patch("rkllama.server.routers.ollama.load_model", return_value=(None, None)):
            response = test_client.post(
                "/api/load",
                json={"model": "test-model"}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_load_already_loaded_model(self, test_client: TestClient, mock_worker_manager: MagicMock):
        """Test loading an already loaded model."""
        mock_worker_manager.exists_model_loaded.return_value = True

        response = test_client.post(
            "/api/load",
            json={"model": "test-model"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "already loaded" in data.get("message", "").lower()

    def test_unload_model_success(self, test_client: TestClient, mock_worker_manager: MagicMock):
        """Test unloading a model via /api/unload."""
        mock_worker_manager.exists_model_loaded.return_value = True

        response = test_client.post(
            "/api/unload",
            json={"model": "test-model"}
        )

        assert response.status_code == 200
        mock_worker_manager.stop_worker.assert_called()

    def test_unload_not_loaded_model(self, test_client: TestClient, mock_worker_manager: MagicMock):
        """Test unloading a model that isn't loaded."""
        mock_worker_manager.exists_model_loaded.return_value = False

        response = test_client.post(
            "/api/unload",
            json={"model": "test-model"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "not loaded" in data.get("message", "").lower()


class TestOllamaShowEndpoint:
    """Tests for /api/show endpoint."""

    def test_show_existing_model(self, test_client: TestClient):
        """Test showing info for an existing model."""
        response = test_client.post(
            "/api/show",
            json={"name": "test-model"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "details" in data

    def test_show_nonexistent_model(self, test_client: TestClient):
        """Test showing info for a nonexistent model."""
        response = test_client.post(
            "/api/show",
            json={"name": "nonexistent-model"}
        )

        assert response.status_code == 404

    def test_show_missing_model_name(self, test_client: TestClient):
        """Test show endpoint with missing model name."""
        response = test_client.post(
            "/api/show",
            json={}
        )

        assert response.status_code == 400
