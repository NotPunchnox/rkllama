"""Integration tests for Modelfile CRUD API endpoints."""

from fastapi.testclient import TestClient


class TestModelfileGetEndpoint:
    """Tests for GET /api/modelfile/{model} endpoint."""

    def test_get_modelfile_success(self, test_client: TestClient):
        """Test getting all properties from a Modelfile."""
        response = test_client.get("/api/modelfile/test-model")
        assert response.status_code == 200
        data = response.json()

        assert "model" in data
        assert "path" in data
        assert "properties" in data
        assert data["model"] == "test-model"
        assert isinstance(data["properties"], dict)

    def test_get_modelfile_includes_properties(self, test_client: TestClient):
        """Test that returned properties match expected values."""
        response = test_client.get("/api/modelfile/test-model")
        assert response.status_code == 200
        data = response.json()

        props = data["properties"]
        assert "TEMPERATURE" in props
        assert "NUM_CTX" in props
        assert "HUGGINGFACE_PATH" in props

    def test_get_modelfile_nonexistent_model(self, test_client: TestClient):
        """Test getting Modelfile for nonexistent model."""
        response = test_client.get("/api/modelfile/nonexistent-model")
        assert response.status_code == 404


class TestModelfilePropertyGetEndpoint:
    """Tests for GET /api/modelfile/{model}/{property} endpoint."""

    def test_get_specific_property(self, test_client: TestClient):
        """Test getting a specific property."""
        response = test_client.get("/api/modelfile/test-model/temperature")
        assert response.status_code == 200
        data = response.json()

        assert data["model"] == "test-model"
        assert data["property"] == "TEMPERATURE"
        assert "value" in data

    def test_get_property_case_insensitive(self, test_client: TestClient):
        """Test that property names are case insensitive."""
        response_lower = test_client.get("/api/modelfile/test-model/temperature")
        response_upper = test_client.get("/api/modelfile/test-model/TEMPERATURE")

        assert response_lower.status_code == 200
        assert response_upper.status_code == 200
        assert response_lower.json()["value"] == response_upper.json()["value"]

    def test_get_nonexistent_property(self, test_client: TestClient):
        """Test getting a property that doesn't exist."""
        response = test_client.get("/api/modelfile/test-model/NONEXISTENT")
        assert response.status_code == 404


class TestModelfilePatchEndpoint:
    """Tests for PATCH /api/modelfile/{model} endpoint."""

    def test_update_single_property(self, test_client: TestClient):
        """Test updating a single property."""
        response = test_client.patch(
            "/api/modelfile/test-model",
            json={"properties": {"temperature": 0.9}}
        )
        assert response.status_code == 200
        data = response.json()

        assert data["properties"]["TEMPERATURE"] == "0.9" or data["properties"]["TEMPERATURE"] == 0.9

    def test_update_multiple_properties(self, test_client: TestClient):
        """Test updating multiple properties at once."""
        response = test_client.patch(
            "/api/modelfile/test-model",
            json={"properties": {"temperature": 0.8, "top_k": 50}}
        )
        assert response.status_code == 200
        data = response.json()

        assert "TEMPERATURE" in data["properties"]
        assert "TOP_K" in data["properties"]

    def test_update_invalid_property_name(self, test_client: TestClient):
        """Test updating with invalid property name."""
        response = test_client.patch(
            "/api/modelfile/test-model",
            json={"properties": {"INVALID_PROPERTY": "value"}}
        )
        assert response.status_code == 400

    def test_update_invalid_temperature_value(self, test_client: TestClient):
        """Test updating temperature with invalid value."""
        response = test_client.patch(
            "/api/modelfile/test-model",
            json={"properties": {"temperature": 5.0}}  # Out of range
        )
        assert response.status_code == 400

    def test_update_invalid_mirostat_value(self, test_client: TestClient):
        """Test updating mirostat with invalid value."""
        response = test_client.patch(
            "/api/modelfile/test-model",
            json={"properties": {"mirostat": 5}}  # Must be 0, 1, or 2
        )
        assert response.status_code == 400

    def test_update_nonexistent_model(self, test_client: TestClient):
        """Test updating Modelfile for nonexistent model."""
        response = test_client.patch(
            "/api/modelfile/nonexistent-model",
            json={"properties": {"temperature": 0.5}}
        )
        assert response.status_code == 404


class TestModelfileDeleteEndpoint:
    """Tests for DELETE /api/modelfile/{model}/{property} endpoint."""

    def test_delete_optional_property(self, test_client: TestClient):
        """Test deleting an optional property."""
        response = test_client.delete("/api/modelfile/test-model/MIROSTAT")
        assert response.status_code == 200
        data = response.json()

        assert "deleted_value" in data
        assert data["property"] == "MIROSTAT"

    def test_cannot_delete_required_property(self, test_client: TestClient):
        """Test that required properties cannot be deleted."""
        response = test_client.delete("/api/modelfile/test-model/FROM")
        assert response.status_code == 400

        response = test_client.delete("/api/modelfile/test-model/HUGGINGFACE_PATH")
        assert response.status_code == 400

    def test_delete_nonexistent_property(self, test_client: TestClient):
        """Test deleting a property that doesn't exist."""
        response = test_client.delete("/api/modelfile/test-model/NONEXISTENT")
        assert response.status_code == 404

    def test_delete_nonexistent_model(self, test_client: TestClient):
        """Test deleting property from nonexistent model."""
        response = test_client.delete("/api/modelfile/nonexistent-model/TEMPERATURE")
        assert response.status_code == 404
