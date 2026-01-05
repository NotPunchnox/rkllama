"""Tests for Pydantic schemas."""

import pytest

from rkllama.api.schemas.common import ModelDetails, Options, ErrorResponse
from rkllama.api.schemas.ollama import (
    ChatMessage,
    ChatRequest,
    GenerateRequest,
    LoadRequest,
    UnloadRequest,
)
from rkllama.api.schemas.openai import (
    OpenAIChatMessage,
    OpenAIChatRequest,
)
from rkllama.api.schemas.modelfile import (
    ModelfileResponse,
    ModelfilePatchRequest,
    validate_property_name,
    validate_property_value,
)


class TestCommonSchemas:
    """Tests for common schemas."""

    def test_model_details_defaults(self):
        """Test ModelDetails with defaults."""
        details = ModelDetails()
        assert details.format == "rkllm"
        assert details.families == []

    def test_options_with_alias(self):
        """Test Options accepts max_tokens alias."""
        opts = Options(max_tokens=100)
        assert opts.num_predict == 100

    def test_error_response(self):
        """Test ErrorResponse creation."""
        error = ErrorResponse(error="Something went wrong", code="ERR_001")
        assert error.error == "Something went wrong"
        assert error.code == "ERR_001"


class TestOllamaSchemas:
    """Tests for Ollama API schemas."""

    def test_chat_message_basic(self):
        """Test basic ChatMessage."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.images is None

    def test_chat_request_minimal(self):
        """Test minimal ChatRequest."""
        req = ChatRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert req.model == "test-model"
        assert len(req.messages) == 1
        assert req.stream is True  # default

    def test_chat_request_with_options(self):
        """Test ChatRequest with options."""
        req = ChatRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hi")],
            stream=False,
            options={"temperature": 0.8},
        )
        assert req.stream is False
        assert req.options["temperature"] == 0.8

    def test_generate_request_raw_mode(self):
        """Test GenerateRequest with raw mode."""
        req = GenerateRequest(
            model="test-model",
            prompt="<|user|>\nHello\n<|assistant|>",
            raw=True,
        )
        assert req.raw is True
        assert req.stream is True  # default

    def test_load_request_name_alias(self):
        """Test LoadRequest accepts both 'model' and 'name'."""
        # Using 'name' (Ollama style)
        req1 = LoadRequest(name="test-model")
        assert req1.model == "test-model"

        # Using 'model' directly
        req2 = LoadRequest(model="another-model")
        assert req2.model == "another-model"

    def test_unload_request(self):
        """Test UnloadRequest."""
        req = UnloadRequest(name="test-model")
        assert req.model == "test-model"


class TestOpenAISchemas:
    """Tests for OpenAI API schemas."""

    def test_openai_chat_message(self):
        """Test OpenAI chat message."""
        msg = OpenAIChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_openai_chat_request(self):
        """Test OpenAI chat request."""
        req = OpenAIChatRequest(
            model="gpt-3.5-turbo",
            messages=[OpenAIChatMessage(role="user", content="Hi")],
            temperature=0.7,
        )
        assert req.model == "gpt-3.5-turbo"
        assert req.temperature == 0.7
        assert req.stream is False  # default


class TestModelfileSchemas:
    """Tests for Modelfile CRUD schemas."""

    def test_modelfile_response(self):
        """Test ModelfileResponse."""
        resp = ModelfileResponse(
            model="test-model",
            path="/models/test-model/Modelfile",
            properties={"TEMPERATURE": "0.7", "NUM_CTX": "4096"},
        )
        assert resp.model == "test-model"
        assert resp.properties["TEMPERATURE"] == "0.7"

    def test_modelfile_patch_request(self):
        """Test ModelfilePatchRequest."""
        req = ModelfilePatchRequest(
            properties={"TEMPERATURE": 0.9, "TOP_K": 50}
        )
        assert req.properties["TEMPERATURE"] == 0.9
        assert req.properties["TOP_K"] == 50

    def test_validate_property_name_valid(self):
        """Test valid property names."""
        assert validate_property_name("TEMPERATURE") is True
        assert validate_property_name("temperature") is True
        assert validate_property_name("NUM_CTX") is True

    def test_validate_property_name_invalid(self):
        """Test invalid property names."""
        assert validate_property_name("INVALID_PROP") is False
        assert validate_property_name("foo") is False

    def test_validate_property_value_temperature(self):
        """Test temperature validation."""
        is_valid, error = validate_property_value("TEMPERATURE", 0.7)
        assert is_valid is True
        assert error is None

        is_valid, error = validate_property_value("TEMPERATURE", 3.0)
        assert is_valid is False
        assert "between 0.0 and 2.0" in error

    def test_validate_property_value_num_ctx(self):
        """Test NUM_CTX validation."""
        is_valid, error = validate_property_value("NUM_CTX", 4096)
        assert is_valid is True

        is_valid, error = validate_property_value("NUM_CTX", 64)
        assert is_valid is False
        assert ">= 128" in error

    def test_validate_property_value_mirostat(self):
        """Test MIROSTAT validation."""
        for val in [0, 1, 2]:
            is_valid, _ = validate_property_value("MIROSTAT", val)
            assert is_valid is True

        is_valid, error = validate_property_value("MIROSTAT", 3)
        assert is_valid is False
