"""
Unit tests for Modelfile generation module.
"""

from pathlib import Path

import pytest


class TestModelfileConfig:
    """Tests for ModelfileConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        from rkllama_converter.modelfile import ModelfileConfig

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
        )

        assert config.model_file == "test.rkllm"
        assert config.huggingface_path == "org/model"
        assert config.system == "You are a helpful AI assistant."
        assert config.temperature == 0.7
        assert config.num_ctx == 4096
        assert config.max_new_tokens == 1024
        assert config.top_k == 40
        assert config.top_p == 0.9
        assert config.repeat_penalty == 1.1
        assert config.mirostat == 0
        assert config.enable_thinking is False

    def test_custom_values(self):
        """Test custom values are stored correctly."""
        from rkllama_converter.modelfile import ModelfileConfig

        config = ModelfileConfig(
            model_file="custom.rkllm",
            huggingface_path="custom/model",
            system="Custom system prompt",
            temperature=0.9,
            num_ctx=8192,
            enable_thinking=True,
        )

        assert config.system == "Custom system prompt"
        assert config.temperature == 0.9
        assert config.num_ctx == 8192
        assert config.enable_thinking is True


class TestGenerateModelfile:
    """Tests for generate_modelfile function."""

    def test_basic_generation(self):
        """Test basic Modelfile generation."""
        from rkllama_converter.modelfile import ModelfileConfig, generate_modelfile

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
        )

        content = generate_modelfile(config)

        assert 'FROM="test.rkllm"' in content
        assert 'HUGGINGFACE_PATH="org/model"' in content
        assert "TEMPERATURE=0.7" in content
        assert "NUM_CTX=4096" in content

    def test_includes_thinking_when_enabled(self):
        """Test Modelfile includes thinking when enabled."""
        from rkllama_converter.modelfile import ModelfileConfig, generate_modelfile

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
            enable_thinking=True,
        )

        content = generate_modelfile(config)
        assert "ENABLE_THINKING=true" in content

    def test_excludes_thinking_when_disabled(self):
        """Test Modelfile excludes thinking when disabled."""
        from rkllama_converter.modelfile import ModelfileConfig, generate_modelfile

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
            enable_thinking=False,
        )

        content = generate_modelfile(config)
        assert "ENABLE_THINKING" not in content

    def test_includes_mirostat_when_enabled(self):
        """Test Modelfile includes mirostat settings when enabled."""
        from rkllama_converter.modelfile import ModelfileConfig, generate_modelfile

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
            mirostat=2,
            mirostat_tau=5.0,
            mirostat_eta=0.2,
        )

        content = generate_modelfile(config)
        assert "MIROSTAT=2" in content
        assert "MIROSTAT_TAU=5.0" in content
        assert "MIROSTAT_ETA=0.2" in content

    def test_excludes_zero_penalties(self):
        """Test frequency/presence penalties excluded when zero."""
        from rkllama_converter.modelfile import ModelfileConfig, generate_modelfile

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        content = generate_modelfile(config)
        assert "FREQUENCY_PENALTY" not in content
        assert "PRESENCE_PENALTY" not in content

    def test_includes_nonzero_penalties(self):
        """Test frequency/presence penalties included when non-zero."""
        from rkllama_converter.modelfile import ModelfileConfig, generate_modelfile

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
            frequency_penalty=0.5,
            presence_penalty=-0.3,
        )

        content = generate_modelfile(config)
        assert "FREQUENCY_PENALTY=0.5" in content
        assert "PRESENCE_PENALTY=-0.3" in content

    def test_escapes_special_characters(self):
        """Test special characters are escaped in strings."""
        from rkllama_converter.modelfile import ModelfileConfig, generate_modelfile

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
            system='System with "quotes" and\nnewlines',
        )

        content = generate_modelfile(config)
        # Quotes should be escaped
        assert '\\"quotes\\"' in content
        # Newlines should be escaped
        assert "\\n" in content


class TestParseModelfile:
    """Tests for parse_modelfile function."""

    def test_parse_basic_modelfile(self):
        """Test parsing a basic Modelfile."""
        from rkllama_converter.modelfile import parse_modelfile

        content = '''FROM="model.rkllm"
HUGGINGFACE_PATH="org/model"
TEMPERATURE=0.7
NUM_CTX=4096'''

        result = parse_modelfile(content)

        assert result["FROM"] == "model.rkllm"
        assert result["HUGGINGFACE_PATH"] == "org/model"
        assert result["TEMPERATURE"] == 0.7
        assert result["NUM_CTX"] == 4096

    def test_parse_skips_comments(self):
        """Test that comments are skipped."""
        from rkllama_converter.modelfile import parse_modelfile

        content = '''# This is a comment
FROM="model.rkllm"
# Another comment
TEMPERATURE=0.8'''

        result = parse_modelfile(content)

        assert result["FROM"] == "model.rkllm"
        assert result["TEMPERATURE"] == 0.8
        assert "#" not in str(result.keys())

    def test_parse_boolean_values(self):
        """Test parsing boolean values."""
        from rkllama_converter.modelfile import parse_modelfile

        content = '''ENABLE_THINKING=true
SOME_FLAG=false'''

        result = parse_modelfile(content)

        assert result["ENABLE_THINKING"] is True
        assert result["SOME_FLAG"] is False

    def test_parse_unescapes_strings(self):
        """Test that escaped strings are unescaped."""
        from rkllama_converter.modelfile import parse_modelfile

        content = r'SYSTEM="Line 1\nLine 2"'

        result = parse_modelfile(content)
        assert result["SYSTEM"] == "Line 1\nLine 2"


class TestSaveModelfile:
    """Tests for save_modelfile function."""

    def test_saves_to_correct_path(self, temp_output_dir):
        """Test Modelfile is saved to correct path."""
        from rkllama_converter.modelfile import ModelfileConfig, save_modelfile

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
        )

        path = save_modelfile(config, temp_output_dir)

        assert path == temp_output_dir / "Modelfile"
        assert path.exists()

    def test_saved_content_is_valid(self, temp_output_dir):
        """Test saved content can be parsed back."""
        from rkllama_converter.modelfile import (
            ModelfileConfig,
            parse_modelfile,
            save_modelfile,
        )

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
            temperature=0.9,
        )

        path = save_modelfile(config, temp_output_dir)
        content = path.read_text()
        parsed = parse_modelfile(content)

        assert parsed["FROM"] == "test.rkllm"
        assert parsed["HUGGINGFACE_PATH"] == "org/model"
        assert parsed["TEMPERATURE"] == 0.9

    def test_creates_directory_if_missing(self, tmp_path):
        """Test directory is created if it doesn't exist."""
        from rkllama_converter.modelfile import ModelfileConfig, save_modelfile

        config = ModelfileConfig(
            model_file="test.rkllm",
            huggingface_path="org/model",
        )

        new_dir = tmp_path / "new" / "nested" / "dir"
        path = save_modelfile(config, new_dir)

        assert path.exists()
        assert new_dir.exists()
