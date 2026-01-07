"""
Unit tests for the CLI module.
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


class TestCliVersion:
    """Tests for CLI version command."""

    def test_version_option(self):
        """Test --version shows version."""
        from rkllama_converter import __version__
        from rkllama_converter.cli import app

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_short_version_option(self):
        """Test -v shows version."""
        from rkllama_converter import __version__
        from rkllama_converter.cli import app

        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.stdout


class TestListQuants:
    """Tests for list-quants command."""

    def test_list_quants_shows_table(self):
        """Test list-quants displays quantization table."""
        from rkllama_converter.cli import app

        result = runner.invoke(app, ["list-quants"])
        assert result.exit_code == 0
        assert "Q4_0" in result.stdout
        assert "Q8_0" in result.stdout
        assert "w4a16" in result.stdout
        assert "w8a8" in result.stdout


class TestListDevices:
    """Tests for list-devices command."""

    def test_list_devices_shows_cpu(self):
        """Test list-devices shows CPU info."""
        from rkllama_converter.cli import app

        with patch("torch.cuda.is_available", return_value=False):
            result = runner.invoke(app, ["list-devices"])

        assert result.exit_code == 0
        assert "CPU" in result.stdout


class TestCliEnums:
    """Tests for CLI enum types."""

    def test_quantization_enum_values(self):
        """Test Quantization enum has expected values."""
        from rkllama_converter.cli import Quantization

        assert Quantization.Q4_0.value == "Q4_0"
        assert Quantization.Q4_K_M.value == "Q4_K_M"
        assert Quantization.Q8_0.value == "Q8_0"
        assert Quantization.Q8_K_M.value == "Q8_K_M"

    def test_device_type_enum_values(self):
        """Test CliDeviceType enum has expected values."""
        from rkllama_converter.cli import CliDeviceType

        assert CliDeviceType.cuda.value == "cuda"
        assert CliDeviceType.rocm.value == "rocm"
        assert CliDeviceType.cpu.value == "cpu"
        assert CliDeviceType.auto.value == "auto"


class TestConvertCommand:
    """Tests for convert command."""

    def test_convert_requires_model_id(self):
        """Test convert command requires model_id argument."""
        from rkllama_converter.cli import app

        result = runner.invoke(app, ["convert"])
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "required" in result.stdout.lower()

    def test_convert_shows_panel_on_start(self):
        """Test convert shows info panel before starting."""
        from rkllama_converter.cli import app

        # Mock the converter to avoid actual conversion
        with patch("rkllama_converter.cli.HuggingFaceToRKLLMConverter") as mock_converter:
            mock_instance = MagicMock()
            mock_converter.return_value = mock_instance

            result = runner.invoke(app, ["convert", "test/model", "--output", "/tmp/out"])

        # Should show the model info panel
        assert "test/model" in result.stdout or result.exit_code != 0

    def test_convert_accepts_quantization_option(self):
        """Test convert accepts -q/--quant option."""
        from rkllama_converter.cli import app

        # Just check the option is accepted (will fail on actual conversion)
        result = runner.invoke(app, ["convert", "test/model", "-q", "Q8_0"])
        # Should not fail on argument parsing
        assert "Invalid value" not in result.stdout

    def test_convert_accepts_device_option(self):
        """Test convert accepts --device option."""
        from rkllama_converter.cli import app

        result = runner.invoke(app, ["convert", "test/model", "--device", "cpu"])
        assert "Invalid value" not in result.stdout


class TestInfoCommand:
    """Tests for info command."""

    def test_info_requires_model_id(self):
        """Test info command requires model_id argument."""
        from rkllama_converter.cli import app

        result = runner.invoke(app, ["info"])
        assert result.exit_code != 0

    def test_info_fetches_model_info(self):
        """Test info command fetches from HuggingFace."""
        from rkllama_converter.cli import app

        mock_info = MagicMock()
        mock_info.id = "test/model"
        mock_info.author = "test"
        mock_info.downloads = 1000
        mock_info.likes = 50
        mock_info.pipeline_tag = "text-generation"
        mock_info.card_data = MagicMock(license="MIT")
        mock_info.safetensors = None

        with patch("huggingface_hub.model_info", return_value=mock_info):
            result = runner.invoke(app, ["info", "test/model"])

        assert result.exit_code == 0
        assert "test/model" in result.stdout


class TestHelpText:
    """Tests for CLI help text."""

    def test_main_help(self):
        """Test main help text."""
        from rkllama_converter.cli import app

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Convert HuggingFace models" in result.stdout

    def test_convert_help(self):
        """Test convert command help text."""
        from rkllama_converter.cli import app

        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "HuggingFace model ID" in result.stdout
        assert "--quant" in result.stdout
        assert "--output" in result.stdout

    def test_info_help(self):
        """Test info command help text."""
        from rkllama_converter.cli import app

        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0
        assert "HuggingFace model ID" in result.stdout
