"""Tests for config schema module using pydantic-settings."""

import pytest
from pydantic import ValidationError

from rkllama.config.config_schema import (
    ModelSettings,
    PathSettings,
    PlatformSettings,
    RKLlamaSettings,
    ServerSettings,
)


class TestServerSettings:
    """Tests for ServerSettings."""

    def test_default_values(self):
        """Test default server settings."""
        settings = ServerSettings()
        assert settings.port == 8080
        assert settings.host == "0.0.0.0"
        assert settings.debug is False

    def test_port_validation_min(self):
        """Test port minimum value validation."""
        with pytest.raises(ValidationError):
            ServerSettings(port=0)

    def test_port_validation_max(self):
        """Test port maximum value validation."""
        with pytest.raises(ValidationError):
            ServerSettings(port=65536)

    def test_port_valid_range(self):
        """Test valid port values."""
        settings = ServerSettings(port=1)
        assert settings.port == 1

        settings = ServerSettings(port=65535)
        assert settings.port == 65535

    def test_debug_true(self):
        """Test debug mode enabled."""
        settings = ServerSettings(debug=True)
        assert settings.debug is True


class TestPathSettings:
    """Tests for PathSettings."""

    def test_default_values(self):
        """Test default path settings."""
        settings = PathSettings()
        assert settings.models == "models"
        assert settings.logs == "logs"
        assert settings.data == "data"
        assert settings.src == "src"
        assert settings.lib == "lib"
        assert settings.temp == "temp"

    def test_custom_paths(self):
        """Test custom path values."""
        settings = PathSettings(
            models="/custom/models",
            logs="/var/log/rkllama",
        )
        assert settings.models == "/custom/models"
        assert settings.logs == "/var/log/rkllama"


class TestModelSettings:
    """Tests for ModelSettings."""

    def test_default_values(self):
        """Test default model settings."""
        settings = ModelSettings()
        assert settings.default == ""
        assert settings.default_temperature == 0.5
        assert settings.default_enable_thinking is False
        assert settings.default_num_ctx == 4096
        assert settings.default_max_new_tokens == 1024
        assert settings.default_top_k == 7
        assert settings.default_top_p == 0.5
        assert settings.default_repeat_penalty == 1.1
        assert settings.default_frequency_penalty == 0.0
        assert settings.default_presence_penalty == 0.0
        assert settings.default_mirostat == 0
        assert settings.default_mirostat_tau == 3.0
        assert settings.default_mirostat_eta == 0.1
        assert settings.max_minutes_loaded_in_memory == 30
        assert settings.max_number_models_loaded_in_memory == 10

    def test_temperature_validation_min(self):
        """Test temperature minimum value."""
        settings = ModelSettings(default_temperature=0.0)
        assert settings.default_temperature == 0.0

    def test_temperature_validation_max(self):
        """Test temperature maximum value."""
        settings = ModelSettings(default_temperature=2.0)
        assert settings.default_temperature == 2.0

    def test_temperature_validation_invalid(self):
        """Test temperature out of range."""
        with pytest.raises(ValidationError):
            ModelSettings(default_temperature=-0.1)

        with pytest.raises(ValidationError):
            ModelSettings(default_temperature=2.1)

    def test_num_ctx_validation(self):
        """Test context window size validation."""
        settings = ModelSettings(default_num_ctx=512)
        assert settings.default_num_ctx == 512

        settings = ModelSettings(default_num_ctx=131072)
        assert settings.default_num_ctx == 131072

    def test_num_ctx_validation_invalid(self):
        """Test context window size out of range."""
        with pytest.raises(ValidationError):
            ModelSettings(default_num_ctx=511)

        with pytest.raises(ValidationError):
            ModelSettings(default_num_ctx=131073)

    def test_mirostat_validation(self):
        """Test mirostat mode validation."""
        for mode in [0, 1, 2]:
            settings = ModelSettings(default_mirostat=mode)
            assert settings.default_mirostat == mode

    def test_mirostat_validation_invalid(self):
        """Test mirostat mode out of range."""
        with pytest.raises(ValidationError):
            ModelSettings(default_mirostat=-1)

        with pytest.raises(ValidationError):
            ModelSettings(default_mirostat=3)

    def test_top_p_validation(self):
        """Test top_p range validation."""
        settings = ModelSettings(default_top_p=0.0)
        assert settings.default_top_p == 0.0

        settings = ModelSettings(default_top_p=1.0)
        assert settings.default_top_p == 1.0

    def test_top_p_validation_invalid(self):
        """Test top_p out of range."""
        with pytest.raises(ValidationError):
            ModelSettings(default_top_p=-0.1)

        with pytest.raises(ValidationError):
            ModelSettings(default_top_p=1.1)


class TestPlatformSettings:
    """Tests for PlatformSettings."""

    def test_default_processor(self):
        """Test default processor value."""
        settings = PlatformSettings()
        assert settings.processor == "rk3588"

    def test_valid_processors(self):
        """Test valid processor values."""
        settings = PlatformSettings(processor="rk3588")
        assert settings.processor == "rk3588"

        settings = PlatformSettings(processor="rk3576")
        assert settings.processor == "rk3576"

    def test_invalid_processor(self):
        """Test invalid processor value."""
        with pytest.raises(ValidationError):
            PlatformSettings(processor="invalid_processor")


class TestRKLlamaSettings:
    """Tests for RKLlamaSettings."""

    def test_default_values(self):
        """Test default combined settings."""
        settings = RKLlamaSettings()

        # Server defaults
        assert settings.server.port == 8080
        assert settings.server.debug is False

        # Paths defaults
        assert settings.paths.models == "models"

        # Model defaults
        assert settings.model.default_temperature == 0.5

        # Platform defaults
        assert settings.platform.processor == "rk3588"

    def test_get_section(self):
        """Test get_section method."""
        settings = RKLlamaSettings()

        server = settings.get_section("server")
        assert server is not None
        assert server.port == 8080

        paths = settings.get_section("paths")
        assert paths is not None
        assert paths.models == "models"

        nonexistent = settings.get_section("nonexistent")
        assert nonexistent is None

    def test_nested_settings(self):
        """Test creating settings with nested values."""
        settings = RKLlamaSettings(
            server=ServerSettings(port=9000, debug=True),
            platform=PlatformSettings(processor="rk3576"),
        )

        assert settings.server.port == 9000
        assert settings.server.debug is True
        assert settings.platform.processor == "rk3576"


class TestPathResolution:
    """Tests for path resolution."""

    def test_absolute_path(self):
        """Test absolute path resolution."""
        from pathlib import Path

        settings = RKLlamaSettings()
        resolved = settings.resolve_path("/absolute/path", Path("/root"))
        assert resolved == "/absolute/path"

    def test_relative_path(self):
        """Test relative path resolution."""
        from pathlib import Path

        settings = RKLlamaSettings()
        resolved = settings.resolve_path("relative/path", Path("/root"))
        assert resolved == "/root/relative/path"

    def test_empty_path(self):
        """Test empty path returns None."""
        from pathlib import Path

        settings = RKLlamaSettings()
        resolved = settings.resolve_path("", Path("/root"))
        assert resolved is None

    def test_home_expansion(self):
        """Test home directory expansion."""
        import os
        from pathlib import Path

        settings = RKLlamaSettings()
        resolved = settings.resolve_path("~/test", Path("/root"))
        assert resolved.startswith(os.path.expanduser("~"))

    def test_env_var_expansion(self, monkeypatch):
        """Test environment variable expansion."""
        from pathlib import Path

        monkeypatch.setenv("TEST_DIR", "/test/dir")
        settings = RKLlamaSettings()
        resolved = settings.resolve_path("$TEST_DIR/subdir", Path("/root"))
        assert resolved == "/test/dir/subdir"
