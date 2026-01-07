"""Tests for config module."""

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestRKLlamaConfigInitialization:
    """Tests for RKLlamaConfig initialization."""

    def test_config_singleton_exists(self):
        """Test that config singleton is available."""
        from rkllama.config.config import config

        assert config is not None

    def test_config_has_app_root(self):
        """Test that config has app_root set."""
        from rkllama.config.config import config

        assert config.app_root is not None
        assert isinstance(config.app_root, Path)

    def test_config_has_config_dir(self):
        """Test that config has config_dir set."""
        from rkllama.config.config import config

        assert config.config_dir is not None
        assert config.config_dir == config.app_root / "config"

    def test_config_loads_defaults(self):
        """Test that config loads default values from schema."""
        from rkllama.config.config import config

        # These should have default values from pydantic models
        assert config._settings.server is not None
        assert config._settings.paths is not None
        assert config._settings.model is not None
        assert config._settings.platform is not None


class TestRKLlamaConfigGet:
    """Tests for RKLlamaConfig.get() method."""

    def test_get_existing_value(self):
        """Test getting an existing configuration value."""
        from rkllama.config import get

        port = get("server", "port")
        assert port is not None

    def test_get_with_default(self):
        """Test getting value with default fallback."""
        from rkllama.config import get

        value = get("nonexistent", "key", default="fallback")
        assert value == "fallback"

    def test_get_missing_section(self):
        """Test getting value from missing section returns default."""
        from rkllama.config import get

        value = get("missing_section", "key", default=None)
        assert value is None

    def test_get_missing_key(self):
        """Test getting missing key returns default."""
        from rkllama.config import get

        value = get("server", "missing_key", default="default_value")
        assert value == "default_value"

    def test_get_with_type_conversion_int(self):
        """Test getting value with int type conversion."""
        from rkllama.config import get

        # Port is stored as int by default
        value = get("server", "port", as_type=int)
        assert isinstance(value, int)

    def test_get_with_type_conversion_bool(self):
        """Test getting value with bool type conversion."""
        from rkllama.config import get

        # Debug is stored as bool by default
        value = get("server", "debug", as_type=bool)
        assert isinstance(value, bool)


class TestRKLlamaConfigSet:
    """Tests for RKLlamaConfig.set() method."""

    def test_set_server_port(self):
        """Test setting server port value."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        cfg.set("server", "port", 9000)
        assert cfg.get("server", "port") == 9000

    def test_set_server_debug(self):
        """Test setting debug boolean value."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        cfg.set("server", "debug", True)
        assert cfg.get("server", "debug") is True

    def test_set_string_conversion(self):
        """Test setting value from string."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        cfg.set("server", "port", "9001")
        assert cfg.get("server", "port") == 9001

    def test_set_model_temperature(self):
        """Test setting model temperature."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        cfg.set("model", "default_temperature", 0.7)
        assert cfg.get("model", "default_temperature") == 0.7


class TestRKLlamaConfigTypeInference:
    """Tests for type inference in config."""

    def test_parse_boolean_true_values(self):
        """Test boolean true value parsing."""
        from rkllama.config.config import config

        for val in ["true", "yes", "1", "on"]:
            result = config._parse_value(val)
            assert result is True, f"Failed for '{val}'"

    def test_parse_boolean_false_values(self):
        """Test boolean false value parsing."""
        from rkllama.config.config import config

        for val in ["false", "no", "0", "off"]:
            result = config._parse_value(val)
            assert result is False, f"Failed for '{val}'"

    def test_parse_integer(self):
        """Test integer parsing."""
        from rkllama.config.config import config

        assert config._parse_value("123") == 123
        assert config._parse_value("-456") == -456

    def test_parse_float(self):
        """Test float parsing."""
        from rkllama.config.config import config

        assert config._parse_value("3.14") == 3.14
        assert config._parse_value("-2.5") == -2.5

    def test_parse_list(self):
        """Test list parsing from comma-separated values."""
        from rkllama.config.config import config

        result = config._parse_value("a,b,c")
        assert result == ["a", "b", "c"]

    def test_parse_string_fallback(self):
        """Test string fallback for non-parseable values."""
        from rkllama.config.config import config

        assert config._parse_value("hello") == "hello"

    def test_non_string_passthrough(self):
        """Test non-string values pass through unchanged."""
        from rkllama.config.config import config

        assert config._parse_value(42) == 42
        assert config._parse_value(True) is True
        assert config._parse_value([1, 2]) == [1, 2]


class TestRKLlamaConfigPaths:
    """Tests for path resolution."""

    def test_get_path(self):
        """Test getting a path configuration."""
        from rkllama.config import get_path

        models_path = get_path("models")
        assert models_path is not None

    def test_get_path_missing(self):
        """Test getting a missing path returns None."""
        from rkllama.config import get_path

        result = get_path("nonexistent")
        assert result is None

    def test_path_caching(self):
        """Test that path resolution is cached."""
        from rkllama.config.config import config

        # Clear cache first
        config._path_cache.clear()

        path1 = config.get_path("models")
        path2 = config.get_path("models")

        assert path1 == path2
        assert "models" in config._path_cache


class TestRKLlamaConfigEnvironmentVariables:
    """Tests for environment variable loading."""

    def test_load_env_var_server_port(self):
        """Test loading server port from environment variable."""
        from rkllama.config.config import RKLlamaConfig

        with patch.dict(os.environ, {"RKLLAMA_SERVER_PORT": "9999"}):
            cfg = RKLlamaConfig()
            assert cfg.get("server", "port") == 9999

    def test_load_env_var_server_debug(self):
        """Test loading debug from environment variable."""
        from rkllama.config.config import RKLlamaConfig

        with patch.dict(os.environ, {"RKLLAMA_SERVER_DEBUG": "true"}):
            cfg = RKLlamaConfig()
            assert cfg.get("server", "debug") is True


class TestRKLlamaConfigINIFiles:
    """Tests for INI file loading."""

    def test_load_config_file(self, tmp_path):
        """Test loading configuration from INI file."""
        from rkllama.config.config import RKLlamaConfig

        # Create a test INI file with known sections
        ini_file = tmp_path / "test.ini"
        ini_file.write_text("""
[server]
port = 9999
debug = true
""")

        cfg = RKLlamaConfig()
        cfg._load_ini_file(ini_file)

        assert cfg.get("server", "port") == 9999
        assert cfg.get("server", "debug") is True

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a nonexistent file does nothing."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        nonexistent = tmp_path / "nonexistent.ini"

        # Should not raise
        cfg._load_ini_file(nonexistent)


class TestRKLlamaConfigCommandLineArgs:
    """Tests for command-line argument loading."""

    def test_load_port_arg(self):
        """Test loading port from command-line args."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        args = argparse.Namespace(port=9000, debug=None, processor=None, models=None, config=None)
        cfg.load_args(args)

        assert cfg.get("server", "port") == 9000

    def test_load_debug_arg(self):
        """Test loading debug from command-line args."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        args = argparse.Namespace(port=None, debug=True, processor=None, models=None, config=None)
        cfg.load_args(args)

        assert cfg.get("server", "debug") is True

    def test_load_processor_arg(self):
        """Test loading processor from command-line args."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        args = argparse.Namespace(port=None, debug=None, processor="rk3576", models=None, config=None)
        cfg.load_args(args)

        assert cfg.get("platform", "processor") == "rk3576"

    def test_load_models_arg(self):
        """Test loading models path from command-line args."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        args = argparse.Namespace(port=None, debug=None, processor=None, models="/custom/models", config=None)
        cfg.load_args(args)

        assert cfg.get("paths", "models") == "/custom/models"

    def test_load_custom_config_file(self, tmp_path):
        """Test loading custom config file from command-line."""
        from rkllama.config.config import RKLlamaConfig

        # Create custom config
        custom_ini = tmp_path / "custom.ini"
        custom_ini.write_text("""
[server]
port = 7777
""")

        cfg = RKLlamaConfig()
        args = argparse.Namespace(port=None, debug=None, processor=None, models=None, config=str(custom_ini))
        cfg.load_args(args)

        assert cfg.get("server", "port") == 7777


class TestRKLlamaConfigDebugMode:
    """Tests for debug mode functionality."""

    def test_is_debug_mode_false(self):
        """Test is_debug_mode returns False by default."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        # Set debug to False explicitly
        cfg.set("server", "debug", False)
        assert cfg.is_debug_mode() is False

    def test_is_debug_mode_true(self):
        """Test is_debug_mode returns True when enabled."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        cfg.set("server", "debug", True)
        assert cfg.is_debug_mode() is True

    def test_module_level_is_debug_mode(self):
        """Test module-level is_debug_mode function."""
        from rkllama.config import is_debug_mode, set

        # Should return the current debug state
        set("server", "debug", False)
        assert is_debug_mode() is False

        set("server", "debug", True)
        assert is_debug_mode() is True


class TestRKLlamaConfigReload:
    """Tests for config reload functionality."""

    def test_reload_config(self):
        """Test reloading configuration."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()

        # Change a value
        cfg.set("server", "port", 12345)

        # Reload should restore from sources
        cfg.reload_config()

        # Note: The reloaded value depends on configuration files
        # For testing, we just ensure it doesn't crash
        assert cfg.get("server", "port") is not None


class TestRKLlamaConfigValidation:
    """Tests for config validation."""

    def test_validate_creates_directories(self, tmp_path):
        """Test that validate creates required directories."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()

        # Set paths to temp directory
        test_models = tmp_path / "test_models"

        cfg.set("paths", "models", str(test_models))

        # Clear path cache so get_path returns new path
        cfg._path_cache.clear()

        # Validate should create directories
        cfg.validate()

        # Check if directory was created
        assert test_models.exists()


class TestRKLlamaConfigConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_module_get(self):
        """Test module-level get function."""
        from rkllama.config import get

        port = get("server", "port", 8080)
        assert port is not None

    def test_module_set(self):
        """Test module-level set function."""
        from rkllama.config import get, set

        original = get("server", "port")
        set("server", "port", 9999)
        assert get("server", "port") == 9999
        # Restore
        set("server", "port", original)

    def test_module_get_path(self):
        """Test module-level get_path function."""
        from rkllama.config import get_path

        models = get_path("models")
        assert models is not None

    def test_module_display(self, capsys):
        """Test module-level display function doesn't crash."""
        from rkllama.config import display

        # Should not raise
        display()

    def test_module_validate(self):
        """Test module-level validate function."""
        from rkllama.config import validate

        # Should return True or False, not crash
        result = validate()
        assert isinstance(result, bool)

    def test_module_reload_config(self):
        """Test module-level reload_config function."""
        from rkllama.config import reload_config

        # Should not raise
        reload_config()


class TestRKLlamaConfigShellConfig:
    """Tests for shell config generation."""

    def test_generate_shell_config(self):
        """Test that shell config file is generated."""
        from rkllama.config.config import config

        # The config.env file should exist in config_dir
        config_env = config.config_dir / "config.env"

        # Regenerate
        config._generate_shell_config()

        assert config_env.exists()

    def test_shell_config_content(self):
        """Test shell config file has expected content."""
        from rkllama.config.config import config

        config._generate_shell_config()
        config_env = config.config_dir / "config.env"

        content = config_env.read_text()

        assert "#!/bin/sh" in content
        assert "RKLLAMA_ROOT=" in content
        assert "RKLLAMA_SERVER_PORT=" in content


class TestRKLlamaConfigEdgeCases:
    """Tests for edge cases and error handling."""

    def test_set_invalid_port_logs_warning(self):
        """Test setting invalid port value logs warning but doesn't crash."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        # Port has min/max constraints from pydantic
        # Setting invalid value should log warning
        cfg.set("server", "port", -1)
        # Just ensure it doesn't crash

    def test_set_unknown_section(self):
        """Test setting value on unknown section logs warning."""
        from rkllama.config.config import RKLlamaConfig

        cfg = RKLlamaConfig()
        # Unknown section should log warning
        cfg.set("unknown_section", "key", "value")
        # Just ensure it doesn't crash

    def test_none_value_handling(self):
        """Test handling of None values."""
        from rkllama.config.config import config

        result = config._parse_value(None)
        assert result is None
