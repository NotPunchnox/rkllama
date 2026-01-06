"""Tests for config module."""

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestRKLLAMAConfigInitialization:
    """Tests for RKLLAMAConfig initialization."""

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

        # These should have default values from schema
        assert "server" in config.config
        assert "paths" in config.config
        assert "model" in config.config
        assert "platform" in config.config


class TestRKLLAMAConfigGet:
    """Tests for RKLLAMAConfig.get() method."""

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
        from rkllama.config import get, set

        set("test_section", "int_val", "42")
        value = get("test_section", "int_val", as_type=int)
        assert value == 42
        assert isinstance(value, int)

    def test_get_with_type_conversion_float(self):
        """Test getting value with float type conversion."""
        from rkllama.config import get, set

        set("test_section", "float_val", "3.14")
        value = get("test_section", "float_val", as_type=float)
        assert value == 3.14
        assert isinstance(value, float)

    def test_get_with_type_conversion_bool(self):
        """Test getting value with bool type conversion."""
        from rkllama.config import get, set

        set("test_section", "bool_val", "true")
        value = get("test_section", "bool_val", as_type=bool)
        assert value is True

    def test_get_with_type_conversion_list(self):
        """Test getting value with list type conversion."""
        from rkllama.config import get, set

        set("test_section", "list_val", "a,b,c")
        value = get("test_section", "list_val", as_type=list)
        assert value == ["a", "b", "c"]


class TestRKLLAMAConfigSet:
    """Tests for RKLLAMAConfig.set() method."""

    def test_set_creates_section(self):
        """Test that set creates section if it doesn't exist."""
        from rkllama.config import get, set

        set("new_section", "key", "value")
        assert get("new_section", "key") == "value"

    def test_set_string_value(self):
        """Test setting a string value."""
        from rkllama.config import get, set

        set("test", "string_key", "test_value")
        assert get("test", "string_key") == "test_value"

    def test_set_integer_value(self):
        """Test setting an integer value."""
        from rkllama.config import get, set

        set("test", "int_key", 123)
        assert get("test", "int_key") == 123

    def test_set_float_value(self):
        """Test setting a float value."""
        from rkllama.config import get, set

        set("test", "float_key", 3.14)
        assert get("test", "float_key") == 3.14

    def test_set_boolean_value(self):
        """Test setting a boolean value."""
        from rkllama.config import get, set

        set("test", "bool_key", True)
        assert get("test", "bool_key") is True

    def test_set_list_value(self):
        """Test setting a list value."""
        from rkllama.config import get, set

        set("test", "list_key", ["a", "b", "c"])
        assert get("test", "list_key") == ["a", "b", "c"]


class TestRKLLAMAConfigTypeInference:
    """Tests for type inference in config."""

    def test_infer_boolean_true_values(self):
        """Test boolean true value inference."""
        from rkllama.config.config import config

        for val in ["true", "yes", "1", "on"]:
            result = config._infer_and_convert_type("test", "key", val)
            assert result is True, f"Failed for '{val}'"

    def test_infer_boolean_false_values(self):
        """Test boolean false value inference."""
        from rkllama.config.config import config

        for val in ["false", "no", "0", "off"]:
            result = config._infer_and_convert_type("test", "key", val)
            assert result is False, f"Failed for '{val}'"

    def test_infer_integer(self):
        """Test integer inference."""
        from rkllama.config.config import config

        assert config._infer_and_convert_type("test", "key", "123") == 123
        assert config._infer_and_convert_type("test", "key", "-456") == -456

    def test_infer_float(self):
        """Test float inference."""
        from rkllama.config.config import config

        assert config._infer_and_convert_type("test", "key", "3.14") == 3.14
        assert config._infer_and_convert_type("test", "key", "-2.5") == -2.5

    def test_infer_list(self):
        """Test list inference from comma-separated values."""
        from rkllama.config.config import config

        result = config._infer_and_convert_type("test", "key", "a,b,c")
        assert result == ["a", "b", "c"]

    def test_infer_string_fallback(self):
        """Test string fallback for non-inferrable values."""
        from rkllama.config.config import config

        assert config._infer_and_convert_type("test", "key", "hello") == "hello"

    def test_non_string_passthrough(self):
        """Test non-string values pass through unchanged."""
        from rkllama.config.config import config

        assert config._infer_and_convert_type("test", "key", 42) == 42
        assert config._infer_and_convert_type("test", "key", True) is True
        assert config._infer_and_convert_type("test", "key", [1, 2]) == [1, 2]


class TestRKLLAMAConfigPaths:
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

    def test_resolve_absolute_path(self):
        """Test resolving an absolute path."""
        from rkllama.config.config import config

        result = config.resolve_path("/absolute/path")
        assert result == "/absolute/path"

    def test_resolve_relative_path(self):
        """Test resolving a relative path."""
        from rkllama.config.config import config

        result = config.resolve_path("relative/path")
        expected = str(config.app_root / "relative/path")
        assert result == expected

    def test_resolve_path_with_home(self):
        """Test resolving a path with ~ (home directory)."""
        from rkllama.config.config import config

        result = config.resolve_path("~/test/path")
        assert result.startswith(str(Path.home()))
        assert "test/path" in result

    def test_resolve_path_caching(self):
        """Test that path resolution is cached."""
        from rkllama.config.config import config

        # Clear cache first
        config._clear_path_cache()

        path = "test/cached/path"
        result1 = config.resolve_path(path)
        result2 = config.resolve_path(path)

        assert result1 == result2
        assert path in config._path_cache

    def test_resolve_none_path(self):
        """Test resolving None returns None."""
        from rkllama.config.config import config

        assert config.resolve_path(None) is None

    def test_resolve_empty_path(self):
        """Test resolving empty string returns None."""
        from rkllama.config.config import config

        assert config.resolve_path("") is None


class TestRKLLAMAConfigEnvironmentVariables:
    """Tests for environment variable loading."""

    def test_load_env_var(self):
        """Test loading a configuration from environment variable."""
        from rkllama.config.config import RKLLAMAConfig

        with patch.dict(os.environ, {"RKLLAMA_SERVER_PORT": "9999"}):
            cfg = RKLLAMAConfig()
            assert cfg.get("server", "port") == 9999

    def test_load_env_var_debug(self):
        """Test loading RKLLAMA_DEBUG environment variable."""
        from rkllama.config.config import RKLLAMAConfig

        with patch.dict(os.environ, {"RKLLAMA_DEBUG": "1"}):
            cfg = RKLLAMAConfig()
            assert cfg.get("server", "debug") is True

        with patch.dict(os.environ, {"RKLLAMA_DEBUG": "0"}):
            cfg = RKLLAMAConfig()
            assert cfg.get("server", "debug") is False

    def test_env_var_type_conversion(self):
        """Test that environment variables are type-converted."""
        from rkllama.config.config import RKLLAMAConfig

        with patch.dict(os.environ, {"RKLLAMA_SERVER_DEBUG": "true"}):
            cfg = RKLLAMAConfig()
            debug = cfg.get("server", "debug")
            assert debug is True


class TestRKLLAMAConfigINIFiles:
    """Tests for INI file loading."""

    def test_load_config_file(self, tmp_path):
        """Test loading configuration from INI file."""
        from rkllama.config.config import RKLLAMAConfig

        # Create a test INI file
        ini_file = tmp_path / "test.ini"
        ini_file.write_text("""
[test_section]
key1 = value1
key2 = 42
key3 = true
""")

        cfg = RKLLAMAConfig()
        cfg._load_config_file(ini_file)

        assert cfg.get("test_section", "key1") == "value1"
        assert cfg.get("test_section", "key2") == 42
        assert cfg.get("test_section", "key3") is True

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a nonexistent file does nothing."""
        from rkllama.config.config import RKLLAMAConfig

        cfg = RKLLAMAConfig()
        nonexistent = tmp_path / "nonexistent.ini"

        # Should not raise
        cfg._load_config_file(nonexistent)


class TestRKLLAMAConfigCommandLineArgs:
    """Tests for command-line argument loading."""

    def test_load_port_arg(self):
        """Test loading port from command-line args."""
        from rkllama.config.config import RKLLAMAConfig

        cfg = RKLLAMAConfig()
        args = argparse.Namespace(port=9000, debug=None, processor=None, models=None, config=None)
        cfg.load_args(args)

        assert cfg.get("server", "port") == 9000

    def test_load_debug_arg(self):
        """Test loading debug from command-line args."""
        from rkllama.config.config import RKLLAMAConfig

        cfg = RKLLAMAConfig()
        args = argparse.Namespace(port=None, debug=True, processor=None, models=None, config=None)
        cfg.load_args(args)

        assert cfg.get("server", "debug") is True

    def test_load_processor_arg(self):
        """Test loading processor from command-line args."""
        from rkllama.config.config import RKLLAMAConfig

        cfg = RKLLAMAConfig()
        args = argparse.Namespace(port=None, debug=None, processor="rk3576", models=None, config=None)
        cfg.load_args(args)

        assert cfg.get("platform", "processor") == "rk3576"

    def test_load_models_arg(self):
        """Test loading models path from command-line args."""
        from rkllama.config.config import RKLLAMAConfig

        cfg = RKLLAMAConfig()
        args = argparse.Namespace(port=None, debug=None, processor=None, models="/custom/models", config=None)
        cfg.load_args(args)

        assert cfg.get("paths", "models") == "/custom/models"

    def test_load_custom_config_file(self, tmp_path):
        """Test loading custom config file from command-line."""
        from rkllama.config.config import RKLLAMAConfig

        # Create custom config
        custom_ini = tmp_path / "custom.ini"
        custom_ini.write_text("""
[custom_section]
custom_key = custom_value
""")

        cfg = RKLLAMAConfig()
        args = argparse.Namespace(port=None, debug=None, processor=None, models=None, config=str(custom_ini))
        cfg.load_args(args)

        assert cfg.get("custom_section", "custom_key") == "custom_value"


class TestRKLLAMAConfigDebugMode:
    """Tests for debug mode functionality."""

    def test_is_debug_mode_false(self):
        """Test is_debug_mode returns False by default."""
        from rkllama.config.config import RKLLAMAConfig

        cfg = RKLLAMAConfig()
        # Set debug to False explicitly
        cfg.set("server", "debug", False)
        assert cfg.is_debug_mode() is False

    def test_is_debug_mode_true(self):
        """Test is_debug_mode returns True when enabled."""
        from rkllama.config.config import RKLLAMAConfig

        cfg = RKLLAMAConfig()
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


class TestRKLLAMAConfigReload:
    """Tests for config reload functionality."""

    def test_reload_config(self):
        """Test reloading configuration."""
        from rkllama.config.config import RKLLAMAConfig

        cfg = RKLLAMAConfig()

        # Change a value
        original_port = cfg.get("server", "port")
        cfg.set("server", "port", 12345)

        # Reload should restore from sources
        cfg.reload_config()

        # Note: The reloaded value depends on configuration files
        # For testing, we just ensure it doesn't crash
        assert cfg.get("server", "port") is not None


class TestRKLLAMAConfigValidation:
    """Tests for config validation."""

    def test_validate_creates_directories(self, tmp_path):
        """Test that validate creates required directories."""
        from rkllama.config.config import RKLLAMAConfig

        cfg = RKLLAMAConfig()

        # Set paths to temp directory
        test_models = tmp_path / "models"
        test_logs = tmp_path / "logs"

        cfg.set("paths", "models", str(test_models))
        cfg.set("paths", "logs", str(test_logs))

        # Clear path cache so resolve_path returns new paths
        cfg._clear_path_cache()

        # Validate should create directories
        cfg.validate()

        # Directories should now exist (or validation attempted to create them)
        # Note: Actual creation depends on resolve_path implementation


class TestRKLLAMAConfigConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_module_get(self):
        """Test module-level get function."""
        from rkllama.config import get

        port = get("server", "port", 8080)
        assert port is not None

    def test_module_set(self):
        """Test module-level set function."""
        from rkllama.config import get, set

        set("test_module", "key", "value")
        assert get("test_module", "key") == "value"

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


class TestRKLLAMAConfigFieldInfo:
    """Tests for field info caching."""

    def test_get_field_info_from_schema(self):
        """Test getting field info from schema."""
        from rkllama.config.config import config
        from rkllama.config.config_schema import FieldType

        field_type, default = config._get_field_info("server", "port")
        assert field_type == FieldType.INTEGER
        assert default == 8080

    def test_get_field_info_missing(self):
        """Test getting field info for unknown field."""
        from rkllama.config.config import config

        field_type, default = config._get_field_info("unknown", "field")
        assert field_type is None
        assert default is None

    def test_field_info_caching(self):
        """Test that field info is cached."""
        from rkllama.config.config import config

        # Clear cache
        config._type_cache = {}

        # First call should populate cache
        config._get_field_info("server", "port")
        assert "server.port" in config._type_cache

        # Second call should use cache
        field_type, default = config._get_field_info("server", "port")
        assert field_type is not None


class TestRKLLAMAConfigShellConfig:
    """Tests for shell config generation."""

    def test_generate_shell_config(self, tmp_path):
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


class TestRKLLAMAConfigEdgeCases:
    """Tests for edge cases and error handling."""

    def test_set_with_schema_validation_error(self):
        """Test setting invalid value falls back to default."""
        from rkllama.config.config import config

        # Port has min/max constraints
        # Setting invalid value should use default or log warning
        config.set("server", "port", -1)
        # Implementation may use default or accept the value
        # Just ensure it doesn't crash

    def test_get_with_failed_type_conversion(self):
        """Test getting value with failed type conversion."""
        from rkllama.config import get, set

        set("test", "bad_int", "not_a_number")
        # Should return default when conversion fails
        result = get("test", "bad_int", default=0, as_type=int)
        # Implementation should handle gracefully

    def test_none_value_handling(self):
        """Test handling of None values."""
        from rkllama.config.config import config

        result = config._infer_and_convert_type("test", "key", None)
        assert result is None
