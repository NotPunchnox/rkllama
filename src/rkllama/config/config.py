"""Configuration management for RKLlama using pydantic-settings."""

import argparse
import configparser
import datetime
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rkllama.config.config_schema import RKLlamaSettings
from rkllama.logging import get_logger

logger = get_logger("rkllama.config")


class RKLlamaConfig:
    """Centralized configuration system for RKLlama using pydantic-settings."""

    def __init__(self):
        self.app_root = self._determine_app_root()
        self.config_dir = self.app_root / "config"
        self._path_cache: dict[str, str] = {}

        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)

        # Load configuration in priority order (lowest to highest):
        # 1. Defaults from pydantic models
        # 2. System INI files
        # 3. User INI files
        # 4. Project INI files
        # 5. Environment variables (handled automatically by pydantic-settings)
        # 6. Command-line args (applied via load_args)

        # Start with defaults
        self._settings = RKLlamaSettings()

        # Load INI files in priority order
        self._load_ini_files()

        # Generate shell configuration for environment exports
        self._generate_shell_config()

    def _determine_app_root(self) -> Path:
        """Find the application root directory."""
        if getattr(sys, "frozen", False):
            # Frozen application (PyInstaller)
            return Path(sys.executable).parent
        else:
            # Regular Python script - go up from config/ to rkllama/
            return Path(__file__).parent.parent

    def _load_ini_files(self) -> None:
        """Load configuration from INI files in priority order."""
        ini_paths = [
            # System-wide
            Path("/etc/rkllama/rkllama.ini"),
            Path("/etc/rkllama.ini"),
            Path("/usr/local/etc/rkllama.ini"),
            self.app_root / "system" / "rkllama.ini",
            # User-specific
            Path.home() / ".config" / "rkllama" / "rkllama.ini",
            Path.home() / ".config" / "rkllama.ini",
            Path.home() / ".rkllama.ini",
            # Project-specific
            self.app_root / "rkllama.ini",
            self.app_root / "config" / "rkllama.ini",
        ]

        for path in ini_paths:
            if path.exists():
                self._load_ini_file(path)

    def _load_ini_file(self, path: Path) -> None:
        """Load and apply settings from an INI file."""
        logger.debug("Loading configuration", path=str(path))

        config = configparser.ConfigParser()
        config.read(path)

        for section in config.sections():
            section_lower = section.lower()
            for key, value in config[section].items():
                self.set(section_lower, key.lower(), value)

    def _parse_value(self, value: str) -> Any:
        """Parse a string value into appropriate Python type."""
        if not isinstance(value, str):
            return value

        # Boolean
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Integer
        try:
            if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                return int(value)
        except (ValueError, AttributeError):
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # List (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",")]

        return value

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a configuration value."""
        # Parse string values
        if isinstance(value, str):
            value = self._parse_value(value)

        # Get the section object
        section_obj = getattr(self._settings, section, None)
        if section_obj is None:
            logger.warning("Unknown config section", section=section)
            return

        # Set the value if it's a valid field
        if hasattr(section_obj, key):
            try:
                # Create a new section with the updated value
                section_dict = section_obj.model_dump()
                section_dict[key] = value

                # Reconstruct the section with validation
                section_class = type(section_obj)
                new_section = section_class(**section_dict)
                setattr(self._settings, section, new_section)

                # Invalidate path cache if paths changed
                if section == "paths":
                    self._path_cache.clear()

            except ValidationError as e:
                logger.warning("Invalid config value", section=section, key=key, value=value, error=str(e))
        else:
            logger.debug("Unknown config key", section=section, key=key)

    def get(self, section: str, key: str, default: Any = None, as_type: type | None = None) -> Any:
        """Get a configuration value with optional type conversion."""
        section_obj = getattr(self._settings, section, None)
        if section_obj is None:
            return default

        value = getattr(section_obj, key, default)
        if value is None:
            return default

        # Type conversion if requested
        if as_type is not None:
            try:
                if as_type is bool:
                    if isinstance(value, bool):
                        return value
                    if isinstance(value, str):
                        return value.lower() in ("true", "yes", "1", "on")
                    return bool(value)
                return as_type(value)
            except (ValueError, TypeError):
                logger.warning("Type conversion failed", section=section, key=key, target_type=as_type.__name__)
                return default

        return value

    def get_path(self, key: str, default: Any = None) -> str | None:
        """Get a path configuration value and resolve it."""
        # Check cache first
        if key in self._path_cache:
            return self._path_cache[key]

        path = self.get("paths", key, default)
        if path is None:
            return None

        resolved = self._settings.resolve_path(path, self.app_root)
        if resolved:
            self._path_cache[key] = resolved
        return resolved

    def load_args(self, args: argparse.Namespace) -> None:
        """Load configuration from command-line arguments (highest priority)."""
        if not args:
            return

        if hasattr(args, "port") and args.port is not None:
            self.set("server", "port", args.port)

        if hasattr(args, "debug") and args.debug:
            self.set("server", "debug", True)

        if hasattr(args, "processor") and args.processor:
            self.set("platform", "processor", args.processor)

        if hasattr(args, "models") and args.models:
            self.set("paths", "models", args.models)

        if hasattr(args, "config") and args.config:
            custom_config = Path(args.config)
            if custom_config.exists():
                self._load_ini_file(custom_config)
            else:
                logger.warning("Config file not found", path=args.config)

    def validate(self) -> bool:
        """Validate configuration and create required directories."""
        errors = []

        # Validate paths and create directories
        for key in ["models", "logs", "data", "temp"]:
            path = self.get_path(key)
            if path and not os.path.exists(path):
                try:
                    os.makedirs(path)
                    logger.info("Created directory", path=path)
                except Exception as e:
                    errors.append(f"Failed to create {key} directory: {e}")

        if errors:
            for error in errors:
                logger.error("Validation error", error=error)
            return False

        return True

    def display(self) -> None:
        """Log the current configuration values."""
        logger.info("Current RKLlama Configuration")
        for section_name in ["server", "paths", "model", "platform"]:
            section = getattr(self._settings, section_name, None)
            if section:
                for key, value in section.model_dump().items():
                    logger.info("Config value", section=section_name, key=key, value=value)

    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get("server", "debug", False, as_type=bool)

    def _generate_shell_config(self) -> None:
        """Create a shell script with environment variables."""
        config_env_path = self.config_dir / "config.env"

        lines = [
            "#!/bin/sh",
            "# Auto-generated shell configuration for RKLlama",
            f"# Generated at: {datetime.datetime.now().isoformat()}",
            "",
            "# Application root",
            f'RKLLAMA_ROOT="{self.app_root}"',
            "",
        ]

        for section_name in ["server", "paths", "model", "platform"]:
            section = getattr(self._settings, section_name, None)
            if section:
                lines.append(f"# {section_name.upper()} configuration")
                for key, value in section.model_dump().items():
                    env_var = f"RKLLAMA_{section_name.upper()}_{key.upper()}"

                    if isinstance(value, bool):
                        str_value = "1" if value else "0"
                    elif isinstance(value, list):
                        str_value = ",".join(str(item) for item in value)
                    else:
                        str_value = str(value)

                    lines.append(f'{env_var}="{str_value}"')

                    if section_name == "paths":
                        resolved = self._settings.resolve_path(str(value), self.app_root)
                        lines.append(f'{env_var}_RESOLVED="{resolved}"')
                lines.append("")

        with open(config_env_path, "w") as f:
            f.write("\n".join(lines))

        os.chmod(config_env_path, 0o755)
        logger.debug("Generated shell configuration", path=str(config_env_path))

    def save_to_project_ini(self) -> None:
        """Save current configuration to project INI file."""
        project_config_path = self.app_root / "rkllama.ini"
        config = configparser.ConfigParser()

        for section_name in ["server", "paths", "model", "platform"]:
            section = getattr(self._settings, section_name, None)
            if section:
                config[section_name] = {k: str(v) for k, v in section.model_dump().items()}

        with open(project_config_path, "w") as f:
            config.write(f)

        logger.info("Saved configuration", path=str(project_config_path))
        self._generate_shell_config()

    def reload_config(self) -> None:
        """Reload configuration from all sources."""
        self._path_cache.clear()
        self._settings = RKLlamaSettings()
        self._load_ini_files()
        self._generate_shell_config()
        logger.debug("Configuration reloaded")


# Singleton instance
config = RKLlamaConfig()


# Module-level convenience functions
def get(section: str, key: str, default: Any = None, as_type: type | None = None) -> Any:
    """Get a configuration value with optional type conversion."""
    return config.get(section, key, default, as_type)


def set(section: str, key: str, value: Any) -> None:
    """Set a configuration value."""
    config.set(section, key, value)


def get_path(key: str, default: Any = None) -> str | None:
    """Get a path configuration value."""
    return config.get_path(key, default)


def display() -> None:
    """Display the current configuration."""
    config.display()


def validate() -> bool:
    """Validate the current configuration."""
    return config.validate()


def load_args(args: argparse.Namespace) -> None:
    """Load configuration from command-line arguments."""
    config.load_args(args)


def save_to_project_ini() -> None:
    """Save current configuration to project INI file."""
    config.save_to_project_ini()


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return config.is_debug_mode()


def reload_config() -> None:
    """Reload configuration from all sources."""
    config.reload_config()
