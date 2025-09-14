import os
import sys
import configparser
import argparse
import logging
import datetime
from typing import Any, Dict, Optional, Union, List, TypeVar, Type, Generic, cast, Callable, Tuple
from pathlib import Path

# Import schema for validation
from config_schema import RKLLAMA_SCHEMA, FieldType

# Configure logger
logger = logging.getLogger("rkllama.config")

# Type variable for generic return types
T = TypeVar('T')

class RKLLAMAConfig:
    """Centralized configuration system for RKLLAMA"""
    
    def __init__(self):
        self.app_root = self._determine_app_root()
        self.config_dir = self.app_root / "config"
        self.config = {}
        # Path cache stores resolved paths to avoid filesystem operations
        self._path_cache = {}
        # Type cache stores schema information to avoid lookups
        self._type_cache = {}
        
        # Create config directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Configuration loading follows priority order:
        self._load_defaults()      # Schema defaults (lowest priority)
        self._load_system_ini()    # System-wide settings 
        self._load_user_ini()      # User preferences
        self._load_project_ini()   # Project-specific settings
        self._load_env_vars()      # Environment variables
        # Command-line args (highest priority)
        
        # Generate shell configuration for environment exports
        self._generate_shell_config()
    
    def _get_field_info(self, section: str, key: str) -> Tuple[Optional[FieldType], Any]:
        """
        Get field type information from schema or cache.
        
        Returns:
            Tuple of (field_type, default_value) or (None, None)
        """
        # Check cache first
        cache_key = f"{section}.{key}"
        if cache_key in self._type_cache:
            return self._type_cache[cache_key]
            
        # Look up in schema
        schema_section = RKLLAMA_SCHEMA.get_section(section)
        if schema_section and key in schema_section.fields:
            field = schema_section.fields[key]
            result = (field.field_type, field.default)
            # Cache for future lookups
            self._type_cache[cache_key] = result
            return result
            
        # Not in schema
        return (None, None)
    
    def _infer_and_convert_type(self, section: str, key: str, value: str) -> Any:
        """
        Converts string values to appropriate Python types.
        
        Uses schema if available, otherwise applies heuristic type detection
        for booleans, numbers, and lists.
        """
        # Handle None values
        if value is None:
            return None
            
        # Check if we already know the type from schema
        field_type, default = self._get_field_info(section, key)
        if field_type is not None:
            # If we know the expected type, use schema validation
            try:
                from config_schema import ConfigField
                temp_field = ConfigField(field_type, default)
                return temp_field.validate(value)
            except ValueError:
                logger.warning(f"Schema validation failed for {section}.{key}={value}. Using default.")
                return default
        
        # No schema information, use heuristic type inference
        
        # For non-string values, return as is (already typed)
        if not isinstance(value, str):
            return value
            
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Handle numeric values
        try:
            # Try integer first
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return int(value)
            
            # Try float
            return float(value)
        except ValueError:
            pass
        
        # Handle lists (comma-separated values)
        if ',' in value:
            # Split and strip each value
            items = [item.strip() for item in value.split(',')]
            return items
            
        # Default to string for anything else
        return value
        
    def _determine_app_root(self) -> Path:
        """Finds the application root directory"""
        if getattr(sys, 'frozen', False):
            # Frozen application (PyInstaller)
            app_path = Path(sys.executable).parent
        else:
            # Regular Python script
            app_path = Path(__file__).parent
            
        return app_path
    
    def _load_defaults(self):
        """Loads default values from schema and creates default.ini file"""
        default_config = {}
        
        # Extract defaults from schema
        for section_name, section_schema in RKLLAMA_SCHEMA.sections.items():
            default_config[section_name] = {}
            for field_name, field in section_schema.fields.items():
                # Store the typed default value directly
                default_config[section_name][field_name] = field.default
        
        # Write default configuration to file if it doesn't exist
        default_ini_path = self.config_dir / "default.ini"
        if not default_ini_path.exists():
            config = configparser.ConfigParser()
            for section, values in default_config.items():
                config[section] = {k: str(v) for k, v in values.items()}
                
            with open(default_ini_path, "w") as f:
                config.write(f)
        
        self.config.update(default_config)
    
    def _load_config_file(self, config_path: Union[str, Path]):
        """
        Loads and parses an INI configuration file.
        Performs type conversion during loading.
        """
        if isinstance(config_path, str):
            config_path = Path(config_path)
            
        if not config_path.exists():
            return
        
        logger.debug(f"Loading configuration from: {config_path}")
        
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Convert to dictionary with proper type inference
        for section in config.sections():
            if section not in self.config:
                self.config[section] = {}
                
            for key, value in config[section].items():
                # Convert string value to appropriate type during loading
                typed_value = self._infer_and_convert_type(section, key, value)
                self.config[section][key] = typed_value
    
    def _load_system_ini(self):
        """Load system-wide configuration"""
        system_config_paths = [
            Path("/etc/rkllama/rkllama.ini"),
            Path("/etc/rkllama.ini"),
            Path("/usr/local/etc/rkllama.ini"),
            self.app_root / "system" / "rkllama.ini"
        ]
        
        for path in system_config_paths:
            if path.exists():
                self._load_config_file(path)
                logger.debug(f"Loaded system configuration from: {path}")
    
    def _load_user_ini(self):
        """Load user-specific configuration"""
        user_config_paths = [
            Path.home() / ".config" / "rkllama" / "rkllama.ini",
            Path.home() / ".config" / "rkllama.ini",
            Path.home() / ".rkllama.ini"
        ]
        
        for path in user_config_paths:
            if path.exists():
                self._load_config_file(path)
                logger.debug(f"Loaded user configuration from: {path}")
    
    def _load_project_ini(self):
        """Load project-specific configuration"""
        project_config_paths = [
            self.app_root / "rkllama.ini",
            self.app_root / "config" / "rkllama.ini"
        ]
        
        for path in project_config_paths:
            if path.exists():
                self._load_config_file(path)
                logger.debug(f"Loaded project configuration from: {path}")
    
    def _load_env_vars(self):
        """
        Load configuration from environment variables.
        Environment variables override ini files.
        """
        # Pattern: RKLLAMA_SECTION_KEY
        for env_var, value in os.environ.items():
            if not env_var.startswith("RKLLAMA_"):
                continue
            
            # Special case for RKLLAMA_DEBUG environment variable
            if env_var == "RKLLAMA_DEBUG":
                if value.lower() in ["1", "true", "yes", "on"]:
                    self.set("server", "debug", True)
                elif value.lower() in ["0", "false", "no", "off"]:
                    self.set("server", "debug", False)
                continue
                
            parts = env_var.split("_")
            if len(parts) < 3:
                continue
                
            section = parts[1].lower()
            key = "_".join(parts[2:]).lower()
            
            if section not in self.config:
                self.config[section] = {}
            
            # Convert environment variable value to appropriate type
            typed_value = self._infer_and_convert_type(section, key, value)
            
            # Environment variables take precedence over ini files
            self.config[section][key] = typed_value
            logger.debug(f"Loaded config from environment: {env_var}={typed_value}")
    
    def load_args(self, args: argparse.Namespace):
        """
        Load configuration from command-line arguments.
        Command-line args have the highest priority.
        """
        # Clear any previous command-line args to ensure clean state
        self._clear_command_line_args()
        
        # Extract all args and apply them
        if args:
            # Handle common explicit arguments
            if hasattr(args, "port") and args.port is not None:
                self.set("server", "port", args.port)
                
            if hasattr(args, "debug") and args.debug:
                self.set("server", "debug", True)
                
            if hasattr(args, "processor") and args.processor:
                self.set("platform", "processor", args.processor)
            
            if hasattr(args, "config") and args.config:
                # Load custom config file with highest priority
                custom_config = Path(args.config)
                if custom_config.exists():
                    self._load_config_file(custom_config)
                else:
                    logger.warning(f"Specified config file not found: {args.config}")
                
            # Look for any other args of the form section_key
            for arg_name, arg_value in vars(args).items():
                if arg_value is None:
                    continue
                    
                if '_' in arg_name:
                    try:
                        section, key = arg_name.split('_', 1)
                        # Command-line args are already typed, so use them directly
                        self.set(section, key, arg_value)
                    except ValueError:
                        # Not a valid section_key pattern
                        continue
    
    def _clear_command_line_args(self):
        """
        Clear any settings that were previously set by command line args.
        This is a placeholder as we don't currently track which settings came from args.
        In a future improvement, we could track the source of each setting.
        """
        # Future implementation could restore values from lower priority sources
        pass
    
    def resolve_path(self, path: str) -> str:
        """Resolve a path relative to the application root"""
        if not path:
            return None
        
        # Check if we have this path in the cache
        if path in self._path_cache:
            return self._path_cache[path]
            
        path_obj = Path(path)
        
        if path_obj.is_absolute():
            resolved = str(path_obj)
        elif '$' in path or '~' in path:
            # Check if path contains environment variables and expand them
            expanded_path = os.path.expanduser(os.path.expandvars(path))
            if os.path.isabs(expanded_path):
                resolved = expanded_path
            else:
                # Relative to app root after expansion
                resolved = str(self.app_root / expanded_path)
        else:
            # Relative to app root
            resolved = str(self.app_root / path)
        
        # Cache the result
        self._path_cache[path] = resolved
        return resolved
    
    def _clear_path_cache(self):
        """Clear the path resolution cache"""
        self._path_cache = {}
    
    def set(self, section: str, key: str, value: Any):
        """
        Sets a configuration value.
        
        Performs schema validation if available.
        Handles type inference for string values.
        Manages path cache for path settings.
        """
        if section not in self.config:
            self.config[section] = {}
        
        # Check if we're updating a path in the 'paths' section
        invalidate_path_cache = (section == "paths" and 
            (key not in self.config.get(section, {}) or 
             self.config[section].get(key) != value))
        
        # Get field type information from schema
        field_type, default_value = self._get_field_info(section, key)
        
        if field_type is not None:
            # We have schema information - validate the value
            try:
                from config_schema import ConfigField
                temp_field = ConfigField(field_type, default_value)
                validated_value = temp_field.validate(value)
                self.config[section][key] = validated_value
            except ValueError as e:
                logger.warning(f"Invalid value for {section}.{key}: {value} - {str(e)}")
                self.config[section][key] = default_value
        else:
            # No schema - use type inference for strings only
            if isinstance(value, str):
                self.config[section][key] = self._infer_and_convert_type(section, key, value)
            else:
                # Store non-string values directly
                self.config[section][key] = value
        
        # If a path was modified, clear the path cache
        if invalidate_path_cache:
            self._clear_path_cache()
            
        # Re-generate shell config when values change
        self._generate_shell_config()
    
    def get(self, section: str, key: str, default: Any = None, as_type: Optional[Union[FieldType, type]] = None) -> Any:
        """
        Retrieves a configuration value with optional type conversion.
        
        Args:
            section: Configuration section name
            key: Configuration key name
            default: Default value if key doesn't exist
            as_type: Type to convert to (FieldType or Python type)
            
        Returns:
            Typed configuration value or default
        """
        # Get raw value
        if section not in self.config:
            return default
            
        if key not in self.config[section]:
            return default
            
        value = self.config[section][key]
        
        # If no type conversion requested, return as is
        if as_type is None:
            return value
            
        # Handle FieldType enum values
        if isinstance(as_type, FieldType):
            return self._convert_to_field_type(value, as_type, section, key, default)
        
        # Handle Python types
        if as_type is bool:
            if isinstance(value, bool):
                return value
            return self._convert_to_field_type(value, FieldType.BOOLEAN, section, key, default)
            
        if as_type is int:
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            return self._convert_to_field_type(value, FieldType.INTEGER, section, key, default)
            
        if as_type is float:
            if isinstance(value, float):
                return value
            if isinstance(value, int) and not isinstance(value, bool):
                return float(value)
            return self._convert_to_field_type(value, FieldType.FLOAT, section, key, default)
            
        if as_type is list or as_type is List:
            if isinstance(value, list):
                return value
            return self._convert_to_field_type(value, FieldType.LIST, section, key, default)
            
        if as_type is str:
            if isinstance(value, str):
                return value
            if value is None:
                return default if default is not None else ""
            return str(value)
            
        # For any other type, try direct casting
        if value is not None:
            try:
                return as_type(value)
            except (ValueError, TypeError):
                logger.warning(f"Failed to convert {section}.{key} to {as_type.__name__}")
                
        return default
    
    def _convert_to_field_type(self, value: Any, field_type: FieldType, section: str, key: str, default: Any) -> Any:
        """
        Converts a value to the specified FieldType.
        Contains optimized paths for common type scenarios.
        """
        # Fast path for correct types
        if field_type == FieldType.BOOLEAN and isinstance(value, bool):
            return value
        elif field_type == FieldType.INTEGER and isinstance(value, int) and not isinstance(value, bool):
            return value
        elif field_type == FieldType.FLOAT and isinstance(value, float):
            return value
        elif field_type == FieldType.FLOAT and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        elif field_type == FieldType.LIST and isinstance(value, list):
            return value
        elif field_type == FieldType.STRING and isinstance(value, str):
            return value
        
        # Need conversion
        try:
            from config_schema import ConfigField
            temp_field = ConfigField(field_type, default)
            return temp_field.validate(value)
        except (ValueError, TypeError):
            logger.warning(f"Type conversion failed for {section}.{key}, expected {field_type.value}, using default")
            return default
    
    def get_path(self, key: str, default: Any = None) -> str:
        """
        Retrieves a path configuration and resolves it.
        Path resolution includes app_root and environment variable expansion.
        """
        path = self.get("paths", key, default)
        return self.resolve_path(path) if path else None
    
    def _generate_shell_config(self):
        """
        Creates a shell script with environment variables.
        Useful for sourcing in shell scripts or CI/CD pipelines.
        """
        config_env_path = self.config_dir / "config.env"
        
        lines = [
            "#!/bin/sh",
            "# Auto-generated shell configuration for RKLLAMA",
            f"# Generated at: {datetime.datetime.now().isoformat()}",
            "",
            "# Application root",
            f"RKLLAMA_ROOT=\"{self.app_root}\"",
            ""
        ]
        
        # Add all configuration values
        for section, values in self.config.items():
            lines.append(f"# {section.upper()} configuration")
            for key, value in values.items():
                # Convert to shell variable format
                env_var = f"RKLLAMA_{section.upper()}_{key.upper()}"
                # Convert typed values to string representation for shell
                str_value = str(value)
                
                # Handle special cases for shell variables
                if isinstance(value, bool):
                    str_value = "1" if value else "0"
                elif isinstance(value, list):
                    str_value = ",".join(str(item) for item in value)
                
                lines.append(f"{env_var}=\"{str_value}\"")
                
                # Special case for paths - add resolved paths as well
                if section == "paths":
                    resolved_path = self.resolve_path(str_value)
                    lines.append(f"{env_var}_RESOLVED=\"{resolved_path}\"")
            
            lines.append("")
        
        # Write to file
        with open(config_env_path, "w") as f:
            f.write("\n".join(lines))
            
        # Make the file executable
        os.chmod(config_env_path, 0o755)
        
        logger.debug(f"Generated shell configuration: {config_env_path}")
        
    def display(self):
        """Logs the current configuration values"""
        logger.info("Current RKLLAMA Configuration:")
        for section, values in self.config.items():
            logger.info(f"[{section}]")
            for key, value in values.items():
                logger.info(f"  {key} = {value}")
                
    def validate(self):
        """
        Validates configuration against schema.
        Creates required directories for path settings.
        """
        errors = []
        
        # Use schema to validate all sections
        for section_name, section_schema in RKLLAMA_SCHEMA.sections.items():
            if section_name in self.config:
                try:
                    # Validate section values against schema
                    validated_values = section_schema.validate_section(self.config[section_name])
                    # Update with validated values
                    self.config[section_name] = validated_values
                except ValueError as e:
                    errors.append(f"Validation error in section '{section_name}': {str(e)}")
            
        # Validate paths
        for key in ["models", "logs", "data", "temp"]:
            path = self.get_path(key)
            if path:  # Only check if path is not None
                if not os.path.exists(path):
                    try:
                        os.makedirs(path)
                        logger.info(f"Created directory: {path}")
                    except Exception as e:
                        errors.append(f"Failed to create {key} directory: {str(e)}")
        
        # Report any errors
        if errors:
            for error in errors:
                logger.error(error)
            return False
            
        return True
    
    def save_to_project_ini(self):
        """
        Saves current configuration to project INI file.
        Converts typed values back to strings.
        """
        project_config_path = os.path.join(self.app_root, "rkllama.ini")
        config = configparser.ConfigParser()
        
        # Add all sections and keys
        for section, values in self.config.items():
            if section not in config:
                config[section] = {}
            for key, value in values.items():
                # Convert typed values back to strings for INI file
                config[section][key] = str(value)
        
        # Write to file
        with open(project_config_path, "w") as f:
            config.write(f)
            
        logger.info(f"Saved configuration to {project_config_path}")
        
        # Re-generate shell config
        self._generate_shell_config()
    
    def is_debug_mode(self) -> bool:
        """Checks if debug mode is enabled"""
        # Simply use the current config setting which already follows our hierarchy
        return self.get("server", "debug", False, as_type=bool)
            
    def reload_config(self):
        """
        Reloads all configuration from all sources.
        Maintains priority order and preserves command-line arguments.
        """
        # Save any command-line args temporarily if they exist
        cmd_args = {}
        for section, values in self.config.items():
            if section not in cmd_args:
                cmd_args[section] = {}
            for key, value in values.items():
                cmd_args[section][key] = value
        
        # Clear current config and caches
        self.config = {}
        self._clear_path_cache()
        self._type_cache = {}
        
        # Reload in proper priority order
        self._load_defaults()
        self._load_system_ini()
        self._load_user_ini()
        self._load_project_ini()
        self._load_env_vars()
        
        # Re-apply stored command-line args with highest priority
        for section, values in cmd_args.items():
            if section not in self.config:
                self.config[section] = {}
            for key, value in values.items():
                self.config[section][key] = value
        
        # Re-generate shell config
        self._generate_shell_config()
        
        logger.debug("Configuration reloaded")

# Singleton instance
config = RKLLAMAConfig()

# Updated convenience functions for module-level access
def get(section: str, key: str, default: Any = None, as_type: Optional[Union[FieldType, type]] = None) -> Any:
    """
    Retrieves a configuration value with optional type conversion.
    
    Examples:
        # Get a string value
        name = get("app", "name", "DefaultApp")
        
        # Get with type conversion
        port = get("server", "port", 8080, as_type=int)
        debug = get("server", "debug", False, as_type=bool)
        hosts = get("server", "allowed_hosts", [], as_type=list)
    """
    return config.get(section, key, default, as_type)
    
def set(section: str, key: str, value: Any):
    """Set a configuration value"""
    config.set(section, key, value)
    
def get_path(key: str, default: Any = None) -> str:
    """Get a path configuration value"""
    return config.get_path(key, default)

def display():
    """Display the current configuration"""
    config.display()
    
def validate():
    """Validate the current configuration"""
    return config.validate()
    
def load_args(args: argparse.Namespace):
    """Load configuration from command-line arguments"""
    config.load_args(args)

def save_to_project_ini():
    """Save current configuration to project INI file"""
    config.save_to_project_ini()

def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return config.get("server", "debug", False, as_type=bool)

def reload_config():
    """Reload configuration from all sources"""
    config.reload_config()
