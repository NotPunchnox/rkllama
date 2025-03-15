# RKLLAMA Configuration System

The RKLLAMA configuration system provides a flexible, type-safe mechanism for managing application settings from multiple sources with proper prioritization.

## Configuration Sources

Settings are loaded in order of increasing priority:

1. **Schema defaults** - Default values defined in the schema
2. **System settings** - System-wide configuration files
3. **User settings** - User-specific configuration files
4. **Project settings** - Local project configuration
5. **Environment variables** - Settings from environment variables
6. **Command-line arguments** - Highest priority settings

## Configuration Files

RKLLAMA searches for configuration files in the following locations:

### System Configuration
- `/etc/rkllama/rkllama.ini`
- `/etc/rkllama.ini`
- `/usr/local/etc/rkllama.ini`
- `<app_root>/system/rkllama.ini`

### User Configuration
- `~/.config/rkllama/rkllama.ini`
- `~/.config/rkllama.ini`
- `~/.rkllama.ini`

### Project Configuration
- `<app_root>/rkllama.ini`
- `<app_root>/config/rkllama.ini`

## Configuration Format

RKLLAMA uses standard INI format with sections and key-value pairs:

```ini
[server]
port = 8080
debug = false

[paths]
models = models
data = data
logs = logs/rkllama
```

## Environment Variables

Environment variables can override settings using the format `RKLLAMA_SECTION_KEY`.

For example:
- `RKLLAMA_SERVER_PORT=9090` sets the server port to 9090
- `RKLLAMA_PATHS_MODELS=/opt/models` sets the models path

Special environment variables:
- `RKLLAMA_DEBUG=(1|true|yes|on)` enables debug mode
- `RKLLAMA_DEBUG=(0|false|no|off)` disables debug mode

## Command-line Arguments

RKLLAMA supports command-line arguments with the highest priority:

```bash
# Load specific configuration file
python -m rkllama --config /path/to/config.ini

# Set server section values
python -m rkllama --server_port 8080 --server_debug
```

## Using the Configuration API

### Basic Usage

```python
from rkllama import config

# Get values with automatic type inference
port = config.get("server", "port")
debug = config.get("server", "debug")

# Get values with explicit types
port = config.get("server", "port", 8080, as_type=int)
debug = config.get("server", "debug", False, as_type=bool)

# Get path values (resolved against app_root)
models_dir = config.get_path("models")

# Set values
config.set("server", "port", 9090)
config.set("logging", "level", "DEBUG")
```

### Displaying and Saving Configuration

```python
# Display current configuration
config.display()

# Validate configuration
if not config.validate():
    print("Configuration validation failed!")

# Save to project configuration file
config.save_to_project_ini()
```

### Reloading Configuration

```python
# Reload configuration from all sources
config.reload_config()
```

## Type Conversion

The configuration system automatically handles type conversion based on the schema or heuristic detection:

- `true`, `yes`, `on`, `1` → Boolean `True`
- `false`, `no`, `off`, `0` → Boolean `False`
- Numeric strings → integers or floats
- Comma-separated values → lists
- Everything else → strings

## Path Resolution

Paths are automatically resolved relative to the application root directory. The system also supports:

- Absolute paths
- Environment variable expansion (`$HOME/data`)
- User home expansion (`~/models`) 

Resolved paths are cached for performance, particularly for frequently accessed paths.

## Shell Integration

The configuration system generates a shell environment file at `<app_root>/config/config.env` that exports all settings as environment variables. This file can be sourced in shell scripts:

```bash
source /path/to/rkllama/config/config.env
echo $RKLLAMA_SERVER_PORT
```
