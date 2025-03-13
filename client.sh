#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RESET='\033[0m'

# Determine script location to find application root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP_ROOT="$SCRIPT_DIR"
CONFIG_DIR="$APP_ROOT/config"

# Default port
PORT="8080"

# Source configuration if available
if [ -f "$CONFIG_DIR/config.env" ]; then
    source "$CONFIG_DIR/config.env"
    
    # Apply values from config.env if available
    if [ -n "$RKLLAMA_SERVER_PORT" ]; then
        PORT="$RKLLAMA_SERVER_PORT"
    fi
fi

# Parse arguments for port specification and other options
# Build a clean args array to pass to the Python script
PYTHON_ARGS=()

# Check for port specification and other args in arguments
for arg in "$@"; do
    if [[ "$arg" == --port=* ]]; then
        PORT="${arg#*=}"
        # This argument will be passed directly to Python
        PYTHON_ARGS+=("$arg")
    elif [[ "$arg" == "--no-conda" ]]; then
        USE_CONDA=false
        # Don't add this to Python args
    else
        # Pass through all other arguments
        PYTHON_ARGS+=("$arg")
    fi
done

# Miniconda installation path
MINICONDA_DIR=~/miniconda3

# Check for --no-conda flag
USE_CONDA=true
for arg in "$@"; do
    if [[ "$arg" == "--no-conda" ]]; then
        USE_CONDA=false
    fi
done

# If Miniconda is enabled, check if it exists and activate it
if $USE_CONDA; then
    if [ -d "$MINICONDA_DIR" ]; then
        source "$MINICONDA_DIR/bin/activate" ""
    else
        echo -e "${YELLOW}Miniconda3 is not installed. Please install it first or use --no-conda.${RESET}"
        exit 1
    fi
fi

# Execute the Python script with all arguments
python3 "$APP_ROOT/client.py" "${PYTHON_ARGS[@]}"