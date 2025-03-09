#!/bin/bash

# Define colors
GREEN='\033[0;32m'
RESET='\033[0m'

# Parse arguments for port specification
PORT="8080"  # Default port

# Check for port specification in arguments
for arg in "$@"; do
    if [[ "$arg" == --port=* ]]; then
        PORT="${arg#*=}"
        # Remove this argument from parameters passed to the Python script
        set -- "${@/$arg/}"
    fi
done

# Update the rkllama.ini file with the port if specified
CONFIG_FILE=~/RKLLAMA/rkllama.ini
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating config file with default port..."
    mkdir -p "$(dirname "$CONFIG_FILE")"
    echo "[server]" > "$CONFIG_FILE"
    echo "port = 8080" >> "$CONFIG_FILE"
fi

# If port was specified, update the config
if [ "$PORT" != "8080" ]; then
    echo "Updating port to $PORT..."
    if grep -q "^\[server\]" "$CONFIG_FILE"; then
        # Section exists, update port
        sed -i "s/^port = .*/port = $PORT/" "$CONFIG_FILE"
    else
        # Section doesn't exist, create it
        echo "[server]" >> "$CONFIG_FILE"
        echo "port = $PORT" >> "$CONFIG_FILE"
    fi
fi

# Miniconda installation path
MINICONDA_DIR=~/miniconda3

# Check for --no-conda flag
USE_CONDA=true
for arg in "$@"; do
    if [[ "$arg" == "--no-conda" ]]; then
        USE_CONDA=false
        # Remove this argument from parameters passed to the Python script
        set -- "${@/$arg/}"
    fi
done

# If Miniconda is enabled, check if it exists and activate it
if $USE_CONDA; then
    if [ -d "$MINICONDA_DIR" ]; then
        source "$MINICONDA_DIR/bin/activate" ""
    else
        echo -e "Miniconda3 is not installed. Please install it first or use --no-conda."
        exit 1
    fi
fi

# Execute the Python script with the remaining arguments
python3 ~/RKLLAMA/client.py "$@"