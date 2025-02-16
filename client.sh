#!/bin/bash

# Define colors
YELLOW='\033[1;33m'
RESET='\033[0m'

# Miniconda installation path
MINICONDA_DIR=~/miniconda3

# Check if --no-conda argument is passed
USE_CONDA=true
for arg in "$@"; do
    if [[ "$arg" == "--no-conda" ]]; then
        USE_CONDA=false
        break
    fi
done

# If Miniconda is enabled, check if it exists and activate it
if $USE_CONDA; then
    if [ -d "$MINICONDA_DIR" ]; then
        source "$MINICONDA_DIR/bin/activate"
    else
        echo -e "${YELLOW}Miniconda not found. Running without it.${RESET}"
    fi
fi

# Launch the client with the mentioned arguments
python3 ~/RKLLAMA/client.py "$@"