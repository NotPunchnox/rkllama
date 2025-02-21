#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RESET='\033[0m'


CPU_MODEL = ""
PORT      = ""

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
        echo -e "${GREEN}Starting the environment with Miniconda3.${RESET}"
        (source "$MINICONDA_DIR/bin/activate" "")
    else
        echo -e "${YELLOW}Launching the installation file...${RESET}"
        # Download and install Miniconda
        bash ./setup.sh
    fi
else
    echo -e "${YELLOW}Miniconda is disabled. Running without it.${RESET}"
fi

# Extract the CPU model if not defined in the environment
if [ -z "$CPU_MODEL" ]; then
    CPU_MODEL=$(cat /proc/cpuinfo | grep "cpu model" | head -n 1 | sed 's/.*Rockchip \(RK[0-9]*\).*/\1/' | tr '[:upper:]' '[:lower:]')
    echo "CPU_MODEL: $CPU_MODEL"
fi

if [ -z "$CPU_MODEL" ]; then
    echo "Erreur : CPU model not found !" 
    exit 1
fi

# Start the server with the CPU model as a parameter
python3 ~/RKLLAMA/server.py --target_platform "$CPU_MODEL" --port $PORT