#!/bin/bash

# Miniconda installation path
MINICONDA_DIR=~/miniconda3

# Check if Miniconda is installed
if [ -d "$MINICONDA_DIR" ]; then
    echo -e "${GREEN}Starting the environment with Miniconda3.${RESET}"
    source "$MINICONDA_DIR/bin/activate"
else
    echo -e "${YELLOW}Launching the installation file...${RESET}"
    # Download and install Miniconda
    bash ./setup.sh
fi

# Extracting the CPU model
cpu_model=$(cat /proc/cpuinfo | grep "cpu model" | head -n 1 | sed 's/.*Rockchip \(RK[0-9]*\).*/\1/' | tr '[:upper:]' '[:lower:]')

# Starting the server with the CPU model as a parameter
python3 ~/RKLLAMA/server.py --target_platform "$cpu_model"
