#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RESET='\033[0m'

# Default values
PORT=8080

# Miniconda installation path
MINICONDA_DIR=~/miniconda3

# Check if --no-conda argument is passed
USE_CONDA=true
for arg in "$@"; do
    if [[ "$arg" == "--no-conda" ]]; then
        USE_CONDA=false
    elif [[ "$arg" == --port=* ]]; then
        PORT="${arg#*=}"
    fi
done

# If Miniconda is enabled, check if it exists and activate it
if $USE_CONDA; then
    if [ -d "$MINICONDA_DIR" ]; then
        echo -e "${GREEN}Starting the environment with Miniconda3.${RESET}"
        source "$MINICONDA_DIR/bin/activate" ""
    else
        echo -e "${YELLOW}Launching the installation file...${RESET}"
        # Download and install Miniconda
        bash ./setup.sh
    fi
else
    echo -e "${YELLOW}Miniconda is disabled. Running without it.${RESET}"
fi

# Function to prompt for CPU selection
select_cpu_model() {
    echo -e "${YELLOW}CPU model not detected automatically.${RESET}"
    echo "Please select your CPU model:"
    echo "1) rk3588"
    echo "2) rk3576"
    
    while true; do
        read -p "Enter selection (1-2): " selection
        case $selection in
            1) CPU_MODEL="rk3588"; break;;
            2) CPU_MODEL="rk3576"; break;;
            *) echo "Invalid selection. Please try again.";;
        esac
    done
    
    echo -e "${GREEN}Selected CPU model: $CPU_MODEL${RESET}"
}

# First check if CPU_MODEL is set in the environment
if [ -z "$CPU_MODEL" ]; then
  # Try to extract the CPU model from cpuinfo if not defined in environment
  CPU_MODEL=$(grep -i "cpu model" /proc/cpuinfo | head -n 1 | sed 's/.*Rockchip \(RK[0-9]*\).*/\1/' | tr '[:upper:]' '[:lower:]')
  echo "CPU_MODEL: $CPU_MODEL"
fi

# If still not detected, prompt for selection (only in interactive mode)
if [ -z "$CPU_MODEL" ]; then
  # Check if we're running in a container (non-interactive)
  if [ -f "/.dockerenv" ]; then
    echo -e "${YELLOW}Running in Docker without CPU_MODEL set. Defaulting to rk3588.${RESET}"
    CPU_MODEL="rk3588"
  else
    select_cpu_model
  fi
fi

python3 ~/RKLLAMA/server.py --target_platform "$CPU_MODEL" --port "$PORT"