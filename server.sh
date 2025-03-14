#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RESET='\033[0m'

# Determine script location to find application root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP_ROOT="$SCRIPT_DIR"
CONFIG_DIR="$APP_ROOT/config"

# Default values
PORT="8080"  # Default port
DEBUG_MODE=false

# Miniconda installation path
MINICONDA_DIR=~/miniconda3

# Source configuration if available
if [ -f "$CONFIG_DIR/config.env" ]; then
    source "$CONFIG_DIR/config.env"
    
    # Apply values from config.env if available
    if [ -n "$RKLLAMA_SERVER_PORT" ]; then
        PORT="$RKLLAMA_SERVER_PORT"
    fi
    if [ "$RKLLAMA_SERVER_DEBUG" = "true" ]; then
        DEBUG_MODE=true
    fi
fi

# Parse command line arguments (override config)
USE_CONDA=true
for arg in "$@"; do
    if [[ "$arg" == "--no-conda" ]]; then
        USE_CONDA=false
    elif [[ "$arg" == "--debug" ]]; then
        DEBUG_MODE=true
    elif [[ "$arg" == --port=* ]]; then
        PORT="${arg#*=}"
        # Validate that port is not empty
        if [[ -z "$PORT" ]]; then
            echo -e "${YELLOW}Warning: Empty port specified, using default port 8080${RESET}"
            PORT="8080"
        fi
    fi
done

# Update the configuration
CONFIG_ARGS=()
if [ -n "$PORT" ]; then
    CONFIG_ARGS+=("--port" "$PORT")
fi
if $DEBUG_MODE; then
    CONFIG_ARGS+=("--debug")
fi

# If Miniconda is enabled, check if it exists and activate it
if $USE_CONDA; then
    if [ -d "$MINICONDA_DIR" ]; then
        echo -e "${GREEN}Starting the environment with Miniconda3.${RESET}"
        source "$MINICONDA_DIR/bin/activate" ""
    else
        echo -e "${YELLOW}Launching the installation file...${RESET}"
        # Download and install Miniconda
        bash "$APP_ROOT/setup.sh"
    fi
else
    echo -e "${YELLOW}Miniconda is disabled. Running without it.${RESET}"
fi

# Function to prompt for processor selection
select_processor() {
    echo -e "${YELLOW}Processor type not detected automatically.${RESET}"
    echo "Please select your processor type:"
    echo "1) rk3588"
    echo "2) rk3576"
    
    while true; do
        read -p "Enter selection (1-2): " selection
        case $selection in
            1) PROCESSOR="rk3588"; break;;
            2) PROCESSOR="rk3576"; break;;
            *) echo "Invalid selection. Please try again.";;
        esac
    done
    
    echo -e "${GREEN}Selected processor: $PROCESSOR${RESET}"
}

# First check if PROCESSOR or CPU_MODEL is set in the environment
if [ -z "$PROCESSOR" ]; then
  if [ -n "$CPU_MODEL" ]; then
    # Use CPU_MODEL if available for backward compatibility
    PROCESSOR="$CPU_MODEL"
  elif [ -n "$RKLLAMA_PLATFORM_PROCESSOR" ]; then
    # Use platform.processor setting from config
    PROCESSOR="$RKLLAMA_PLATFORM_PROCESSOR"
  else
    # Try to extract the processor type from cpuinfo if not defined
    PROCESSOR=$(grep -i "cpu model" /proc/cpuinfo | head -n 1 | sed 's/.*Rockchip \(RK[0-9]*\).*/\1/' | tr '[:upper:]' '[:lower:]')
  fi
  echo "Processor: $PROCESSOR"
fi

# If still not detected, prompt for selection (only in interactive mode)
if [ -z "$PROCESSOR" ]; then
  # Check if we're running in a container (non-interactive)
  if [ -f "/.dockerenv" ]; then
    echo -e "${YELLOW}Running in Docker without processor specified. Defaulting to rk3588.${RESET}"
    PROCESSOR="rk3588"
  else
    select_processor
  fi
fi

# Build full command with all arguments
COMMAND=("python3" "$APP_ROOT/server.py" "--processor" "$PROCESSOR" "--port" "$PORT")

# Add debug flag if enabled
if $DEBUG_MODE; then
    COMMAND+=("--debug")
fi

# Execute the Python script
"${COMMAND[@]}"