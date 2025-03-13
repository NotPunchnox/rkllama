#!/bin/bash

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Determine script location to find application root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP_ROOT="$SCRIPT_DIR"
CONFIG_DIR="$APP_ROOT/config"

# Source configuration if available
if [ -f "$CONFIG_DIR/config.env" ]; then
    source "$CONFIG_DIR/config.env"
fi

# Confirmation prompt
echo -e "${YELLOW}Are you sure you want to uninstall RKLLama? This action will delete all associated files.${RESET}"
read -p "Type 'y' to continue, 'n' to cancel: " confirmation

if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
    echo -e "${RED}Uninstallation canceled.${RESET}"
    exit 1
fi

# Ask if the user wants to back up the models before uninstallation
echo -e "${YELLOW}Do you want to back up the models?${RESET}"
read -p "Type 'y' to continue, 'n' to cancel: " confirmation_models

if [[ "$confirmation_models" == "y" || "$confirmation_models" == "Y" ]]; then
    # Get models directory from configuration or use default
    MODELS_DIR="${RKLLAMA_PATHS_MODELS_RESOLVED:-$APP_ROOT/models}"
    BACKUP_DIR="/home/$(whoami)/Desktop/rkllm_models"
    
    # Back up the models
    echo -e "${GREEN}The models will be saved to $BACKUP_DIR${RESET}"
    # Create the directory if needed
    mkdir -p "$BACKUP_DIR"
    
    # Only copy if models directory exists and contains files
    if [ -d "$MODELS_DIR" ] && [ "$(ls -A "$MODELS_DIR")" ]; then
        cp "$MODELS_DIR"/*.rkllm "$BACKUP_DIR" 2>/dev/null
        echo -e "${GREEN}Models have been successfully backed up.${RESET}"
    else
        echo -e "${YELLOW}No models found to back up.${RESET}"
    fi
fi

# Start uninstallation message
echo -e "${CYAN}Starting uninstallation...${RESET}"

# Check if the RKLLAMA directory exists
if [ -d "$HOME/RKLLAMA" ]; then
    echo -e "${YELLOW}Deleting the ~/RKLLAMA directory...${RESET}"
    rm -rf ~/RKLLAMA/
    echo -e "${GREEN}~/RKLLAMA directory deleted successfully.${RESET}"
else
    echo -e "${RED}The ~/RKLLAMA directory does not exist, no deletion performed.${RESET}"
fi

# Check if the rkllama executable exists in /usr/local/bin/
if [ -f "/usr/local/bin/rkllama" ]; then
    echo -e "${YELLOW}Deleting the /usr/local/bin/rkllama executable...${RESET}"
    sudo rm /usr/local/bin/rkllama
    echo -e "${GREEN}/usr/local/bin/rkllama executable deleted successfully.${RESET}"
else
    echo -e "${RED}/usr/local/bin/rkllama executable does not exist, no deletion performed.${RESET}"
fi

# End of uninstallation message
echo -e "${GREEN}Uninstallation completed successfully.${RESET}"
