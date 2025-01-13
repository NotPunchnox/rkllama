#!/bin/bash

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

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
    # Back up the models
    echo -e "${GREEN}The models will be saved to /home/$(whoami)/Desktop/rkllm_models.${RESET}"
    mkdir -p "/home/$(whoami)/Desktop/rkllm_models"
    cp ~/RKLLAMA/models/*.rkllm "/home/$(whoami)/Desktop/rkllm_models"

    echo -e "${GREEN}Models have been successfully backed up.${RESET}"
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
