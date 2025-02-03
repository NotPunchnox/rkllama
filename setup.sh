#!/bin/bash

# Definition of colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Checking for Git repository updates
#echo -e "${CYAN}Checking for updates...${RESET}"
#git pull
#echo -e "${GREEN}Update check completed successfully!${RESET}"

# Miniconda installation path
MINICONDA_DIR=~/miniconda3
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"

# Check if Miniconda is installed
if [ -d "$MINICONDA_DIR" ]; then
    echo -e "${GREEN}Miniconda is already installed.${RESET}"
else
    echo -e "${YELLOW}Miniconda is not installed. Proceeding with installation...${RESET}"
    # Download and install Miniconda
    wget "$MINICONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm /tmp/miniconda.sh
    echo -e "${GREEN}Miniconda was successfully installed.${RESET}"
fi

# Creating the RKLLAMA directory
echo -e "${CYAN}Creating the ~/RKLLAMA directory...${RESET}"
mkdir -p ~/RKLLAMA/
echo -e "${CYAN}Installing resources in ~/RKLLAMA...${RESET}"
cp -rf . ~/RKLLAMA/

# Activating Miniconda
source "$MINICONDA_DIR/bin/activate"

# Installing dependencies
echo -e "${CYAN}Installing dependencies from requirements.txt...${RESET}"
pip3 install -r ~/RKLLAMA/requirements.txt

# Making client.sh and server.sh executable
echo -e "${CYAN}Making ./client.sh and ./server.sh executable${RESET}"
chmod +x ~/RKLLAMA/client.sh
chmod +x ~/RKLLAMA/server.sh
chmod +x ~/RKLLAMA/uninstall.sh

# Exporting client.sh as a global command
echo -e "${CYAN}Creating a global executable for rkllama...${RESET}"
sudo cp ~/RKLLAMA/client.sh /usr/local/bin/rkllama
sudo chmod +x /usr/local/bin/rkllama
echo -e "${CYAN}Executable created successfully: /usr/local/bin/rkllama${RESET}"

# Displaying statuses and available commands
echo -e "${GREEN}+ Configuration: OK.${RESET}"
echo -e "${GREEN}+ Installation : OK.${RESET}"

echo -e "${BLUE}Server${GREEN}  : ./server.sh${RESET}"
echo -e "${BLUE}Client${GREEN}   : ./client.sh${RESET}\n"
echo -e "${BLUE}Global command : ${RESET}rkllama"
