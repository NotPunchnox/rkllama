#!/bin/bash

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Check for the argument to disable Miniconda
USE_CONDA=true
CONDA_ARG=""  # This will hold "--no-conda" if conda is disabled
if [[ "$1" == "--no-conda" ]]; then
    USE_CONDA=false
    CONDA_ARG="--no-conda"
    echo -e "${YELLOW}Miniconda is disabled for this installation.${RESET}"
fi

# Miniconda installation path
MINICONDA_DIR=~/miniconda3
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"

# Install Miniconda (if enabled)
if $USE_CONDA; then
    if [ -d "$MINICONDA_DIR" ]; then
        echo -e "${GREEN}Miniconda is already installed.${RESET}"
    else
        echo -e "${YELLOW}Miniconda is not installed. Proceeding with installation...${RESET}"
        wget "$MINICONDA_URL" -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
        rm /tmp/miniconda.sh
        echo -e "${GREEN}Miniconda was successfully installed.${RESET}"
    fi
fi

# Check for Git repository updates
echo -e "${CYAN}Checking for updates...${RESET}"
git pull
echo -e "${GREEN}Update check completed successfully!${RESET}"

# Create the RKLLAMA directory
echo -e "${CYAN}Creating the ~/RKLLAMA directory...${RESET}"
mkdir -p ~/RKLLAMA/
echo -e "${CYAN}Installing resources in ~/RKLLAMA...${RESET}"
cp -rf . ~/RKLLAMA/

# Activate Miniconda and install dependencies (if enabled)
if $USE_CONDA; then
    source "$MINICONDA_DIR/bin/activate"
fi

# Install dependencies using pip
echo -e "${CYAN}Installing dependencies from requirements.txt...${RESET}"
pip3 install -r ~/RKLLAMA/requirements.txt

# Make client.sh and server.sh executable
echo -e "${CYAN}Making ./client.sh and ./server.sh executable${RESET}"
chmod +x ~/RKLLAMA/client.sh
chmod +x ~/RKLLAMA/server.sh
chmod +x ~/RKLLAMA/uninstall.sh

# Modify client.sh and server.sh to always use --no-conda if conda is disabled
if ! $USE_CONDA; then
    echo -e "${CYAN}Ensuring client.sh and server.sh always run with --no-conda${RESET}"
    
    # Add --no-conda to client.sh if not already present
    if ! grep -q -- "--no-conda" ~/RKLLAMA/client.sh; then
        sed -i 's|#!/bin/bash|#!/bin/bash\nexec "$0" --no-conda "$@"|' ~/RKLLAMA/client.sh
    fi

    # Add --no-conda to server.sh if not already present
    if ! grep -q -- "--no-conda" ~/RKLLAMA/server.sh; then
        sed -i 's|#!/bin/bash|#!/bin/bash\nexec "$0" --no-conda "$@"|' ~/RKLLAMA/server.sh
    fi
fi

# Create a global executable for rkllama
echo -e "${CYAN}Creating a global executable for rkllama...${RESET}"
echo -e "#!/bin/bash\nexec ~/RKLLAMA/client.sh $CONDA_ARG \"\$@\"" | sudo tee /usr/local/bin/rkllama > /dev/null
sudo chmod +x /usr/local/bin/rkllama
echo -e "${CYAN}Executable created successfully: /usr/local/bin/rkllama${RESET}"

# Display statuses and available commands
echo -e "${GREEN}+ Configuration: OK.${RESET}"
echo -e "${GREEN}+ Installation : OK.${RESET}"

echo -e "${BLUE}Server${GREEN}  : ./server.sh $CONDA_ARG${RESET}"
echo -e "${BLUE}Client${GREEN}  : ./client.sh $CONDA_ARG${RESET}\n"
echo -e "${BLUE}Global command  : ${RESET}rkllama"