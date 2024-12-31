#!/bin/bash

# Définition des couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Chemin d'installation de Miniconda
MINICONDA_DIR=~/miniconda3
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# Vérifie si Miniconda est installé
if [ -d "$MINICONDA_DIR" ]; then
    echo -e "${GREEN}Miniconda est déjà installé.${RESET}"
else
    echo -e "${YELLOW}Miniconda n'est pas installé. Installation en cours...${RESET}"
    # Téléchargement et installation de Miniconda
    wget "$MINICONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm /tmp/miniconda.sh
    echo -e "${GREEN}Miniconda a été installé avec succès.${RESET}"
fi

# Activation de Miniconda
source "$MINICONDA_DIR/bin/activate"

# Installation des dépendances
echo -e "${CYAN}Installation des dépendances depuis requirements.txt...${RESET}"
pip3 install -r requirements.txt

# Rendre client.sh et server.sh exécutable
echo -e "${CYAN}Rendre ./client.sh et ./server.sh exécutable${RESET}"
chmod +x ./server.sh
chmod +x ./client.sh
echo -e "${GREEN}+ server.sh & client.sh : OK"

#clear
echo -e "${GREEN}+ Configuration         : OK.${RESET}"
echo -e "${GREEN}+ Installation          : OK.${RESET}"

echo -e "${BLUE}Pour lancer le serveur faites ${GREEN}./server.sh${RESET}"
echo -e "${BLUE}Pour lancer le client faites ${GREEN}./client.sh${RESET}"