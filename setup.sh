#!/bin/bash

# Définition des couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Vérification de mise à jour du dépôt Git
echo -e "${CYAN}Vérification de mise à jour...${RESET}"
git pull
echo -e "${GREEN}Vérification des mises à jour terminée avec succès!${RESET}"

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

# Création du répertoire RKLLAMA
echo -e "${CYAN}Création du répertoire ~/RKLLAMA...${RESET}"
mkdir -p ~/RKLLAMA/
echo -e "${CYAN}Installation des ressources dans ~/RKLLAMA...${RESET}"
cp -rf . ~/RKLLAMA/

# Activation de Miniconda
source "$MINICONDA_DIR/bin/activate"

# Installation des dépendances
echo -e "${CYAN}Installation des dépendances depuis requirements.txt...${RESET}"
pip3 install -r ~/RKLLAMA/requirements.txt

# Rendre client.sh et server.sh exécutable
echo -e "${CYAN}Rendre ./client.sh et ./server.sh exécutable${RESET}"
chmod +x ~/RKLLAMA/client.sh
chmod +x ~/RKLLAMA/server.sh
chmod +x ~/RKLLAMA/uninstall.sh

# Exporter client.sh en tant que commande globale
echo -e "${CYAN}Création d'un exécutable global pour rkllama...${RESET}"
sudo cp ~/RKLLAMA/client.sh /usr/local/bin/rkllama
sudo chmod +x /usr/local/bin/rkllama
echo -e "${CYAN}Exécutable créé avec succès: /usr/local/bin/rkllama${RESET}"

# Affichage des statuts et des commandes désormais disponibles
echo -e "${GREEN}+ Configuration : OK.${RESET}"
echo -e "${GREEN}+ Installation  : OK.${RESET}"

echo -e "${BLUE}Serveur${GREEN}  : ./server.sh${RESET}"
echo -e "${BLUE}Client${GREEN}   : ./client.sh${RESET}\n"
echo -e "${BLUE}Commande globale  : ${RESET}rkllama"
