#!/bin/bash

# Vérification de mise à jours
echo "${CYAN}Vérification de mise à jours...${RESET}"
git pull
echo "${GREEN}Vérification des mises à jours terminé avec succès!${RESET}"

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

echo "${CYAN}Création du répertoire ~/RKLLAMA...${RESET}"
mkdir ~/RKLLAMA/
echo "${CYAN}Installation de répertoire dans le dossier ~/RKLLAMA...${RESET}"
cp . ~/RKLLAMA/

# Activation de Miniconda
source "$MINICONDA_DIR/bin/activate"

# Installation des dépendances
echo -e "${CYAN}Installation des dépendances depuis requirements.txt...${RESET}"
pip3 install -r requirements.txt

# Rendre client.sh et server.sh exécutable
echo -e "${CYAN}Rendre ./client.sh et ./server.sh exécutable${RESET}"
chmod +x ./server.sh
chmod +x ./client.sh

# Exporter client
echo -e "${CYAN}Création d'un exécutable global pour rkllama..${RESET}"
sudo cp ./client.sh /usr/local/bin/rkllama
sudo chmod +x /usr/local/bin/rkllama
echo -e "${CYAN}Exécutable créé avec succès: /usr/local/bin/client${RESET}"

#Afficher les status et commandes désormais disponibles
echo -e "${GREEN}+ Configuration : OK.${RESET}"
echo -e "${GREEN}+ Installation  : OK.${RESET}"

echo -e "${BLUE}Serveur${GREEN}  : ./server.sh${RESET}"
echo -e "${BLUE}Client${GREEN}   : ./client.sh${RESET}\n"

echo -e "${BLUE}Commande global  : ${RESET}rkllama"