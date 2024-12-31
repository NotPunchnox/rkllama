#!/bin/bash

# Chemin d'installation de Miniconda
MINICONDA_DIR=~/miniconda3

# Vérifie si Miniconda est installé
if [ -d "$MINICONDA_DIR" ]; then
    echo -e "${GREEN}Lancement de l'environnement avec Miniconda3.${RESET}"
    source "$MINICONDA_DIR/bin/activate"
else
    echo -e "${YELLOW}Lancement du fichier d'installation...${RESET}"
    # Téléchargement et installation de Miniconda
    bash ./setup.sh
fi

# Extraction du modèle du processeur
cpu_model=$(cat /proc/cpuinfo | grep "cpu model" | head -n 1 | sed 's/.*Rockchip \(RK[0-9]*\).*/\1/' | tr '[:upper:]' '[:lower:]')

# Lancement du serveur avec le modèle de processeur en paramètre
python3 ~/RKLLAMA/server.py --target_platform "$cpu_model"