#!/bin/bash

# Définition des couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Demande de confirmation
echo -e "${YELLOW}Êtes-vous sûr de vouloir désinstaller RKLLama ? Cette action supprimera tous les fichiers associés.${RESET}"
read -p "Tapez 'y' pour continuer, 'n' pour annuler: " confirmation

if [[ "$confirmation" != "y" && "$confirmation" != "Y" ]]; then
    echo -e "${RED}Désinstallation annulée.${RESET}"
    exit 1
fi

# Message de début de désinstallation
echo -e "${CYAN}Démarrage de la désinstallation...${RESET}"

# Vérification si le répertoire RKLLM existe
if [ -d "$HOME/RKLLM" ]; then
    echo -e "${YELLOW}Suppression du répertoire ~/RKLLM/...${RESET}"
    rm -rf ~/RKLLM/
    echo -e "${GREEN}Répertoire ~/RKLLM/ supprimé avec succès.${RESET}"
else
    echo -e "${RED}Le répertoire ~/RKLLM/ n'existe pas, aucune suppression effectuée.${RESET}"
fi

# Vérification si l'exécutable rkllama existe dans /usr/local/bin/
if [ -f "/usr/local/bin/rkllama" ]; then
    echo -e "${YELLOW}Suppression de l'exécutable /usr/local/bin/rkllama...${RESET}"
    sudo rm /usr/local/bin/rkllama
    echo -e "${GREEN}Exécutable /usr/local/bin/rkllama supprimé avec succès.${RESET}"
else
    echo -e "${RED}L'exécutable /usr/local/bin/rkllama n'existe pas, aucune suppression effectuée.${RESET}"
fi

# Message de fin de désinstallation
echo -e "${GREEN}Désinstallation terminée avec succès.${RESET}"
