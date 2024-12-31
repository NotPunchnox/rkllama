# RK-LLama: Serveur et Client LLM pour Rockchip 3588/3576

## Présentation
Un serveur permettant d'exécuter et d'interagir avec des modèles LLM optimisé pour les platformes Rockchip RK3588(S) et RK3576.
La différence avec les autres logiciels de ce type comme: [Ollama](https://ollama.com) ou [Llama.cpp](https://github.com/ggerganov/llama.cpp), c'est que RK-LLama permet de lancer les modèles sur NPU.

Testé sur un Orange pi 5 Pro ( 16Go RAM ).

## Structure des fichiers
- **`./models`** : Placez vos modèles.rkllm ici.
- **`./lib`** : Bibliothèque C++ `rkllm` utilisée pour l'inférence et `fix_freqence_platform`.
- **`./app.py`** : Serveur API Rest.
- **`./client.py`** : Client pour interagir avec le serveur.

## Version de Python supportés:
- Python 3.8
- Python 3.9

## Matériel et environnement testés
- **Matériel** : Rockchip RK3588S, NPU 6 TOPS.
- **OS** : Ubuntu 24.04 arm64.

## Fonctionnalités principales
- **Exécution des modèles sur NPU.**
- **Liste des modèles disponibles.**
- **Chargement et déchargement dynamique des modèles.**
- **Requêtes d'inférence.**
- **Mode streaming et non-streaming.**
- **Historique de messages.**

## Instructions
1. Placez vos modèles (fichiers.rkllm) dans `./models`.
2. Installer les dépendances nécessaires :
```
chmod +x setup.sh
sudo ./setup.sh
```

