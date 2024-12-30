## LLM server & client for Rockchip 3588/rk3576

Déjà fonctionnel mais en cours de développement...

Mise à jours du repo 31 décembre 2024

./models = dossier contenant vos modèles llm
./lib    = lib C rkllm compilé

./app.py = serveur rkllm
./client.py = client permettant de communiquer avec le serveur


lib C utilisé par défaut: 1.1.4

versions python supportées:
    - Python 3.8
    - Python 3.9