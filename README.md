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


Testé sur `Orange pi 5 pro (16Go RAM)`:
    - `OS`: Ubuntu 24.04 arm64
    - `Chip`: Rockchip 3588S
    - `Processeur`: 8-core 64-bit processor /big.LITTLE Architecture: 4-core Cortex-A76 and 4-core Cortex-A55, big core cluster is 2.4GHz, and little core cluster is 1.8GHz frequency
    - `NPU`: Embedded NPU supports INT4/INT8/INT16 mixed operation, with up to 6TOPS computing power