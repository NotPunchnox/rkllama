# LLM Server & Client for Rockchip 3588/rk3576

## Présentation
Un serveur et un client fonctionnels permettant d'exécuter des modèles LLM optimisés pour les plateformes Rockchip RK3588 et RK3576.

Mise à jour du dépôt : **31 décembre 2024**

Ce projet est en cours de développement, mais il est déjà utilisable. Il offre des capacités de déploiement et d'interaction avec des modèles d'IA stockés localement.

## Structure des fichiers
- **`./models`** : Contient les modèles LLM disponibles pour le serveur. Placez vos modèles dans ce dossier.
- **`./lib`** : Contient la bibliothèque C++ compilée `rkllm` utilisée pour l'inférence.
- **`./app.py`** : Le serveur Flask permettant de charger, décharger, et interagir avec les modèles.
- **`./client.py`** : Client pour communiquer avec le serveur via des requêtes HTTP.

## Détails de la bibliothèque utilisée
- **Lib C utilisée par défaut** : Version `1.1.4`

## Versions Python supportées
- Python 3.8
- Python 3.9

## Configuration matérielle testée
Testé sur **Orange Pi 5 Pro (16Go RAM)** :
- **OS** : Ubuntu 24.04 arm64
- **Chip** : Rockchip 3588S
- **Processeur** :
  - 8-core 64-bit big.LITTLE architecture
  - 4-core Cortex-A76 (2.4GHz) + 4-core Cortex-A55 (1.8GHz)
- **NPU** : Embedded NPU supportant des opérations INT4/INT8/INT16 avec une puissance de calcul allant jusqu'à 6 TOPS.

## Fonctionnalités principales

### Endpoints du serveur Flask

1. **Lister les modèles disponibles**
   - **Route** : `GET /models`
   - **Description** : Retourne la liste des modèles disponibles dans le dossier `./models`.
   - **Exemple de réponse** :
     ```json
     {
       "models": ["model1.rkllm", "model2.rkllm"]
     }
     ```

2. **Charger un modèle**
   - **Route** : `POST /load_model`
   - **Description** : Charge un modèle spécifique en mémoire pour l'utiliser.
   - **Requête** :
     ```json
     {
       "model_name": "nom_du_modele"
     }
     ```
   - **Exemple de réponse (succès)** :
     ```json
     {
       "message": "Modèle nom_du_modele chargé avec succès."
     }
     ```
   - **Exemple de réponse (erreur)** :
     ```json
     {
       "error": "Un modèle est déjà chargé. Veuillez d'abord le décharger."
     }
     ```

3. **Décharger un modèle**
   - **Route** : `POST /unload_model`
   - **Description** : Libère les ressources en déchargeant le modèle chargé.
   - **Exemple de réponse** :
     ```json
     {
       "message": "Modèle déchargé avec succès."
     }
     ```

4. **Envoyer une requête au modèle**
   - **Route** : `POST /rkllm_chat`
   - **Description** : Envoie une requête d'inférence au modèle actuellement chargé.
   - **Exemple de réponse (erreur)** :
     ```json
     {
       "error": "Aucun modèle n'est actuellement chargé."
     }
     ```

## Instructions d'utilisation

1. Placez vos modèles dans le dossier `./models`.
2. Démarrez le serveur avec :
   ```bash
   python3 app.py
   ```
3. Utilisez un client HTTP pour interagir avec le serveur via les routes mentionnées ci-dessus. Vous pouvez également utiliser le fichier `client.py` pour simplifier cette interaction.

## Notes supplémentaires
- Avant de charger un nouveau modèle, assurez-vous de décharger le modèle actuellement chargé.
- Le dossier `./models` doit contenir uniquement des modèles compatibles avec la bibliothèque C++ `rkllm`.

---

### Développement futur
Des améliorations continues sont prévues pour améliorer l'efficacité et la convivialité du serveur, notamment des fonctionnalités supplémentaires pour la gestion des modèles et l'optimisation des performances sur d'autres matériels.

