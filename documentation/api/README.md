# Documentation de l'API REST - RKLLama

## Base URL

```
http://localhost:8080/
```

## Endpoints de l'API

### 1. **GET /models**

#### Description
Retourne la liste des modèles disponibles dans le répertoire `~/RKLLAMA/models`.

#### Requête

```http
GET /models
```

#### Réponse
- **Code 200** : Liste des modèles disponibles.
  ```json
  {
    "models": [
      "model1.rkllm",
      "model2.rkllm",
      "model3.rkllm"
    ]
  }
  ```

- **Code 500** : Si le répertoire `~/RKLLAMA/models` est introuvable.
  ```json
  {
    "error": "Le dossier ~/RKLLAMA/models est introuvable."
  }
  ```

---

### 2. **POST /load_model**

#### Description
Charge un modèle spécifique dans la mémoire. Le nom du modèle est fourni dans le corps de la requête.

#### Requête

```http
POST /load_model
Content-Type: application/json
```

#### Paramètres de la requête
```json
{
  "model_name": "nom_du_modèle.rkllm"
}
```

#### Réponse
- **Code 200** : Modèle chargé avec succès.
  ```json
  {
    "message": "Modèle <model_name> chargé avec succès."
  }
  ```

- **Code 400** : Si un modèle est déjà chargé ou si un paramètre est manquant.
  ```json
  {
    "error": "Un modèle est déjà chargé. Veuillez d'abord le décharger."
  }
  ```

- **Code 400** : Si le modèle spécifié est introuvable.
  ```json
  {
    "error": "Modèle <model_name> introuvable dans le dossier /models."
  }
  ```

---

### 3. **POST /unload_model**

#### Description
Décharge le modèle actuellement chargé.

#### Requête

```http
POST /unload_model
```

#### Réponse
- **Code 200** : Modèle déchargé avec succès.
  ```json
  {
    "message": "Modèle déchargé avec succès."
  }
  ```

- **Code 400** : Si aucun modèle n'est actuellement chargé.
  ```json
  {
    "error": "Aucun modèle n'est actuellement chargé."
  }
  ```

---

### 4. **GET /current_model**

#### Description
Retourne le nom du modèle actuellement chargé.

#### Requête

```http
GET /current_model
```

#### Réponse
- **Code 200** : Modèle actuellement chargé.
  ```json
  {
    "model_name": "nom_du_modèle"
  }
  ```

- **Code 404** : Si aucun modèle n'est actuellement chargé.
  ```json
  {
    "error": "Aucun modèle n'est actuellement chargé."
  }
  ```

---

### 5. **POST /generate**

#### Description
Envoie une requête pour générer une sortie en utilisant le modèle actuellement chargé. Le corps de la requête peut inclure des données nécessaires au traitement par le modèle.

#### Requête

```http
POST /generate
Content-Type: application/json
```

#### Paramètres de la requête
```js
{
  "messages": "prompt ou chat_template",
  "stream"  : Boolean(true || false)
}
```

#### Réponse
- **Code 200** : Sortie générée par le modèle.
  ```json
  {
        "id": "rkllm_chat",
        "object": "rkllm_chat",
        "created": None,
        "choices": [{
            "role": "assistant",
            "content": "sortie_rkllama",
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        }
    }
  ```

- **Code 400** : Si aucun modèle n'est actuellement chargé.
  ```json
  {
    "error": "Aucun modèle n'est actuellement chargé."
  }
  ```

---

### 6. **GET /**

#### Description
Route par défaut qui affiche un message de bienvenue et un lien vers le projet GitHub.

#### Requête

```http
GET /
```

#### Réponse
- **Code 200** : Message de bienvenue et lien vers GitHub.
  ```json
  {
    "message": "Welcome to RK-LLama !",
    "github": "https://github.com/notpunhnox/rk-llama"
  }
  ```

---

## Gestion des erreurs

Les erreurs de l'API suivent les codes HTTP standards avec des messages détaillés dans le corps de la réponse :

- **400 (Bad Request)** : Erreur liée à une mauvaise requête (paramètres manquants, modèle déjà chargé, etc.).
- **404 (Not Found)** : Le modèle demandé n'existe pas ou aucun modèle n'est chargé.
- **500 (Internal Server Error)** : Erreur serveur (répertoire introuvable, problème interne).

Exemples d'erreurs possibles :

- **400** : 
  ```json
  {
    "error": "Veuillez fournir le nom du modèle à charger."
  }
  ```

- **404** : 
  ```json
  {
    "error": "Aucun modèle n'est actuellement chargé."
  }
  ```

- **500** : 
  ```json
  {
    "error": "Le dossier ~/RKLLAMA/models est introuvable."
  }
  ```