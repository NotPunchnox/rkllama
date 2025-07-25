# **Documentation de l'API REST - RKLLama**

## **Base URL**

```
http://localhost:8080/
```

---

## **Mémo rapide**
### Commandes principales :
- **Lister les modèles disponibles** : `GET /models`  
- **Charger un modèle** : `POST /load_model`  
- **Décharger le modèle** : `POST /unload_model`  
- **Obtenir le modèle chargé** : `GET /current_model`  
- **Générer une sortie** : `POST /generate`  
- **Télécharger un modèle depuis Hugging Face** : `POST /pull`  
- **Supprimer un modèle** : `POST /rm`

---

## **Endpoints de l'API**

### **1. GET /models**
#### **Description**
Retourne la liste des modèles disponibles dans le répertoire `~/RKLLAMA/models`.

#### **Requête**
```http
GET /models
```

#### **Réponse**
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

- **Code 500** : Répertoire introuvable.
  ```json
  {
    "error": "Le dossier ~/RKLLAMA/models est introuvable."
  }
  ```

#### **Exemple**
```bash
curl -X GET http://localhost:8080/models
```

---

### **2. POST /load_model**
#### **Description**
Charge un modèle spécifique dans la mémoire.

#### **Requête**
```http
POST /load_model
Content-Type: application/json
```

##### **Paramètres**
```json
{
  "model_name": "nom_du_modèle.rkllm"
}
```

#### **Réponse**
- **Code 200** : Modèle chargé avec succès.
  ```json
  {
    "message": "Modèle <model_name> chargé avec succès."
  }
  ```

- **Code 400** : Erreur liée à un modèle déjà chargé ou à des paramètres manquants.
  ```json
  {
    "error": "Un modèle est déjà chargé. Veuillez d'abord le décharger."
  }
  ```

- **Code 400** : Modèle introuvable.
  ```json
  {
    "error": "Modèle <model_name> introuvable dans le dossier /models."
  }
  ```

#### **Exemple**
```bash
curl -X POST http://localhost:8080/load_model \
-H "Content-Type: application/json" \
-d '{"model_name": "model1.rkllm"}'
```

---

### **3. POST /unload_model**
#### **Description**
Décharge le modèle actuellement chargé.

#### **Requête**
```http
POST /unload_model
```

#### **Réponse**
- **Code 200** : Succès.
  ```json
  {
    "message": "Modèle déchargé avec succès."
  }
  ```

- **Code 400** : Aucun modèle n'est chargé.
  ```json
  {
    "error": "Aucun modèle n'est actuellement chargé."
  }
  ```

#### **Exemple**
```bash
curl -X POST http://localhost:8080/unload_model
```

---

### **4. GET /current_model**
#### **Description**
Retourne le nom du modèle actuellement chargé.

#### **Requête**
```http
GET /current_model
```

#### **Réponse**
- **Code 200** : Succès.
  ```json
  {
    "model_name": "nom_du_modèle"
  }
  ```

- **Code 404** : Aucun modèle n'est chargé.
  ```json
  {
    "error": "Aucun modèle n'est actuellement chargé."
  }
  ```

#### **Exemple**
```bash
curl -X GET http://localhost:8080/current_model
```

---

### **5. POST /generate**
#### **Description**
Génère une réponse en utilisant le modèle chargé.

#### **Requête**
```http
POST /generate
Content-Type: application/json
```

##### **Paramètres**
```json
{
  "messages": "prompt ou chat_template",
  "stream": true
}
```

#### **Réponse**
- **Code 200** : Réponse générée.
  ```json
  {
    "id": "rkllm_chat",
    "object": "rkllm_chat",
    "created": null,
    "choices": [{
      "role": "assistant",
      "content": "sortie_rkllama",
      "finish_reason": "stop"
    }],
    "usage": {
      "prompt_tokens": null,
      "completion_tokens": null,
      "total_tokens": null
    }
  }
  ```

- **Code 400** : Aucun modèle n'est chargé.
  ```json
  {
    "error": "Aucun modèle n'est actuellement chargé."
  }
  ```

#### **Exemple**
```bash
curl -X POST http://localhost:8080/generate \
-H "Content-Type: application/json" \
-d '{"messages": "Bonjour, comment vas-tu ?", "stream": false}'
```

---

### **6. POST /pull**
#### **Description**
Télécharge et installe un modèle depuis Hugging Face.

#### **Requête**
```http
POST /pull
Content-Type: application/json
```

##### **Paramètres**
```json
{
  "model": "hf/nom_utilisateur/id_dépôt/fichier.rkllm"
}
```

#### **Réponse**
- **Code 200** : Téléchargement en cours.
```txt
Téléchargement <fichier> (<taille> MB)...
<progression>%
```

- **Code 400** : Erreur lors du téléchargement.
```txt
Erreur pendant le téléchargement : <erreur>
```

#### **Exemple**
```bash
curl -X POST http://localhost:8080/pull \
-H "Content-Type: application/json" \
-d '{"model": "hf/username/repo/file.rkllm"}'
```

---

### **7. DELETE /rm**
#### **Description**
Supprime un modèle spécifique.

#### **Requête**
```http
POST /rm
Content-Type: application/json
```

##### **Paramètres**
```json
{
  "model": "nom_du_modèle.rkllm"
}
```

#### **Réponse**
- **Code 200** : Succès.
  ```json
  {
    "message": "Le modèle a été supprimé avec succès."
  }
  ```

- **Code 404** : Modèle introuvable.
  ```json
  {
    "error": "Le modèle : {model} est introuvable."
  }
  ```

#### **Exemple**
```bash
curl -X DELETE http://localhost:8080/rm \
-H "Content-Type: application/json" \
-d '{"model": "model1.rkllm"}'
```

---

## **Appel d'Outils/Fonctions**

RKLLama prend en charge une fonctionnalité complète d'appel d'outils avec une compatibilité totale avec l'API Ollama. Les outils permettent aux modèles d'appeler des fonctions externes avec des paramètres structurés.

### **Endpoints Supportés**
- **`/api/chat`** - Support complet des appels d'outils  
- **`/api/generate`** - Détection d'outils dans le texte généré

### **Requête Basique d'Appel d'Outil**

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:3b",
    "messages": [
      {"role": "user", "content": "Quel temps fait-il à Paris ?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "obtenir_meteo_actuelle",
          "description": "Obtenir la météo actuelle dans un lieu donné",
          "parameters": {
            "type": "object",
            "properties": {
              "lieu": {
                "type": "string",
                "description": "La ville et le département, ex: Paris, Ile-de-France"
              },
              "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Unité de température à utiliser"
              }
            },
            "required": ["lieu"]
          }
        }
      }
    ]
  }'
```

### **Format de Réponse des Appels d'Outils**

Quand un modèle appelle un outil, la réponse inclut `tool_calls` :

```json
{
  "model": "qwen2.5:3b",
  "created_at": "2024-12-21T10:30:00.000Z",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "obtenir_meteo_actuelle",
          "arguments": {
            "lieu": "Paris, FR",
            "format": "celsius"
          }
        }
      }
    ]
  },
  "done_reason": "tool_calls",
  "done": true
}
```

### **Compatibilité des Modèles**

| Type de Modèle | Format | Niveau de Support |
|-----------------|--------|-------------------|
| **Qwen 2.5+** | Balises `<tool_call></tool_call>` | ✅ Natif |
| **Llama 3.2+** | Détection JSON générique | ✅ Complet |
| **Autres LLMs** | Analyse JSON de fallback | ✅ Compatible |

### **Fonctionnalités**

- **Méthodes de détection multiples** pour différents formats de modèles
- **Support du streaming** - les appels d'outils fonctionnent en mode streaming et non-streaming
- **Normalisation des formats** - gère automatiquement `parameters` vs `arguments`
- **Analyse JSON robuste** avec méthodes de fallback
- **Appels d'outils multiples** dans une seule réponse

Pour la documentation complète, des exemples et le dépannage, voir le [Guide d'Appel d'Outils](./tools.md).

---

### **8. GET /**
#### **Description**
Affiche un message de bienvenue et un lien vers le projet GitHub.

#### **Réponse**
- **Code 200** :
  ```json
  {
    "message": "Bienvenue sur RK-LLama !",
    "github": "https://github.com/notpunhnox/rkllama"
  }
  ```

#### **Exemple**
```bash
curl -X GET http://localhost:8080/
```

---

## **Gestion des erreurs**
- **400** : Erreur liée à une mauvaise requête.  
- **404** : Ressource introuvable.  
- **500** : Erreur interne au serveur.

---

## **Conseils pratiques**
- **Validation des paramètres** : Vérifiez toujours les noms des modèles et les chemins des fichiers.  
- **Dépannage** : Consultez les journaux du serveur pour plus de détails sur les erreurs internes.