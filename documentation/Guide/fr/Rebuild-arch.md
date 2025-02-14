# Nouvelle Architecture - Version 0.0.3

## Mise à jour de l'architecture pour les modèles installés avant la version 0.0.3

Si vous avez des modèles installés avec une version antérieure à la 0.0.3, voici les étapes pour adapter votre installation à la nouvelle architecture.

---

## Nouvelle Organisation de l'Arborescence

La nouvelle structure de répertoires se présente comme suit :

```
~/RKLLAMA
    └── models
        |
        |── DeepSeek-v3-7b
        |   |── Modelfile
        |   └── deepseek.rkllm
        |
        └── Llama3-7b
            |── Modelfile
            └── deepseek.rkllm
```

### Ancienne Architecture (Version 0.0.1)

Avant la mise à jour, l'organisation était la suivante :

```
~/RKLLAMA
    └── models
        |── llama3-7b.rkllm
        └── qwen2.5-3b.rkllm
```

---

## Réorganisation Automatique

Pour remettre en place la nouvelle architecture, il vous suffit d'exécuter la commande suivante :

```bash
rkllama list
```

Cette commande va :
- Créer un dossier dédié pour chaque modèle.
- Déplacer le fichier `.rkllm` correspondant dans le dossier du modèle.

---

## Gestion des Erreurs

Si vous rencontrez l'erreur suivante lors du lancement d'un modèle :

```
- Modelfile not found in 'model_name' directory.
```

Vous devrez alors lancer le modèle avec la commande suivante :

```bash
rkllama run modelname file.rkllm huggingface_repo
```

### Remarques :
- **huggingface_repo** : Il est nécessaire de fournir un lien vers un dépôt HuggingFace afin de récupérer le tokenizer et le chattemplate. Vous pouvez utiliser un dépôt différent de celui du modèle, à condition que le tokenizer corresponde et que le chattemplate convienne à vos besoins.
- Pour la version quantisée du modèle `Qwen2.5-3B`, vous pouvez utiliser le dépôt officiel comme **huggingface_repo** (exemple : [https://huggingface.co/Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)).

---

## Exemple de Commande

Pour un modèle tel que [deepseek-llm-7b-chat-rk3588-1.1.1](https://huggingface.co/c01zaut/deepseek-llm-7b-chat-rk3588-1.1.1), la commande pourrait ressembler à :

```bash
rkllama run deepseek-llm-7b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.5 deepseek-llm-7b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm c01zaut/deepseek-llm-7b-chat-rk3588-1.1.1
```

Les logs du serveur afficheront :

```bash
FROM: deepseek-llm-7b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm
HuggingFace Path: c01zaut/deepseek-llm-7b-chat-rk3588-1.1.1
```

Le **Modelfile** sera initialisé avec les valeurs suivantes :

```env
FROM="deepseek-llm-7b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm"
HUGGINGFACE_PATH="c01zaut/deepseek-llm-7b-chat-rk3588-1.1.4"
SYSTEM=""
TEMPERATURE=1.0
```

---

## En Résumé

- **Nouvelle organisation** : Chaque modèle dispose désormais de son propre dossier avec un Modelfile et le fichier `.rkllm` à l'intérieur.
- **Mise à jour automatique** : La commande `rkllama list` recrée l'arborescence pour les modèles existants.
- **Création du Modelfile** : Si le Modelfile n'est pas présent, utilisez la commande `rkllama run modelname file.rkllm huggingface_repo` pour le générer (cette opération est nécessaire une seule fois pour chaque modèle mis à jour).
- **Intégration avec HuggingFace** : Le chemin HuggingFace permet d'initialiser automatiquement le tokenizer et le chattemplate.
  
Prenez en compte ces informations si vous avez des modèles installés avant la version 0.0.3.