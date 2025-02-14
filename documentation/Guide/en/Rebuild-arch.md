# New Architecture - Version 0.0.3

## Updating the Architecture for Models Installed Before Version 0.0.3

If you have models installed with a version older than 0.0.3, follow these steps to adapt your installation to the new architecture.

---

## New Directory Structure

The new directory structure is as follows:

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

### Previous Architecture (Version 0.0.1)

Before the update, the structure was as follows:

```
~/RKLLAMA
    └── models
        |── llama3-7b.rkllm
        └── qwen2.5-3b.rkllm
```

---

## Automatic Reorganization

To set up the new architecture, simply run the following command:

```bash
rkllama list
```

This command will:
- Create a dedicated folder for each model.
- Move the corresponding `.rkllm` file into the model's folder.

---

## Error Handling

If you encounter the following error when launching a model:

```
- Modelfile not found in 'model_name' directory.
```

You will need to launch the model using the following command:

```bash
rkllama run modelname file.rkllm huggingface_repo
```

### Notes:
- **huggingface_repo**: You must provide a link to a HuggingFace repository to retrieve the tokenizer and chattemplate. You can use a repository different from that of the model, as long as the tokenizer is compatible and the chattemplate meets your needs.
- For the quantized version of the `Qwen2.5-3B` model, you can use the official repository as the **huggingface_repo** (example: [https://huggingface.co/Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)).

---

## Example Command

For a model such as [deepseek-llm-7b-chat-rk3588-1.1.1](https://huggingface.co/c01zaut/deepseek-llm-7b-chat-rk3588-1.1.1), the command might look like this:

```bash
rkllama run deepseek-llm-7b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.5 deepseek-llm-7b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm c01zaut/deepseek-llm-7b-chat-rk3588-1.1.1
```

The server logs will display:

```bash
FROM: deepseek-llm-7b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm
HuggingFace Path: c01zaut/deepseek-llm-7b-chat-rk3588-1.1.1
```

The **Modelfile** will be initialized with the following values:

```env
FROM="deepseek-llm-7b-chat-rk3588-w8a8-opt-0-hybrid-ratio-0.5.rkllm"
HUGGINGFACE_PATH="c01zaut/deepseek-llm-7b-chat-rk3588-1.1.4"
SYSTEM=""
TEMPERATURE=1.0
```

---

## In Summary

- **New Organization**: Each model now has its own folder containing a Modelfile and the `.rkllm` file.
- **Automatic Update**: The `rkllama list` command reorganizes the directory structure for existing models.
- **Modelfile Creation**: If the Modelfile is missing, use the command `rkllama run modelname file.rkllm huggingface_repo` to generate it (this step is required only once for each updated model).
- **HuggingFace Integration**: The HuggingFace path automatically initializes the tokenizer and chattemplate.

Please take note of these changes if you have models installed prior to version 0.0.3.