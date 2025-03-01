import threading, time, json
from transformers import AutoTokenizer
from flask import Flask, request, jsonify, Response
import src.variables as variables
import datetime


def Request(modele_rkllm, custom_request=None):
    """
    Process a request to the language model
    
    Args:
        modele_rkllm: The language model instance
        custom_request: Optional custom request object that mimics Flask request
    
    Returns:
        Flask response with generated text
    """
    try:
        # Mettre le serveur en état de blocage.
        isLocked = True

        # Use custom_request if provided, otherwise use Flask's request
        req = custom_request if custom_request is not None else request
        data = req.json
        
        if data and 'messages' in data:
            # Réinitialiser les variables globales.
            variables.global_status = -1

            # Définir la structure de la réponse renvoyée.
            llmResponse = {
                "id": "rkllm_chat",
                "object": "rkllm_chat",
                "created": int(time.time()),
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tokens_per_second": 0,
                    "total_tokens": 0
                }
            }

            # Check if this is an Ollama-style request
            is_ollama_request = req.path.startswith('/api/')
            
            # Récupérer l'historique du chat depuis la requête JSON
            messages = data["messages"]

            # Mise en place du tokenizer
            tokenizer = AutoTokenizer.from_pretrained(variables.model_id, trust_remote_code=True)
            supports_system_role = "raise_exception('System role not supported')" not in tokenizer.chat_template

            if variables.system and supports_system_role:
                prompt = [{"role": "system", "content": variables.system}] + messages
            else:
                prompt = messages

            for i in range(1, len(prompt)):
                if prompt[i]["role"] == prompt[i - 1]["role"]:
                    raise ValueError("Les rôles doivent alterner entre 'user' et 'assistant'.")

            # Mise en place du chat Template
            prompt = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            llmResponse["usage"]["prompt_tokens"] = llmResponse["usage"]["total_tokens"] = len(prompt)

            sortie_rkllm = ""

            if not "stream" in data.keys() or data["stream"] == False:
                # Créer un thread pour l'inférence du modèle.
                thread_modele = threading.Thread(target=modele_rkllm.run, args=(prompt,))
                try:
                    thread_modele.start()
                    print("Thread d'inférence démarré")
                except Exception as e:
                    print("Erreur lors du démarrage du thread:", e)


                # Attendre la fin du modèle et vérifier périodiquement le thread d'inférence.
                threadFinish = False
                count = 0
                start = time.time()

                while not threadFinish:
                    while len(variables.global_text) > 0:
                        count += 1
                        sortie_rkllm += variables.global_text.pop(0)
                        time.sleep(0.005)

                        thread_modele.join(timeout=0.005)
                    threadFinish = not thread_modele.is_alive()

                total = time.time() - start
                llmResponse["choices"] = [{
                    "role": "assistant",
                    "content": sortie_rkllm,
                    "logprobs": None,
                    "finish_reason": "stop"
                }]
                llmResponse["usage"]["total_tokens"] = count + llmResponse["usage"]["prompt_tokens"]
                llmResponse["usage"]["completion_tokens"] = count
                llmResponse["usage"]["tokens_per_second"] = count / total

                if is_ollama_request:
                    # Transform to Ollama format
                    ollama_response = {
                        "model": variables.model_id,
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "message": {
                            "role": "assistant",
                            "content": sortie_rkllm
                        },
                        "done": True,
                        "total_duration": total * 1000000000,  # Convert to nanoseconds
                        "prompt_eval_count": llmResponse["usage"]["prompt_tokens"],
                        "prompt_eval_duration": 0,  # Not available
                        "eval_count": count,
                        "eval_duration": total * 1000000000  # Convert to nanoseconds
                    }
                    return jsonify(ollama_response), 200
                else:
                    return jsonify(llmResponse), 200

            else:
                def generate():
                    thread_modele = threading.Thread(target=modele_rkllm.run, args=(prompt,))
                    thread_modele.start()

                    thread_modele_terminé = False
                    count = 0
                    start = time.time()

                    while not thread_modele_terminé:
                        while len(variables.global_text) > 0:
                            count += 1
                            sortie_rkllm = variables.global_text.pop(0)

                            llmResponse["choices"] = [
                                {
                                "role": "assistant",
                                "content": sortie_rkllm,
                                "logprobs": None,
                                "finish_reason": "stop" if variables.global_status == 1 else None,
                                }
                            ]
                            llmResponse["usage"]["completion_tokens"] = count
                            llmResponse["usage"]["total_tokens"] += 1
                            
                            if is_ollama_request:
                                # Transform to Ollama format for streaming
                                current_time = time.time() - start
                                ollama_chunk = {
                                    "model": variables.model_id,
                                    "created_at": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                                    "message": {
                                        "role": "assistant",
                                        "content": sortie_rkllm
                                    },
                                    "done": variables.global_status == 1,
                                    "total_duration": current_time * 1000000000 if variables.global_status == 1 else 0,
                                    "prompt_eval_count": llmResponse["usage"]["prompt_tokens"],
                                    "eval_count": count
                                }
                                yield f"{json.dumps(ollama_chunk)}\n"
                            else:
                                yield f"{json.dumps(llmResponse)}\n\n"

                        # Calcul du temps de traitement
                        total = time.time() - start

                        # Calcul du nombre de tokens par seconde et du nombre ttal de tokens
                        llmResponse["usage"]["tokens_per_second"] = count / total

                        thread_modele.join(timeout=0.005)
                        thread_modele_terminé = not thread_modele.is_alive()

                # Use appropriate content type based on request type
                if is_ollama_request:
                    return Response(generate(), content_type='application/x-ndjson')
                else:
                    return Response(generate(), content_type='text/plain')
        else:
            return jsonify({'status': 'error', 'message': 'Données JSON invalides !'}), 400
    finally:
        # No need to release the lock here as it should be handled by the calling function
        if custom_request is None:
            variables.verrou.release()
        est_bloqué = False
