import threading, time, json
from transformers import AutoTokenizer
from flask import Flask, request, jsonify, Response
import src.variables as variables


def Request(modele_rkllm):

    try:
        # Mettre le serveur en état de blocage.
        isLocked = True

        data = request.json
        if data and 'messages' in data:
            # Réinitialiser les variables globales.
            variables.global_status = -1

            # Définir la structure de la réponse renvoyée.
            llmResponse = {
                "id": "rkllm_chat",
                "object": "rkllm_chat",
                "created": None,
                "choices": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "tokens_per_second": 0,
                    "total_tokens": 0
                }
            }

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
            #print("Prompt final: ", prompt)
            #print("Messages reçus :", messages)

            sortie_rkllm = ""

            if not "stream" in data.keys() or data["stream"] == False:
                # Créer un thread pour l'inférence du modèle.
                thread_modele = threading.Thread(target=modele_rkllm.run, args=(prompt,))
                try:
                    thread_modele.start()
                    print("Thread d’inférence démarré")
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
                            yield f"{json.dumps(llmResponse)}\n\n"

                        # Calcul du temps de traitement
                        total = time.time() - start

                        # Calcul du nombre de tokens par seconde et du nombre ttal de tokens
                        llmResponse["usage"]["tokens_per_second"] = count / total

                        thread_modele.join(timeout=0.005)
                        thread_modele_terminé = not thread_modele.is_alive()

                return Response(generate(), content_type='text/plain')
        else:
            return jsonify({'status': 'error', 'message': 'Données JSON invalides !'}), 400
    finally:
        variables.verrou.release()
        est_bloqué = False
