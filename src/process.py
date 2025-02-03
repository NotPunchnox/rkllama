import threading, time, json
from flask import Flask, request, jsonify, Response
from .variables import global_text, global_status, verrou

PREFIX_MESSAGE  = "<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user"
SUFIX_MESSAGE   = "<|im_end|><|im_start|>assistant"

def Request(modele_rkllm):

    try:
        # Mettre le serveur en état de blocage.
        isLocked = True

        # Obtenir les données JSON de la requête POST.
        data = request.json
        if data and 'messages' in data:
            # Réinitialiser les variables globales.
            global_status = -1

            # Définir la structure de la réponse renvoyée.
            llmResponse = {
                "id": "rkllm_chat",
                "object": "rkllm_chat",
                "created": None,
                "choices": [],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "tokens_per_second": None,
                    "total_tokens": None
                }
            }

            # Traiter les données reçues.
            messages = data['messages']
            userInput = messages
            sortie_rkllm = ""

            print("Messages reçus :", messages)

            if not "stream" in data.keys() or data["stream"] == False:
                
                # Créer un thread pour l'inférence du modèle.
                thread_modele = threading.Thread(target=modele_rkllm.run, args=(userInput,))
                thread_modele.start()

                # Attendre la fin du modèle et vérifier périodiquement le thread d'inférence.
                threadFinish = False
                count = 0
                start = time.time()

                while not threadFinish:
                    while len(global_text) > 0:
                        count += 1
                        sortie_rkllm += global_text.pop(0)
                        time.sleep(0.005)

                        thread_modele.join(timeout=0.005)
                        threadFinish = not thread_modele.is_alive()

                    total = time.time() - start

                    llmResponse["usage"]["tokens_per_second"] = count / tokens 
                    llmResponse["usage"]["completion_tokens"] = count
                    llmResponse["choices"] = [{
                        "role": "assistant",
                        "content": sortie_rkllm,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }]
                return jsonify(llmResponse), 200

            else:
                def generate():
                    thread_modele = threading.Thread(target=modele_rkllm.run, args=(userInput,))
                    thread_modele.start()

                    thread_modele_terminé = False
                    count = 0
                    start = time.time()

                    while not thread_modele_terminé:
                        while len(global_text) > 0:
                            count += 1
                            sortie_rkllm = global_text.pop(0)

                            llmResponse["choices"] = [
                                {
                                "role": "assistant",
                                "content": sortie_rkllm,
                                "logprobs": None,
                                "finish_reason": "stop" if global_status == 1 else None,
                                }
                            ]
                            yield f"{json.dumps(llmResponse)}\n\n"

                        # Calcul du temps de traitement
                        total = time.time() - start

                        # Calcul du nombre de tokens par seconde et du nombre ttal de tokens
                        llmResponse["usage"]["tokens_per_second"] = count / total
                        llmResponse["usage"]["completion_tokens"] = count

                        thread_modele.join(timeout=0.005)
                        thread_modele_terminé = not thread_modele.is_alive()

                return Response(generate(), content_type='text/plain')
        else:
            return jsonify({'status': 'error', 'message': 'Données JSON invalides !'}), 400
    finally:
        verrou.release()
        est_bloqué = False
