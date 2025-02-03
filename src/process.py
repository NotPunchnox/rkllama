import threading, time, json
from transformers import AutoTokenizer
from flask import Flask, request, jsonify, Response
from .variables import global_text, global_status, verrou

bos = "<|im_start|>"
eos = "<|im_end|>"

system   = "Tu es un assistant artificiel."
history  = []
model_id = "punchnox/Tinnyllama-1.1B-rk3588-rkllm-1.1.4"

# messages = [{
#     "role": "user",
#     "content": "Quelle est la capitale de la France ?"
# },
# {
#     "role": "assistant",
#     "content": "La capitale de la France est Paris."
# }]

def Request(modele_rkllm):
    global history

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
            user = {"role": "user", "content": data["messages"]}

            # Ajout du system prompt si il y en a un
            if len(system) < 1:
                prompt = [user]
            else:
                prompt = [
                    {"role": "system", "content": system},
                    user
                ]

            history.append({ role: "assistant", content: "" })

            # Mettre en place le tokenizer et le chatTemplate            
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            prompt = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)

            sortie_rkllm = ""
            print("Messages reçus :", messages)


            if not "stream" in data.keys() or data["stream"] == False:
                
                # Créer un thread pour l'inférence du modèle.
                thread_modele = threading.Thread(target=modele_rkllm.run, args=(prompt,))
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
                    thread_modele = threading.Thread(target=modele_rkllm.run, args=(prompt,))
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
