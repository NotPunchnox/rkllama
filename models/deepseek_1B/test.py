import requests, json

def get_model_info(user_id, model_id):
    url = f"https://huggingface.co/api/models/{user_id}/{model_id}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erreur {response.status_code}: Impossible de récupérer les données.")
        return None

user_id = input("user_id: ")
model_id = input("model_id: ")

data = get_model_info(user_id, model_id)

if data:
    # print(json.dumps(model_info, sort_keys=True, indent=4))
    # print(model_info)
    tokenizer_config = data["config"]["tokenizer_config"]

    print("Arch :", data["config"]["architectures"])
    print("Model type :", data["config"]["model_type"])
    # print("Tokenizer config :", tokenizer_config)
    print("Chat template: ", tokenizer_config["chat_template"])

    print("\nSpecial tokens:")

    for i in tokenizer_config.keys():
        if "token" in i:
            if type(tokenizer_config[i]) is not str and tokenizer_config[i] is not None:
                print(i, ":", tokenizer_config[i]["content"])
            else:
                print(i, ":", tokenizer_config[i])