import ctypes, sys
from .classes import *
from .variables import *

# Definir la fonction de rappel
def callback_impl(resultat, donnees_utilisateur, etat):
    global split_byte_data

    if etat == LLMCallState.RKLLM_RUN_FINISH:
        global_status = etat
        print("\n")
        sys.stdout.flush()
    elif etat == LLMCallState.RKLLM_RUN_ERROR:
        global_status = etat
        print("erreur d'execution")
        sys.stdout.flush()
    elif etat == LLMCallState.RKLLM_RUN_GET_LAST_HIDDEN_LAYER:
        '''
        Si vous utilisez la fonction GET_LAST_HIDDEN_LAYER, l'interface de rappel renverra le pointeur de memoire : last_hidden_layer,
        le nombre de tokens : num_tokens, et la taille de la couche cachee : embd_size.
        Avec ces trois parametres, vous pouvez recuperer les donnees de last_hidden_layer.
        Remarque : Les donnees doivent etre recuperees pendant le rappel actuel ; si elles ne sont pas obtenues a temps, le pointeur sera libere lors du prochain rappel.
        '''
        if resultat.last_hidden_layer.embd_size != 0 and resultat.last_hidden_layer.num_tokens != 0:
            taille_donnees = resultat.last_hidden_layer.embd_size * resultat.last_hidden_layer.num_tokens * ctypes.sizeof(ctypes.c_float)
            print(f"taille_donnees : {taille_donnees}")
            
            global_text.append(f"taille_donnees : {taille_donnees}\n")
            chemin_sortie = os.getcwd() + "/last_hidden_layer.bin"

            with open(chemin_sortie, "wb") as fichier_sortie:
                donnees = ctypes.cast(resultat.last_hidden_layer.hidden_states, ctypes.POINTER(ctypes.c_float))
                type_tableau_float = ctypes.c_float * (taille_donnees // ctypes.sizeof(ctypes.c_float))
                tableau_float = type_tableau_float.from_address(ctypes.addressof(donnees.contents))
                fichier_sortie.write(bytearray(tableau_float))
                print(f"Donnees sauvegardees dans {chemin_sortie} avec succes !")
                global_text.append(f"Donnees sauvegardees dans {chemin_sortie} avec succes !")
        else:
            print("Donnees de la couche cachee invalides.")
            global_text.append("Donnees de la couche cachee invalides.")
        
        global_status = etat
        time.sleep(0.05)  # Attente de 0,05 seconde pour attendre le resultat de la sortie
        sys.stdout.flush()
    else:
        # Sauvegarder le texte du token de sortie et l'etat d'execution de RKLLM
        global_status = etat
        # Surveiller si les donnees en bytes actuelles sont completes ; si incompletes, les enregistrer pour une analyse ulterieure
        try:
            global_text.append((split_byte_data + resultat.contents.text).decode('utf-8'))
            print((split_byte_data + resultat.contents.text).decode('utf-8'), end='')
            split_byte_data = bytes(b"")
        except:
            split_byte_data += resultat.contents.text
        sys.stdout.flush()
