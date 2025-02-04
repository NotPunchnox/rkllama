import threading

isLocked = False
global_status = -1
global_text = []
split_byte_data = bytes(b"")

verrou = threading.Lock()

model_id = ""
system   = "Tu es un assistant artificiel."