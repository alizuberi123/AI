import os
import json

BASE_DIR = "rag_sessions"
#HI
def create_rag(name):
    path = os.path.join(BASE_DIR, name)
    if os.path.exists(path):
        return False
    os.makedirs(os.path.join(path, "chroma_db"), exist_ok=True)
    os.makedirs(os.path.join(path, "files"), exist_ok=True)
    with open(os.path.join(path, "chat_history.json"), "w") as f:
        json.dump([], f)
    return True

def list_rags():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    return [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]

def get_rag_path(name):
    return os.path.join(BASE_DIR, name)

def save_chat_history(rag_name, history):
    path = os.path.join(BASE_DIR, rag_name, "chat_history.json")
    with open(path, "w") as f:
        json.dump(history, f)

def load_chat_history(rag_name):
    path = os.path.join(BASE_DIR, rag_name, "chat_history.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []
