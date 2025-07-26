import pickle

def load_chunks(path):
    with open(path, "rb") as f:  # ✅ open in binary mode
        chunks, _ = pickle.load(f)  # ✅ extract only the chunks
    return chunks