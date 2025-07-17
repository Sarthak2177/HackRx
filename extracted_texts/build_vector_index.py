import os
import glob
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Folder where .txt files were saved
TEXT_FOLDER = "extracted_texts"

# Output paths — these should match query_engine.py
INDEX_FILE = "vector_store/faiss_index.bin"   
CHUNKS_FILE = "vector_store/chunks.pkl"       

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, max_length=500):
    """Chunk long text into smaller pieces for better embeddings"""
    chunks = []
    current_chunk = ""
    for line in text.split("\n"):
        if len(current_chunk) + len(line) < max_length:
            current_chunk += line + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def build_index():
    all_chunks = []
    chunk_sources = []

    for txt_file in glob.glob(f"{TEXT_FOLDER}/*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        chunks = chunk_text(raw_text)
        all_chunks.extend(chunks)
        chunk_sources.extend([txt_file] * len(chunks))

    print(f"✅ Total Chunks: {len(all_chunks)}")

    # Embed all chunks
    embeddings = model.encode(all_chunks, show_progress_bar=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    faiss.write_index(index, INDEX_FILE)

    # Save chunks and their source files
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump((all_chunks, chunk_sources), f)

    print("✅ FAISS index and chunks saved.")

if __name__ == "__main__":
    build_index()
