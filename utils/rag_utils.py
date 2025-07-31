import fitz  # PyMuPDF
import requests
import numpy as np
import re

# Lazy load embedding model only when needed
def get_embedder():
    global embed_model
    if 'embed_model' not in globals():
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded: all-MiniLM-L6-v2")
    return embed_model

def download_pdf_and_extract_text(url: str) -> str:
    """Download PDF from URL and extract text using PyMuPDF."""
    response = requests.get(url)
    response.raise_for_status()

    with open("temp_doc.pdf", "wb") as f:
        f.write(response.content)

    with fitz.open("temp_doc.pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)

    return text

def chunk_text(text: str, max_length: int = 1000) -> list:
    """Split text into clause-based chunks for better semantic retrieval."""
    parts = re.split(r'(Clause\s+\d+(?:\.\d+)*)', text)
    chunks = []
    for i in range(1, len(parts), 2):
        clause = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ''
        full = f"{clause}: {content}"
        if full:
            chunks.append(full)

    # ✅ Filter out short/incomplete chunks
    chunks = [c for c in chunks if len(c.split()) > 10]
    return chunks

def embed_chunks(chunks: list) -> np.ndarray:
    """Convert chunks to vector embeddings."""
    model = get_embedder()
    return np.array(model.encode(chunks)).astype("float32")

def get_top_k_chunks(query: str, chunks: list, chunk_embeddings: np.ndarray, k: int = 3) -> list:
    """Return the top-k chunks most similar to the query."""
    model = get_embedder()
    query_embedding = model.encode([query])
    scores = (query_embedding @ chunk_embeddings.T)[0]
    top_indices = scores.argsort()[::-1][:k]
    return [chunks[i] for i in top_indices]
