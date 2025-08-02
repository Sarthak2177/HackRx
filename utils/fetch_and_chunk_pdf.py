import fitz  # PyMuPDF
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model once
embed_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

def download_pdf_and_extract_text(url: str) -> str:
    """Download PDF from URL and extract text using PyMuPDF."""
    response = requests.get(url)
    response.raise_for_status()

    with open("temp_doc.pdf", "wb") as f:
        f.write(response.content)

    with fitz.open("temp_doc.pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)

    return text

def chunk_text(text: str, max_length: int = 500) -> list:
    """Split text into smaller chunks for better LLM context matching."""
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) < max_length:
            current += line + " "
        else:
            chunks.append(current.strip())
            current = line + " "
    if current:
        chunks.append(current.strip())
    return chunks

def embed_chunks(chunks: list) -> np.ndarray:
    """Convert chunks to vector embeddings."""
    return np.array(embed_model.encode(chunks)).astype("float32")

def get_top_k_chunks(query: str, chunks: list, chunk_embeddings: np.ndarray, k: int = 5) -> list:
    """Return the top-k chunks most similar to the query."""
    query_embedding = embed_model.encode([query])
    scores = (query_embedding @ chunk_embeddings.T)[0]
    top_indices = scores.argsort()[::-1][:k]
    return [chunks[i] for i in top_indices]
