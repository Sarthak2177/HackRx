import fitz  # PyMuPDF
import requests
import numpy as np
import re
import openai
import os

# Set your OpenAI API key (use env variable in deployment)
openai.api_key = os.getenv("OPENAI_API_KEY")

def download_pdf_and_extract_text(url: str) -> str:
    """Download PDF from URL and extract text using PyMuPDF."""
    response = requests.get(url)
    response.raise_for_status()

    with open("temp_doc.pdf", "wb") as f:
        f.write(response.content)

    with fitz.open("temp_doc.pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)

    os.remove("temp_doc.pdf")
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

    chunks = [c for c in chunks if len(c.split()) > 10]
    return chunks[:30]  # Limit to 30 chunks for cost and performance

def embed_chunks(chunks: list) -> np.ndarray:
    """Use OpenAI API to embed text chunks."""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    return np.array([r.embedding for r in response.data], dtype=np.float32)

def get_top_k_chunks(query: str, chunks: list, chunk_embeddings: np.ndarray, k: int = 3) -> list:
    """Embed query and return top-k similar chunks."""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = np.array(response.data[0].embedding, dtype=np.float32)
    scores = query_embedding @ chunk_embeddings.T
    top_indices = scores.argsort()[::-1][:k]
    return [chunks[i] for i in top_indices]
