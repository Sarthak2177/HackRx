from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from query_parser import parse_query
from chunk_utils import load_chunks
from dynamic_decision import get_embed_model

# Optional batching (not required now but scalable)
def embed_chunks(chunks: List[str]) -> np.ndarray:
    model = get_embed_model()
    return model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

def chunk_text(raw_text: str) -> List[str]:
    paragraphs = raw_text.split("\n")
    chunks, current_chunk = [], ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= 500:
            current_chunk += para + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def get_top_k_chunks(query: str, chunks: List[str], embeddings: np.ndarray, k: int = 5) -> List[str]:
    model = get_embed_model()
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    scores = np.dot(embeddings, query_embedding)
    top_k_indices = scores.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k_indices]
