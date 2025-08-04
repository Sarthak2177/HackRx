import pickle
import re
from typing import List

def load_chunks(path: str) -> List[str]:
    """Load pre-saved chunks from pickle file"""
    with open(path, "rb") as f:
        chunks, _ = pickle.load(f)
    return chunks

def chunk_text(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Splits the full text into semantically meaningful chunks, capped at max_chunk_size characters.
    Tries to split by paragraph or sentence boundaries.
    """
    # First, split by double line breaks (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds limit, save current chunk and start a new one
        if len(current_chunk) + len(para) + 1 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n" + para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
