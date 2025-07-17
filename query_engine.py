import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from utils.chunk_utils import load_chunks  # Reuse chunk loader
from utils.llm_decision import make_decision_from_context  # LLM decision logic

# Load model and index
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)

# ✅ Corrected file names
INDEX_PATH = "vector_store/faiss_index.bin"
CHUNK_PATH = "vector_store/chunks.pkl"

print("\n🔍 Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
chunks = load_chunks(CHUNK_PATH)

assert index.ntotal == len(chunks), "Mismatch between index and chunks"

# Step 1 - Parse user query
from utils.query_parser import parse_query  # We'll implement this too

# Step 2 - Semantic search
def search_similar_chunks(user_query, top_k=5):
    embedding = model.encode([user_query])
    distances, indices = index.search(np.array(embedding), top_k)
    return [chunks[i] for i in indices[0]]

# Step 3 - Run full pipeline
def process_user_query(user_input):
    print(f"\n💬 User query: {user_input}")

    # 1. Parse details
    parsed = parse_query(user_input)
    print("Parsed Query:", parsed)

    # 2. Semantic search
    matched_chunks = search_similar_chunks(user_input, top_k=5)
    print("\n📄 Retrieved Chunks:")
    for i, chunk in enumerate(matched_chunks, 1):
        print(f"[{i}] {chunk[:200]}...")

    # 3. Ask LLM to make decision
    response = make_decision_from_context(user_input, matched_chunks, parsed)
    return response

if __name__ == "__main__":
    while True:
        user_query = input("📝 Enter your query : ").strip()
        if user_query.lower() in ("exit", "quit"):
            break

        print(f"\n💬 User query: {user_query}")
        parsed = parse_query(user_query)

        if all(value is None for value in parsed.values()):
            print("❗ Couldn't extract meaningful information. Please try a better query.\n")
            continue

        print("Parsed Query:", parsed)

        matched_chunks = search_similar_chunks(user_query, top_k=5)
        print("\n📄 Retrieved Chunks:")
        for i, chunk in enumerate(matched_chunks, 1):
            print(f"[{i}] {chunk[:200]}...")

        # ✅ Fix: only pass 2 args (user_input, chunks)
        response = make_decision_from_context(user_query, matched_chunks)

        print("\n📌 Final Decision:")
        try:
            parsed_response = json.loads(response)
            print(json.dumps(parsed_response, indent=2, ensure_ascii=False))
        except Exception:
            print(response)

