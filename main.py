import os
import time
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fetch_and_chunk_pdf import download_pdf_and_extract_text
from rag_utils import chunk_text, embed_chunks, get_top_k_chunks
from dynamic_decision import DynamicDecisionEngine

# ✅ Limit PyTorch threads to save memory
torch.set_num_threads(1)

app = FastAPI()

decision_engine = DynamicDecisionEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PDFPayload(BaseModel):
    documents: str
    query: str

@app.post("/rag")
async def process_pdf_and_query(payload: PDFPayload):
    start = time.time()

    raw_text = download_pdf_and_extract_text(payload.documents)
    chunks = chunk_text(raw_text)

    # ✅ Cap chunks to prevent memory overload
    if len(chunks) > 100:
        chunks = chunks[:100]

    chunk_embeddings = embed_chunks(chunks)
    top_chunks = get_top_k_chunks(payload.query, chunks, chunk_embeddings)
    answer = decision_engine.query(payload.query, top_chunks)

    # ✅ Cleanup large vars to release memory
    del raw_text, chunk_embeddings, chunks

    if os.path.exists("temp_doc.pdf"):
        os.remove("temp_doc.pdf")

    return {
        "answer": answer,
        "time_taken": round(time.time() - start, 2)
    }

@app.get("/")
def read_root():
    return {"status": "working"}
