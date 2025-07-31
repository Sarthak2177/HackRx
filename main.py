# ✅ main.py (RAG-enabled with HackRx-compliant input + auto question extraction + temp cleanup)

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import os
import numpy as np
import re
import time
from utils.dynamic_decision import DynamicDecisionEngine
from utils.fetch_and_chunk_pdf import (
    download_pdf_and_extract_text,
    chunk_text,
    embed_chunks,
    get_top_k_chunks
)

app = FastAPI()
decision_engine = DynamicDecisionEngine()

security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str  # URL of the PDF document
    questions: List[str] = []  # Optional: will be extracted if not provided

class QueryResponse(BaseModel):
    answers: List[str]
    response_time_seconds: float

def extract_questions_from_text(text: str, max_q: int = 10) -> List[str]:
    question_words = (
        "what", "how", "why", "can", "does", "is", "are", "do",
        "should", "could", "when", "who", "where", "which", "will", "would"
    )
    lines = re.findall(r"[^\n\r]+?[?]", text)
    questions = [
        line.strip() for line in lines
        if line.strip().lower().startswith(question_words) and len(line.strip()) > 20
    ]
    return questions[:max_q]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_decision_engine(
    payload: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    start_time = time.time()

    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    try:
        import hashlib, pickle
        cache_key = hashlib.md5(payload.documents.encode()).hexdigest()
        cache_path = f"cache/{cache_key}.pkl"

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                chunks, chunk_embeddings = cached_data["chunks"], cached_data["embeddings"]
        else:
            raw_text = download_pdf_and_extract_text(payload.documents)
            chunks = chunk_text(raw_text)
            chunk_embeddings = embed_chunks(chunks)
            os.makedirs("cache", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"chunks": chunks, "embeddings": chunk_embeddings}, f)
        if os.path.exists("temp_doc.pdf"):
            os.remove("temp_doc.pdf")  # Cleanup temporary PDF file
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    if not payload.questions:
        payload.questions = extract_questions_from_text(raw_text)

    try:
        answers = []
        batch_size = 3
        for i in range(0, len(payload.questions), batch_size):
            batch_questions = payload.questions[i:i+batch_size]
            batch_chunks = []
            for q in batch_questions:
                batch_chunks.extend(get_top_k_chunks(q, chunks, chunk_embeddings, k=4))
            batch_chunks = list(set(batch_chunks))
            joined_questions = "\n\n".join(batch_questions)
            result = decision_engine.make_decision_from_context(joined_questions, {}, batch_chunks)

            parsed_result = json.loads(result)
            if isinstance(parsed_result, dict):
                if 'questions_analysis' in parsed_result:
                    answers.extend([qa.get('justification', '') or qa.get('answer', '') for qa in parsed_result["questions_analysis"]])
                elif 'decision' in parsed_result:
                    answers.append(parsed_result.get("justification", "") or parsed_result.get("answer", ""))
                else:
                    answers.append(result)
            elif isinstance(parsed_result, list):
                answers.extend([a.get("justification", "") or a.get("answer", "") for a in parsed_result])
            else:
                answers.append(result)
    except Exception:
        answers = ["Could not determine answer from retrieved chunks."] * len(payload.questions)

