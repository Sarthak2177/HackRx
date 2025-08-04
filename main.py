from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import os
import re
import time
import hashlib
import pickle
from utils.dynamic_decision import DynamicDecisionEngine
from utils.extract_text_from_pdfs import extract_text_from_pdf as download_pdf_and_extract_text
from utils.chunk_utils import load_chunks as chunk_text

app = FastAPI()
security = HTTPBearer()
decision_engine = DynamicDecisionEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str] = []

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

def get_relevant_chunks(questions: List[str], chunks: List[str], top_k: int = 10, max_chars: int = 1500) -> List[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    question_text = " ".join(questions)
    documents = chunks + [question_text]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    question_vec = vectors[-1]
    chunk_vecs = vectors[:-1]

    similarities = cosine_similarity([question_vec], chunk_vecs)[0]
    ranked_chunks = sorted(zip(similarities, chunks), reverse=True)

    top_chunks = [c[:max_chars] for _, c in ranked_chunks[:top_k]]
    return top_chunks

def extract_definitions(text: str, keywords: List[str]) -> List[str]:
    definitions = []
    pattern = re.compile(r"(Clause\s\d+(\.\d+)*).*?(?:(?=Clause\s\d+)|$)", re.DOTALL)
    for match in pattern.finditer(text):
        clause_text = match.group(0)
        if any(kw.lower() in clause_text.lower() for kw in keywords):
            definitions.append(clause_text.strip())
    return definitions

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
        cache_key = hashlib.md5(payload.documents.encode()).hexdigest()
        cache_path = f"cache/{cache_key}.pkl"

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                chunks, chunk_embeddings = cached_data["chunks"], cached_data["embeddings"]
                raw_text = "\n".join(chunks)
        else:
            raw_text = download_pdf_and_extract_text(payload.documents)
            chunks = chunk_text(raw_text)
            chunk_embeddings = ["embedding_placeholder"] * len(chunks)
            os.makedirs("cache", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"chunks": chunks, "embeddings": chunk_embeddings}, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    if not payload.questions:
        payload.questions = extract_questions_from_text(raw_text)

    try:
        answers = []
        batch_size = 2
        for i in range(0, len(payload.questions), batch_size):
            batch_questions = payload.questions[i:i+batch_size]
            joined_questions = "\n\n".join(batch_questions)
            relevant_chunks = get_relevant_chunks(batch_questions, chunks)
            result = decision_engine.make_decision_from_context(joined_questions, {}, relevant_chunks)

            parsed_result = json.loads(result)
            if isinstance(parsed_result, dict):
                if 'questions_analysis' in parsed_result:
                    batch_answers = [qa.get('justification', '') or qa.get('answer', '') for qa in parsed_result["questions_analysis"]]
                elif 'decision' in parsed_result:
                    batch_answers = [parsed_result.get("justification", "") or parsed_result.get("answer", "")]
                else:
                    batch_answers = [result]
            elif isinstance(parsed_result, list):
                batch_answers = [a.get("justification", "") or a.get("answer", "") for a in parsed_result]
            else:
                batch_answers = [result]

            for ans in batch_answers:
                if len(ans) > 1000:
                    ans = ans[:950].rsplit('.', 1)[0] + '.'
                answers.append(ans.strip())

    except Exception:
        answers = ["Could not determine answer from retrieved chunks."] * len(payload.questions)

    response_time = round(time.time() - start_time, 2)
    return {"answers": answers, "response_time_seconds": response_time}


