from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import pickle
import os
import fitz  # PyMuPDF
import re
from utils.dynamic_decision import DynamicDecisionEngine

app = FastAPI()
decision_engine = DynamicDecisionEngine()

# Enable Swagger "Authorize" button
security = HTTPBearer()

# Allow Swagger UI to work cross-origin (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str  # kept for compatibility
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_decision_engine(
    payload: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    try:
        with open("vector_store/chunks.pkl", "rb") as f:
            text_chunks = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load document chunks: {str(e)}")

    MAX_TOKENS_PER_REQUEST = 3000
    answers = []

    for question in payload.questions:
        answer_found = False
        for chunk in text_chunks:
            chunk_text = " ".join(chunk) if isinstance(chunk, list) else chunk
            short_chunk = chunk_text[:MAX_TOKENS_PER_REQUEST * 4]

            result = decision_engine.make_decision_from_context(question, {}, [short_chunk])
            try:
                parsed_result = json.loads(result)
                if isinstance(parsed_result, dict) and 'decision' in parsed_result:
                    answers.append(parsed_result.get("justification", "") or parsed_result.get("answer", ""))
                    answer_found = True
                    break
                elif isinstance(parsed_result, list):
                    answers.extend([a.get("justification", "") or a.get("answer", "") for a in parsed_result])
                    answer_found = True
                    break
                else:
                    answers.append(result)
                    answer_found = True
                    break
            except Exception:
                continue

        if not answer_found:
            answers.append("Could not determine answer from available document chunks.")

    return {"answers": answers}

@app.post("/hackrx/upload", response_model=QueryResponse)
async def upload_and_ask(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(contents)

    try:
        with fitz.open("temp_uploaded.pdf") as doc:
            combined_text = "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        if os.path.exists("temp_uploaded.pdf"):
            os.remove("temp_uploaded.pdf")

    # Auto-generate questions or important statements from the text
    extracted_sentences = re.findall(r'[^\n\r]+?[.?!]', combined_text)
    questions = [s.strip() for s in extracted_sentences if len(s.strip()) > 20][:5]  # top 5 statements/questions

    MAX_TOKENS_PER_REQUEST = 3000
    answers = []

    for question in questions:
        answer_found = False
        for chunk in combined_text.split("\n\n"):
            if not chunk.strip():
                continue
            short_chunk = chunk[:MAX_TOKENS_PER_REQUEST * 4]
            result = decision_engine.make_decision_from_context(question, {}, [short_chunk])
            try:
                parsed_result = json.loads(result)
                if isinstance(parsed_result, dict) and 'decision' in parsed_result:
                    answers.append(parsed_result.get("justification", "") or parsed_result.get("answer", ""))
                    answer_found = True
                    break
                elif isinstance(parsed_result, list):
                    answers.extend([a.get("justification", "") or a.get("answer", "") for a in parsed_result])
                    answer_found = True
                    break
                else:
                    answers.append(result)
                    answer_found = True
                    break
            except Exception:
                continue

        if not answer_found:
            answers.append("Could not determine answer from uploaded document.")

    return {"answers": answers}
