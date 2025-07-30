import os
import tempfile
import json
import faiss
import numpy as np
import logging
import asyncio
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
from utils.chunk_utils import load_chunks
from contextlib import asynccontextmanager
from utils.dynamic_decision import DynamicDecisionEngine


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
index = None
chunks = None
decision_engine = None

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Lifespan for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    await loop.run_in_executor(None, load_decision_engine)
    logger.info("Application startup complete")

    yield  # App runs here

    logger.info("Shutting down application...")
    try:
        global model, index, chunks, decision_engine
        if model:
            del model
        if index:
            del index
        if chunks:
            del chunks
        if decision_engine:
            del decision_engine
        logger.info("Resources released successfully")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# FastAPI app instance
app = FastAPI(
    title="LLM Query Retrieval API",
    description="API for document-based query retrieval using LLM",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS setup (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Request and response models
class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v

class UploadResponse(BaseModel):
    filename: str
    text_preview: str

# Global variables with type hints
model: Optional[SentenceTransformer] = None
index: Optional[faiss.Index] = None
chunks: Optional[List[str]] = None
decision_engine: Optional[DynamicDecisionEngine] = None

# OAuth2 for Bearer token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # Placeholder, not used for validation

# Load models with enhanced error handling and logging (synchronous for now)
def load_models() -> tuple:
    global model, index, chunks
    try:
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loading FAISS index...")
        index = faiss.read_index("vector_store/faiss_index.bin")
        logger.info("Loading chunks...")
        chunks = load_chunks("vector_store/chunks.pkl")
        if index.ntotal != len(chunks):
            raise ValueError(f"Mismatch: {index.ntotal} vectors vs {len(chunks)} chunks")
        logger.info(f"Successfully loaded {len(chunks)} chunks")
        return model, index, chunks
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

# Load decision engine with logging (synchronous for now)
def load_decision_engine() -> DynamicDecisionEngine:
    global decision_engine
    try:
        logger.info("Initializing DynamicDecisionEngine...")
        decision_engine = DynamicDecisionEngine()
        logger.info("Decision engine initialized successfully")
        return decision_engine
    except Exception as e:
        logger.error(f"Failed to load decision engine: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Decision engine loading failed: {str(e)}")

# File extraction functions with improved error handling
def extract_text_with_pypdf2(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

def extract_text_with_ocr(pdf_path: str) -> str:
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        images = convert_from_path(pdf_path)
        return "\n".join(pytesseract.image_to_string(img) for img in images)
    except Exception as e:
        logger.error(f"OCR extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(e)}")

def extract_text_from_docx(docx_path: str) -> str:
    try:
        doc = docx.Document(docx_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.error(f"DOCX extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DOCX extraction failed: {str(e)}")

def extract_text_from_msg(msg_path: str) -> str:
    try:
        msg = extract_msg.Message(msg_path)
        return f"Subject: {msg.subject}\nBody:\n{msg.body}"
    except Exception as e:
        logger.error(f"MSG extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MSG extraction failed: {str(e)}")

def extract_text_from_file(uploaded_file: UploadFile) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.filename) as tmp_file:
            tmp_file.write(uploaded_file.file.read())
            tmp_path = tmp_file.name

        ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if ext == ".pdf":
            text = extract_text_with_pypdf2(tmp_path)
            if not text or len(text) < 100:
                text = extract_text_with_ocr(tmp_path)
        elif ext == ".docx":
            text = extract_text_from_docx(tmp_path)
        elif ext == ".msg":
            text = extract_text_from_msg(tmp_path)
        elif ext == ".txt":
            text = uploaded_file.file.read().decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        return text.strip()
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"File extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File extraction failed: {str(e)}")
    finally:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)

# Enhanced semantic search with relevance scoring
def search_similar_chunks_with_scores(user_query: str, model: SentenceTransformer, index: faiss.Index, chunks: List[str], top_k: int = 5) -> List[Dict]:
    try:
        embedding = model.encode([user_query], convert_to_tensor=True)
        distances, indices = index.search(np.array(embedding), top_k)
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            similarity = 1 / (1 + distance) if distance >= 0 else 1.0
            relevance = "High" if similarity > 0.7 else "Medium" if similarity > 0.5 else 'Low'
            results.append({
                'rank': i + 1,
                'chunk': chunks[idx],
                'similarity_score': float(similarity),
                'relevance': relevance
            })
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

def safe_to_float(value: any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def transform_to_required_format(query: str, response_json: Dict, matched_chunks: List[str], document_name: str = "Unknown") -> Dict:
    try:
        if isinstance(response_json, str):
            response_json = json.loads(response_json)

        structured = {
            "query": query,
            "decision": "Unknown",
            "amount": None,
            "conditions": [],
            "justification": "No justification provided.",
            "confidence": 0.5,
            "source_clauses": []
        }

        qa = response_json.get("questions_analysis", [{}])[0] if "questions_analysis" in response_json else {}
        scenario = response_json.get("scenario_analysis", {})

        target = qa or scenario or response_json

        structured.update({
            "decision": target.get("decision") or target.get("answer", "Unknown"),
            "amount": target.get("amount"),
            "conditions": target.get("conditions", []),
            "justification": target.get("justification", "No justification provided."),
            "confidence": safe_to_float(target.get("confidence", 0.5)),
            "source_clauses": [
                {
                    "clause_id": clause.split(':')[0].strip() if ':' in clause else f"clause_{i}",
                    "text": clause,
                    "document": document_name
                } for i, clause in enumerate(target.get("referenced_clauses", matched_chunks))
            ]
        })

        if structured["confidence"] < 0.7 and len(structured["source_clauses"]) > 1:
            structured["confidence"] = min(0.95, structured["confidence"] + 0.1 * len(structured["source_clauses"]))

        if not structured["source_clauses"]:
            structured["source_clauses"] = [
                {"clause_id": f"chunk_{i}", "text": chunk, "document": document_name}
                for i, chunk in enumerate(matched_chunks[:3])
            ]

        return structured
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid response format: {str(e)}")
    except Exception as e:
        logger.error(f"Transform error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transform failed: {str(e)}")

def process_user_query(user_input: str, model: SentenceTransformer, index: faiss.Index, chunks: List[str], decision_engine: DynamicDecisionEngine, context_text: str = "") -> Dict:
    try:
        if not user_input.strip():
            raise HTTPException(status_code=400, detail="User query is empty")

        search_query = f"{user_input}\n\nContext: {context_text}" if context_text else user_input
        logger.info(f"Processing query: {search_query}")
        parsed_query_details = parse_query(user_input)
        chunk_results = search_similar_chunks_with_scores(search_query, model, index, chunks, top_k=8)

        if not chunk_results:
            logger.warning("No relevant chunks found for query")
            raise HTTPException(status_code=404, detail="No relevant information found")

        if context_text and not any(context_text in res['chunk'] for res in chunk_results):
            chunk_results.insert(0, {
                'rank': 0, 'chunk': context_text, 'similarity_score': 1.0, 'relevance': 'High'
            })

        matched_chunks = [res['chunk'] for res in chunk_results]
        document_name = "Uploaded_Document" if context_text else "Policy_Database"

        json_response_str = decision_engine.make_decision_from_context(user_input, parsed_query_details, matched_chunks)
        response_json = json.loads(json_response_str)
        return transform_to_required_format(user_input, response_json, matched_chunks, document_name)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid LLM response: {str(e)}")
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Authentication dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    expected_token = os.environ.get("AUTH_TOKEN", "SECRET123")  # Fallback for local testing
    if not token or token != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    logger.info("Authentication successful")
    return token

@app.post("/query", response_model=Dict, summary="Process a query against document data")
async def process_api_query(request: QueryRequest, token: str = Depends(get_current_user)):
    return process_user_query(request.query, model, index, chunks, decision_engine, request.context)

@app.post("/upload", response_model=UploadResponse, summary="Upload a document for processing")
async def upload_file(file: UploadFile = File(...), token: str = Depends(get_current_user)):
    text = extract_text_from_file(file)
    if not text:
        raise HTTPException(status_code=400, detail="Failed to extract text from file")
    return UploadResponse(filename=file.filename, text_preview=text[:500] + "..." if len(text) > 500 else text)

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")  # Optional debug
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
