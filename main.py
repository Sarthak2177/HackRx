# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import re
import uvicorn
from utils.rag_utils import (
    get_text_from_url,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain
)

app = FastAPI(
    title="Real-time Document Q&A API",
    description="An API that takes a PDF URL and a question, and returns an answer based on the document's content."
)

class QueryRequest(BaseModel):
    pdf_url: str
    query: Optional[str] = None # Made query optional to support auto-extraction

class QueryResponse(BaseModel):
    answer: str

def extract_questions_from_text(text: str, max_q: int = 10) -> List[str]:
    """
    Finds and extracts potential questions from the text content of the document.
    This function has been restored to match your original functionality.
    """
    print("No query provided. Attempting to extract questions from document text...")
    question_words = (
        "what", "how", "why", "can", "does", "is", "are", "do",
        "should", "could", "when", "who", "where", "which"
    )
    
    # Improved regex to find sentences ending with a question mark
    sentences = re.findall(r'([^.!?]*\?)', text)
    
    # Fallback to find sentences starting with question words if no '?' is found
    if not sentences:
        sentences = re.findall(r'(\b(?:' + '|'.join(question_words) + r')\b[^.!?]*\.)', text, re.IGNORECASE)

    extracted_questions = [q.strip() for q in sentences]
    
    print(f"✅ Extracted {len(extracted_questions)} potential questions.")
    return extracted_questions[:max_q]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Q&A API. Use the /ask endpoint to query a PDF."}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    This endpoint performs the entire RAG process in real-time.
    If no query is provided, it extracts questions from the document.
    """
    if not request.pdf_url:
        raise HTTPException(status_code=400, detail="'pdf_url' is required.")

    try:
        # --- Step 1: Get Text from PDF ---
        raw_text = get_text_from_url(request.pdf_url)
        if not raw_text:
            raise HTTPException(status_code=500, detail="Failed to extract text from the PDF.")

        # --- Step 2: Determine Query ---
        query_to_process = request.query
        if not query_to_process:
            # If no query is provided, extract questions from the text
            questions = extract_questions_from_text(raw_text)
            if not questions:
                raise HTTPException(status_code=404, detail="No query was provided and no questions could be extracted from the document.")
            # We'll process the first extracted question for this example
            query_to_process = questions[0]
            print(f"Using first extracted question: '{query_to_process}'")


        # --- Step 3: Chunk the Text ---
        text_chunks = get_text_chunks(raw_text)
        if not text_chunks:
            raise HTTPException(status_code=500, detail="Failed to split text into chunks.")

        # --- Step 4: Create In-Memory Vector Store ---
        vector_store = get_vector_store(text_chunks)
        if not vector_store:
            raise HTTPException(status_code=500, detail="Failed to create the vector store.")

        # --- Step 5: Search for Relevant Documents ---
        print(f"Searching for relevant documents for the query: '{query_to_process}'")
        docs = vector_store.similarity_search(query_to_process)
        print(f"Found {len(docs)} relevant documents.")

        # --- Step 6: Generate Answer using Conversational Chain ---
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": query_to_process}, return_only_outputs=True)
        
        print("✅ Successfully generated a response.")
        return {"answer": response["output_text"]}

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
