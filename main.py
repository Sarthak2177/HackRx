# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Q&A API. Use the /ask endpoint to query a PDF."}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    This endpoint performs the entire RAG process in real-time:
    1. Downloads the PDF from the provided URL.
    2. Extracts and chunks the text.
    3. Creates a temporary, in-memory vector store.
    4. Queries the vector store and a language model to find the answer.
    """
    if not request.pdf_url or not request.query:
        raise HTTPException(status_code=400, detail="Both 'pdf_url' and 'query' are required.")

    try:
        # --- Step 1: Get Text from PDF ---
        raw_text = get_text_from_url(request.pdf_url)
        if not raw_text:
            raise HTTPException(status_code=500, detail="Failed to extract text from the PDF.")

        # --- Step 2: Chunk the Text ---
        text_chunks = get_text_chunks(raw_text)
        if not text_chunks:
            raise HTTPException(status_code=500, detail="Failed to split text into chunks.")

        # --- Step 3: Create In-Memory Vector Store ---
        vector_store = get_vector_store(text_chunks)
        if not vector_store:
            raise HTTPException(status_code=500, detail="Failed to create the vector store.")

        # --- Step 4: Search for Relevant Documents ---
        print("Searching for relevant documents in the vector store...")
        docs = vector_store.similarity_search(request.query)
        print(f"Found {len(docs)} relevant documents.")

        # --- Step 5: Generate Answer using Conversational Chain ---
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": request.query}, return_only_outputs=True)
        
        print("✅ Successfully generated a response.")
        return {"answer": response["output_text"]}

    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
