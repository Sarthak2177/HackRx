# utils/rag_utils.py
import os
import requests
import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.Youtubeing import load_qa_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables (for local development)
load_dotenv()

def get_text_from_url(pdf_url: str) -> str:
    """Downloads a PDF from a URL and extracts all text content."""
    print(f"Downloading PDF from: {pdf_url}")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
        print("✅ Text extraction successful.")
        return text
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def get_text_chunks(text: str) -> list:
    """Splits a long text into smaller, manageable chunks."""
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks

def get_vector_store(text_chunks: list):
    """Generates embeddings and creates an in-memory FAISS vector store."""
    if not text_chunks:
        return None
    print("Initializing embedding model...")
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("✅ Vector store created successfully.")
    return vector_store

def get_conversational_chain():
    """Loads and initializes the QA chain with the Groq language model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context". Don't provide a wrong answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    print("Initializing Groq language model...")
    model = ChatGroq(temperature=0, model_name="llama3-8b-8192")
    print("Loading question-answering chain...")
    chain = load_qa_chain(llm=model, chain_type="stuff")
    print("✅ Conversational chain ready.")
    return chain
