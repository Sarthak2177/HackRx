# utils/rag_utils.py
import requests
import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.Youtubeing import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

# It's good practice to load environment variables for API keys
load_dotenv()

def get_text_from_url(pdf_url: str) -> str:
    """
    Downloads a PDF from a URL and extracts all text content.
    """
    print(f"Downloading PDF from: {pdf_url}")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        
        # Use an in-memory byte stream to handle the PDF content
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        print("✅ Text extraction successful.")
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF: {e}")
        return None
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def get_text_chunks(text: str) -> list:
    """
    Splits a long text into smaller, manageable chunks.
    """
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks

def get_vector_store(text_chunks: list):
    """
    Generates vector embeddings for text chunks and creates an in-memory FAISS vector store.
    """
    if not text_chunks:
        print("No text chunks to process. Skipping vector store creation.")
        return None

    print("Initializing embedding model...")
    # Using 'hkunlp/instructor-large' for high-quality embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    
    print("Creating FAISS vector store from text chunks...")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("✅ Vector store created successfully in memory.")
    return vector_store

def get_conversational_chain():
    """
    Loads and initializes the question-answering chain with a language model.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context". Don't provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    print("Initializing language model...")
    # Using Google's Flan-T5 model from Hugging Face Hub
    model = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    print("Loading question-answering chain...")
    chain = load_qa_chain(llm=model, chain_type="stuff")
    print("✅ Conversational chain ready.")
    return chain
