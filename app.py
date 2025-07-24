import sys
import os

# Comprehensive fix for Streamlit + PyTorch compatibility
def patch_torch_classes():
    try:
        import torch
        if hasattr(torch, 'classes'):
            class MockPath:
                def __init__(self):
                    self._path = []
                
                def __iter__(self):
                    return iter(self._path)
                
                def __getitem__(self, index):
                    return self._path[index]
            
            if not hasattr(torch.classes, '__path__'):
                torch.classes.__path__ = MockPath()
                
    except ImportError:
        pass

patch_torch_classes()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tempfile
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.chunk_utils import load_chunks
from utils.llm_decision import make_decision_from_context
from utils.query_parser import parse_query

# Import your existing extraction functions
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import docx
import extract_msg

# Set page config
st.set_page_config(
    page_title="Document Query Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load model and index (cache these for performance)
@st.cache_resource
def load_models():
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(EMBED_MODEL)
    
    INDEX_PATH = "vector_store/faiss_index.bin"
    CHUNK_PATH = "vector_store/chunks.pkl"
    
    try:
        st.write("🔍 Loading FAISS index...")
        index = faiss.read_index(INDEX_PATH)
        chunks = load_chunks(CHUNK_PATH)
        
        assert index.ntotal == len(chunks), "Mismatch between index and chunks"
        st.success(f"✅ Loaded {index.ntotal} vectors and {len(chunks)} chunks")
        
        return model, index, chunks
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return model, None, None

# Your existing extraction functions
def extract_text_with_pypdf2(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
        return ""

def extract_text_with_ocr(pdf_path):
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        images = convert_from_path(pdf_path)
        return "\n".join(pytesseract.image_to_string(img) for img in images)
    except Exception as e:
        st.error(f"OCR error: {e}")
        return ""

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        st.error(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_msg(msg_path):
    try:
        msg = extract_msg.Message(msg_path)
        return f"Subject: {msg.subject}\nBody:\n{msg.body}"
    except Exception as e:
        st.error(f"MSG extraction error: {e}")
        return ""

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        if ext == ".pdf":
            text = extract_text_with_pypdf2(tmp_path)
            if not text or len(text) < 100:
                text = extract_text_with_ocr(tmp_path)
        elif ext == ".docx":
            text = extract_text_from_docx(tmp_path)
        elif ext == ".msg":
            text = extract_text_from_msg(tmp_path)
        elif ext == ".txt":
            text = uploaded_file.getvalue().decode("utf-8")
        else:
            return None
            
        return text.strip()
    finally:
        os.unlink(tmp_path)

# Enhanced semantic search with similarity scores
def search_similar_chunks_with_scores(user_query, model, index, chunks, top_k=5):
    """Enhanced search with similarity scores"""
    if index is None or chunks is None:
        return []
    
    embedding = model.encode([user_query])
    distances, indices = index.search(np.array(embedding), top_k)
    
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        # Convert distance to similarity score (lower distance = higher similarity)
        similarity = 1 / (1 + distance)
        results.append({
            'rank': i + 1,
            'chunk': chunks[idx],
            'similarity_score': similarity,
            'relevance': 'High' if similarity > 0.7 else 'Medium' if similarity > 0.5 else 'Low'
        })
    
    return results

# Enhanced display function for Streamlit
def display_chunks_nicely(chunk_results):
    """Display chunks with better formatting in Streamlit"""
    if not chunk_results:
        st.warning("No relevant chunks found.")
        return
    
    st.markdown("### 📚 Knowledge Base References")
    
    for result in chunk_results:
        # Create columns for better layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.metric(f"Reference #{result['rank']}", f"{result['similarity_score']:.3f}", f"{result['relevance']} Relevance")
        
        with col2:
            # Show chunk content in expandable section
            chunk_text = result['chunk']
            preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            
            with st.expander(f"📖 Reference {result['rank']} Preview"):
                st.text_area(
                    f"Content (Relevance: {result['relevance']})", 
                    value=chunk_text, 
                    height=150, 
                    key=f"chunk_{result['rank']}"
                )

# Enhanced response formatting for Streamlit
def format_final_response(response, chunk_results):
    """Format and display final response with reasoning"""
    st.markdown("### 🎯 Final Decision & Reasoning")
    
    try:
        parsed_response = json.loads(response)
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["Decision", "Details", "Sources"])
        
        with tab1:
            # Display main decision
            if 'decision' in parsed_response:
                st.success(f"**Decision:** {parsed_response['decision']}")
            
            # Display reasoning
            if 'reasoning' in parsed_response:
                st.markdown("**Reasoning:**")
                st.write(parsed_response['reasoning'])
            
            # Display confidence
            if 'confidence' in parsed_response:
                confidence = parsed_response['confidence']
                st.metric("Confidence Level", confidence)
        
        with tab2:
            # Display eligibility details if present
            if 'eligibility' in parsed_response:
                st.markdown("**Eligibility Details:**")
                eligibility = parsed_response['eligibility']
                if isinstance(eligibility, dict):
                    for key, value in eligibility.items():
                        st.write(f"• **{key}:** {value}")
                else:
                    st.write(eligibility)
            
            # Display any additional fields
            other_fields = {k: v for k, v in parsed_response.items() 
                          if k not in ['decision', 'reasoning', 'confidence', 'eligibility']}
            
            if other_fields:
                st.markdown("**Additional Information:**")
                for key, value in other_fields.items():
                    if isinstance(value, dict):
                        st.write(f"**{key.title()}:**")
                        for k, v in value.items():
                            st.write(f"  • {k}: {v}")
                    elif isinstance(value, list):
                        st.write(f"**{key.title()}:**")
                        for item in value:
                            st.write(f"  • {item}")
                    else:
                        st.write(f"**{key.title()}:** {value}")
        
        with tab3:
            # Show source summary
            if chunk_results:
                high_rel = sum(1 for r in chunk_results if r['relevance'] == 'High')
                med_rel = sum(1 for r in chunk_results if r['relevance'] == 'Medium')
                low_rel = sum(1 for r in chunk_results if r['relevance'] == 'Low')
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("High Relevance", high_rel)
                with col2:
                    st.metric("Medium Relevance", med_rel)
                with col3:
                    st.metric("Low Relevance", low_rel)
                with col4:
                    avg_score = sum(r['similarity_score'] for r in chunk_results) / len(chunk_results)
                    st.metric("Avg. Score", f"{avg_score:.3f}")
                
    except json.JSONDecodeError:
        st.markdown("**Response:**")
        st.write(response)

# Enhanced process query function
def process_user_query(user_input, model, index, chunks, context_text=""):
    """Enhanced process user query with better presentation"""
    st.write(f"💬 **User Query:** {user_input}")
    
    # If we have context text, create a combined query for search
    search_query = f"{user_input}\n\nContext: {context_text}" if context_text else user_input
    
    # 1. Parse details
    parsed = parse_query(user_input)
    with st.expander("🔍 Query Analysis"):
        st.json(parsed)
    
    if all(value is None for value in parsed.values()):
        st.warning("❗ Couldn't extract meaningful information. Please try a better query.")
        return None, [], parsed
    
    # 2. Enhanced semantic search with scores
    chunk_results = search_similar_chunks_with_scores(search_query, model, index, chunks, top_k=5)
    
    # Add context text as first chunk if provided
    if context_text:
        chunk_results.insert(0, {
            'rank': 0,
            'chunk': context_text,
            'similarity_score': 1.0,
            'relevance': 'High'
        })
    
    # 3. Display chunks nicely
    display_chunks_nicely(chunk_results)
    
    # 4. Get LLM decision
    matched_chunks = [result['chunk'] for result in chunk_results]
    response = make_decision_from_context(user_input, matched_chunks)
    
    # 5. Format and display final response
    format_final_response(response, chunk_results)
    
    return response, matched_chunks, parsed

# Main UI
def main():
    st.title("🔍 Document Query Assistant")
    st.markdown("Ask questions about your documents or upload new ones for analysis!")
    
    # Load models at startup
    model, index, chunks = load_models()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'docx', 'msg', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, MSG, or TXT files"
        )
        
        # Process uploaded files
        uploaded_texts = {}
        if uploaded_files:
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    text = extract_text_from_file(file)
                    if text:
                        uploaded_texts[file.name] = text
                        st.success(f"✅ {file.name} processed ({len(text)} chars)")
                    else:
                        st.error(f"❌ Failed to process {file.name}")
        
        # Add example queries
        st.header("💡 Example Queries")
        st.markdown("""
        - What are the eligibility criteria for insurance?
        - What is the age limit for policy applications?
        - What documents are required for claims?
        - What are the premium rates for different age groups?
        """)
    
    # Main content area
    st.header("💬 Query Interface")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Type your question", "Use document context", "Upload & Ask"]
    )
    
    user_query = ""
    context_text = ""
    
    if input_method == "Type your question":
        user_query = st.text_area(
            "Enter your question:",
            placeholder="What information are you looking for?",
            height=100
        )
        
    elif input_method == "Use document context":
        context_text = st.text_area(
            "Paste document text here:",
            placeholder="Paste the document content you want to query about...",
            height=200
        )
        user_query = st.text_input(
            "Ask about the document:",
            placeholder="What do you want to know about this document?"
        )
        
    elif input_method == "Upload & Ask":
        if uploaded_texts:
            selected_file = st.selectbox(
                "Select uploaded document:",
                list(uploaded_texts.keys())
            )
            if selected_file:
                context_text = uploaded_texts[selected_file]
                st.text_area(
                    "Document preview:",
                    value=context_text[:500] + "..." if len(context_text) > 500 else context_text,
                    height=150,
                    disabled=True
                )
                user_query = st.text_input(
                    "Ask about this document:",
                    placeholder="What do you want to know about this document?"
                )
        else:
            st.info("👆 Please upload documents in the sidebar first")
    
    # Query button
    if st.button("🔍 Search", type="primary", disabled=not user_query):
        if index is None or chunks is None:
            st.error("❌ Vector store not loaded. Please check your index files.")
            return
            
        with st.spinner("Processing your query..."):
            try:
                response, matched_chunks, parsed = process_user_query(user_query, model, index, chunks, context_text)
                
                if response:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "query": user_query,
                        "response": response,
                        "chunks": matched_chunks,
                        "parsed": parsed
                    })
                    
                    st.success("✅ Query processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                st.exception(e)
    
    # Chat history in sidebar
    if st.session_state.chat_history:
        with st.sidebar:
            st.header("📜 Recent Queries")
            for i, item in enumerate(reversed(st.session_state.chat_history[-3:]), 1):
                with st.expander(f"Query {len(st.session_state.chat_history) - i + 1}"):
                    st.write(f"**Q:** {item['query'][:50]}...")
            
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()
