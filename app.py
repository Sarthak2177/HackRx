import sys
import os
import streamlit as st
import tempfile
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.chunk_utils import load_chunks
from utils.dynamic_decision import DynamicDecisionEngine # Updated import
from utils.query_parser import parse_query, extract_entities_summary # Enhanced imports

# Import your existing extraction functions
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import docx
import extract_msg

# Comprehensive fix for Streamlit + PyTorch compatibility
# (This needs to be at the very top of app.py for effect)
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

# Initialize DynamicDecisionEngine (cache this as well)
@st.cache_resource
def load_decision_engine():
    return DynamicDecisionEngine()

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

# Enhanced response formatting for Streamlit to handle new structured output
def format_final_response(response_json, chunk_results):
    """Format and display final response with structured decision, amount, and justification"""
    st.markdown("### 🎯 Analysis Results")
    
    try:
        parsed_response = response_json if isinstance(response_json, dict) else json.loads(response_json)
    except json.JSONDecodeError:
        st.error("❌ Failed to parse response")
        st.code(response_json)
        return
    
    # Handle different response formats
    if 'questions_analysis' in parsed_response:
        # Multiple questions format
        st.markdown("### 📋 Questions Analysis")
        
        for i, qa in enumerate(parsed_response['questions_analysis'], 1):
            with st.expander(f"📝 Question {i}: {qa.get('question', 'N/A')[:100]}..."):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if 'decision' in qa:
                        if qa['decision'].lower() == 'approved':
                            st.success(f"**Decision:** {qa['decision']}")
                        elif qa['decision'].lower() == 'rejected':
                            st.error(f"**Decision:** {qa['decision']}")
                        else:
                            st.info(f"**Decision:** {qa['decision']}")
                    
                    if 'answer' in qa:
                        st.markdown("**Answer:**")
                        st.write(qa['answer'])
                    
                    if 'justification' in qa:
                        st.markdown("**Justification:**")
                        st.write(qa['justification'])
                
                with col2:
                    if 'amount' in qa:
                        st.metric("Amount", qa['amount'])
                    
                    if 'confidence' in qa:
                        confidence = qa['confidence']
                        st.metric("Confidence", confidence)
                    
                    if 'referenced_clauses' in qa:
                        st.markdown("**Referenced Clauses:**")
                        for clause in qa['referenced_clauses']:
                            st.code(clause, language=None)
    
    elif 'scenario_analysis' in parsed_response:
        # Vague query scenario analysis format
        scenario = parsed_response['scenario_analysis']
        
        # Main scenario analysis
        st.markdown("### 🔍 Scenario Analysis")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**Scenario:**")
            st.write(scenario.get('scenario', 'N/A'))
            
            if 'decision' in scenario:
                if scenario['decision'].lower() == 'approved':
                    st.success(f"**Decision:** {scenario['decision']}")
                elif scenario['decision'].lower() == 'rejected':
                    st.error(f"**Decision:** {scenario['decision']}")
                else:
                    st.info(f"**Decision:** {scenario['decision']}")
            
            st.markdown("**Justification:**")
            st.write(scenario.get('justification', 'N/A'))
        
        with col2:
            if 'amount' in scenario:
                st.metric("Coverage Amount", scenario['amount'])
            
            if 'confidence' in scenario:
                st.metric("Confidence Level", scenario['confidence'])
        
        with col3:
            if 'referenced_clauses' in scenario:
                st.markdown("**Referenced Clauses:**")
                for clause in scenario['referenced_clauses']:
                    st.code(clause, language=None)
        
        # Relevant questions answered
        if 'relevant_questions_answered' in parsed_response:
            st.markdown("### ❓ Relevant Policy Questions")
            
            for i, rqa in enumerate(parsed_response['relevant_questions_answered'], 1):
                with st.expander(f"Question {i}: {rqa.get('question', 'N/A')[:80]}..."):
                    st.markdown("**Question:**")
                    st.write(rqa.get('question', 'N/A'))
                    
                    st.markdown("**Answer:**")
                    st.write(rqa.get('answer', 'N/A'))
                    
                    if 'referenced_clause' in rqa:
                        st.markdown("**Referenced Clause:**")
                        st.code(rqa['referenced_clause'], language=None)
    
    else:
        # Single decision format (backward compatibility)
        tab1, tab2, tab3 = st.tabs(["Decision Summary", "Justification", "Sources"])
        
        with tab1:
            # Display main decision
            if 'decision' in parsed_response:
                if parsed_response['decision'].lower() == 'approved':
                    st.success(f"**Decision:** {parsed_response['decision']}")
                elif parsed_response['decision'].lower() == 'rejected':
                    st.error(f"**Decision:** {parsed_response['decision']}")
                else:
                    st.info(f"**Decision:** {parsed_response['decision']}")
            
            # Display amount
            if 'amount' in parsed_response:
                st.metric("Amount", parsed_response['amount'])
            
            # Display confidence
            if 'confidence' in parsed_response:
                confidence = parsed_response['confidence']
                st.metric("Confidence Level", confidence)
        
        with tab2:
            # Display justification
            if 'justification' in parsed_response:
                st.markdown("**Justification:**")
                st.write(parsed_response['justification'])
            
            # Display referenced clauses
            if 'referenced_clauses' in parsed_response:
                st.markdown("**Referenced Clauses:**")
                for clause in parsed_response['referenced_clauses']:
                    st.code(clause, language=None)
        
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
                    if len(chunk_results) > 0:
                        avg_score = sum(r['similarity_score'] for r in chunk_results) / len(chunk_results)
                        st.metric("Avg. Score", f"{avg_score:.3f}")
                    else:
                        st.metric("Avg. Score", "N/A")

# Enhanced process query function
def process_user_query(user_input, model, index, chunks, decision_engine, context_text=""):
    """Enhanced process user query with better presentation"""
    st.write(f"💬 **User Query:** {user_input}")
    
    # If we have context text, create a combined query for search
    search_query = f"{user_input}\n\nContext: {context_text}" if context_text else user_input
    
    # 1. Parse details with enhanced entity extraction
    parsed_query_details = parse_query(user_input) # Get parsed details
    
    with st.expander("🔍 Query Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Extracted Entities:**")
            entities_summary = extract_entities_summary(parsed_query_details)
            st.info(entities_summary)
        
        with col2:
            st.markdown("**Raw Parsed Data:**")
            st.json(parsed_query_details)
    
    if all(value is None for value in parsed_query_details.values()):
        st.info("⚠️ No structured entities detected, analyzing as free-form query.")

    # 2. Enhanced semantic search with scores
    chunk_results = search_similar_chunks_with_scores(search_query, model, index, chunks, top_k=8)
    
    # Add context text as first chunk if provided (and not already highly relevant)
    if context_text:
        is_context_in_chunks = any(context_text in res['chunk'] for res in chunk_results)
        if not is_context_in_chunks:
            chunk_results.insert(0, {
                'rank': 0,
                'chunk': context_text,
                'similarity_score': 1.0,
                'relevance': 'High'
            })
    
    # 3. Display chunks nicely
    display_chunks_nicely(chunk_results)
    
    # 4. Get LLM decision using the enhanced DynamicDecisionEngine
    matched_chunks = [result['chunk'] for result in chunk_results]
    
    # Pass parsed_query_details to the decision engine
    json_response_str = decision_engine.make_decision_from_context(user_input, parsed_query_details, matched_chunks)
    
    try:
        response_json = json.loads(json_response_str)
    except json.JSONDecodeError:
        st.error("❌ Failed to parse LLM response as JSON. Raw response:")
        st.code(json_response_str)
        return None, [], parsed_query_details

    # 5. Format and display final response with enhanced formatting
    format_final_response(response_json, chunk_results)
    
    return response_json, matched_chunks, parsed_query_details

# Main UI
def main():
    st.title("🔍 Document Query Assistant")
    st.markdown("Ask questions about your documents or upload new ones for analysis!")
    
    # Load models at startup
    model, index, chunks = load_models()
    decision_engine = load_decision_engine()
    
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
        
        # Add enhanced example queries
        st.header("💡 Example Queries")
        st.markdown("""
        **📝 Explicit Questions:**
        - What are the eligibility criteria for insurance?
        - What is the age limit for policy applications?
        - What documents are required for claims?
        - What are the premium rates for different age groups?
        
        **🎯 Vague Scenarios:**
        - **46-year-old male, knee surgery in Pune, 3-month-old insurance policy**
        - **Senior citizen diabetes treatment urgent Mumbai 2 years policy**
        - **Female 35 pregnancy delivery family floater**
        - **Emergency cardiac surgery 60 years old existing policy**
        
        **📄 PDF Analysis:**
        - Upload a PDF and type "answer questions" to get all PDF questions answered
        - Upload a policy document and ask specific queries about it
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
            "Enter your question or scenario:",
            placeholder="Examples:\n• What is the waiting period for surgeries?\n• 46-year-old male, knee surgery in Pune, 3-month-old insurance policy\n• What are the age limits and coverage amounts?",
            height=120
        )
        
    elif input_method == "Use document context":
        context_text = st.text_area(
            "Paste document text here:",
            placeholder="Paste the document content you want to query about...",
            height=200
        )
        user_query = st.text_input(
            "Ask about the document:",
            placeholder="What do you want to know about this document? E.g., 'Is this procedure covered?' or 'Answer all questions'"
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
                    placeholder="Examples: 'Answer all questions', 'What procedures are covered?', 'Is cardiac surgery covered for 60-year-old?'"
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
                response, matched_chunks, parsed = process_user_query(user_query, model, index, chunks, decision_engine, context_text)
                
                if response:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "query": user_query,
                        "response": response,
                        "chunks": matched_chunks,
                        "parsed": parsed,
                        "context": context_text if context_text else None
                    })
                    
                    st.success("✅ Query processed successfully!")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.exception(e)
    
    # Enhanced chat history in sidebar
    if st.session_state.chat_history:
        with st.sidebar:
            st.header("📜 Recent Queries")
            for i, item in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
                with st.expander(f"Query {len(st.session_state.chat_history) - i + 1}"):
                    st.write(f"**Q:** {item['query'][:80]}...")
                    
                    # Show query type
                    if item.get('parsed', {}).get('query_type'):
                        st.caption(f"Type: {item['parsed']['query_type'].replace('_', ' ').title()}")
                    
                    # Show extracted entities if any
                    if item.get('parsed'):
                        entities = extract_entities_summary(item['parsed'])
                        if entities != "No structured entities detected":
                            st.caption(f"Entities: {entities[:60]}...")
            
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Footer with system status
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        if index is not None and chunks is not None:
            st.success(f"✅ System Ready ({len(chunks)} chunks loaded)")
        else:
            st.error("❌ System Not Ready")

if __name__ == "__main__":
    main()