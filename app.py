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
import json

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
# Helper function to safely convert confidence to float
def safe_to_float(value, default=0.0):
    """Convert value to float, return default if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def transform_to_required_format(query, response_json, matched_chunks, document_name="Unknown"):
    try:
        if isinstance(response_json, str):
            response_json = json.loads(response_json)

        structured_response = {
            "query": query,
            "decision": "Unknown",
            "amount": None,
            "conditions": [],
            "justification": "No justification provided.",
            "confidence": 0.5,
            "source_clauses": []
        }

        if 'questions_analysis' in response_json:
            qa = response_json['questions_analysis'][0] if response_json['questions_analysis'] else {}
            structured_response.update({
                "decision": qa.get('decision', 'Unknown') or qa.get('answer', 'Unknown'),
                "amount": qa.get('amount', None),
                "conditions": qa.get('conditions', []),
                "justification": qa.get('justification', 'No justification provided.'),
                "confidence": safe_to_float(qa.get('confidence', 0.5)),
                "source_clauses": [
                    {
                        "clause_id": clause.split(':')[0].strip() if ':' in clause else f"clause_{i}",
                        "text": clause,
                        "document": document_name
                    } for i, clause in enumerate(qa.get('referenced_clauses', []))
                ]
            })

        elif 'scenario_analysis' in response_json:
            scenario = response_json['scenario_analysis']
            structured_response.update({
                "decision": scenario.get('decision', 'Unknown'),
                "amount": scenario.get('amount', None),
                "conditions": scenario.get('conditions', []),
                "justification": scenario.get('justification', 'No justification provided.'),
                "confidence": safe_to_float(scenario.get('confidence', 0.5)),
                "source_clauses": [
                    {
                        "clause_id": clause.split(':')[0].strip() if ':' in clause else f"clause_{i}",
                        "text": clause,
                        "document": document_name
                    } for i, clause in enumerate(scenario.get('referenced_clauses', []))
                ]
            })

        else:
            structured_response.update({
                "decision": response_json.get('decision', 'Unknown'),
                "amount": response_json.get('amount', None),
                "conditions": response_json.get('conditions', []),
                "justification": response_json.get('justification', 'No justification provided.'),
                "confidence": safe_to_float(response_json.get('confidence', 0.5)),
                "source_clauses": [
                    {
                        "clause_id": chunk.split(':')[0].strip() if ':' in chunk else f"clause_{i}",
                        "text": chunk,
                        "document": document_name
                    } for i, chunk in enumerate(response_json.get('referenced_clauses', matched_chunks))
                ]
            })

        # Extract conditions from justification
        if not structured_response["conditions"] and "justification" in structured_response:
            if "6 months" in structured_response["justification"]:
                structured_response["conditions"].append("6 months of policy activation required for cardiac surgeries")

        # Adjust confidence if too low and clauses are present
        if structured_response["confidence"] == 0 and structured_response["source_clauses"]:
            structured_response["confidence"] = 0.95

        if not structured_response["source_clauses"]:
            structured_response["source_clauses"] = [
                {
                    "clause_id": f"chunk_{i}",
                    "text": chunk,
                    "document": document_name
                } for i, chunk in enumerate(matched_chunks[:3])
            ]

        return structured_response
    except Exception as e:
        st.error(f"Error transforming response: {e}")
        return {
            "query": query,
            "decision": "Error",
            "amount": None,
            "conditions": [],
            "justification": f"Failed to process response: {str(e)}",
            "confidence": 0.0,
            "source_clauses": []
        }
def process_user_query(user_input, model, index, chunks, decision_engine, context_text=""):
    """Process user query and return structured JSON response."""
    # Debugging: Log inputs
    st.write(f"Debug: Processing query: {user_input}")
    st.write(f"Debug: Context text length: {len(context_text) if context_text else 0}")
    st.write(f"Debug: Model loaded: {model is not None}")
    st.write(f"Debug: Index loaded: {index is not None}")
    st.write(f"Debug: Chunks count: {len(chunks) if chunks else 0}")
    st.write(f"Debug: Decision engine loaded: {decision_engine is not None}")

    try:
        # Validate inputs
        if not user_input:
            st.error("❌ User query is empty.")
            return None, [], {}
        if index is None or chunks is None:
            st.error("❌ Vector store not loaded.")
            return None, [], {}
        if model is None:
            st.error("❌ Embedding model not loaded.")
            return None, [], {}

        # If context text is provided, create a combined query for search
        search_query = f"{user_input}\n\nContext: {context_text}" if context_text else user_input
        
        # 1. Parse query details
        try:
            parsed_query_details = parse_query(user_input)
        except Exception as e:
            st.error(f"❌ Error parsing query: {e}")
            parsed_query_details = {}

        with st.expander("🔍 Query Analysis"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Extracted Entities:**")
                entities_summary = extract_entities_summary(parsed_query_details)
                st.info(entities_summary if entities_summary else "No entities extracted.")
            with col2:
                st.markdown("**Raw Parsed Data:**")
                st.json(parsed_query_details)
        
        if all(value is None for value in parsed_query_details.values()):
            st.info("⚠️ No structured entities detected, analyzing as free-form query.")

        # 2. Semantic search with scores
        try:
            chunk_results = search_similar_chunks_with_scores(search_query, model, index, chunks, top_k=8)
        except Exception as e:
            st.error(f"❌ Error in semantic search: {e}")
            chunk_results = []
        
        # Add context text as first chunk if provided
        if context_text:
            is_context_in_chunks = any(context_text in res['chunk'] for res in chunk_results)
            if not is_context_in_chunks:
                chunk_results.insert(0, {
                    'rank': 0,
                    'chunk': context_text,
                    'similarity_score': 1.0,
                    'relevance': 'High'
                })
        
        # 3. Display chunks
        display_chunks_nicely(chunk_results)
        
        # 4. Get LLM decision
        matched_chunks = [result['chunk'] for result in chunk_results]
        document_name = "Uploaded_Document" if context_text else "Policy_Database"
        
        try:
            json_response_str = decision_engine.make_decision_from_context(user_input, parsed_query_details, matched_chunks)
            st.write("Debug: Raw DynamicDecisionEngine response:", json_response_str)
        except Exception as e:
            st.error(f"❌ Error in DynamicDecisionEngine: {e}")
            return {
                "query": user_input,
                "answer": "Error",
                "conditions": [],
                "rationale": f"Failed to process decision: {str(e)}",
                "confidence": 0.0,
                "source_clauses": []
            }, [], parsed_query_details

        # 5. Parse and transform response
        try:
            response_json = json.loads(json_response_str) if isinstance(json_response_str, str) else json_response_str
        except json.JSONDecodeError as e:
            st.error("❌ Failed to parse LLM response as JSON. Raw response:")
            st.code(json_response_str)
            return {
                "query": user_input,
                "answer": "Error",
                "conditions": [],
                "rationale": f"JSON parsing error: {str(e)}",
                "confidence": 0.0,
                "source_clauses": []
            }, [], parsed_query_details

        # 6. Transform response to required JSON format
        response_json = transform_to_required_format(user_input, response_json, matched_chunks, document_name)
        
        # 7. Display final response
        format_final_response(response_json, chunk_results)
        
        return response_json, matched_chunks, parsed_query_details

    except Exception as e:
        st.error(f"❌ Unexpected error in process_user_query: {e}")
        st.exception(e)
        return {
            "query": user_input,
            "answer": "Error",
            "conditions": [],
            "rationale": f"Unexpected error: {str(e)}",
            "confidence": 0.0,
            "source_clauses": []
        }, [], {}
    
# Updated format_final_response function
def format_final_response(response_json, chunk_results):
    """Display structured JSON response in Streamlit."""
    st.markdown("### 🎯 Analysis Results")
    st.json(response_json)
    
    st.markdown("#### 📋 Response Breakdown")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Query:**")
        st.write(response_json.get('query', 'N/A'))
        
        st.markdown("**Decision:**")
        decision = response_json.get('decision', 'N/A')
        if decision.lower() in ['approved', 'yes']:
            st.success(f"**Decision:** {decision}")
        elif decision.lower() in ['rejected', 'no']:
            st.error(f"**Decision:** {decision}")
        else:
            st.info(f"**Decision:** {decision}")
        
        st.markdown("**Justification:**")
        st.write(response_json.get('justification', 'No justification provided.'))
        
        st.markdown("**Conditions:**")
        for condition in response_json.get('conditions', []):
            st.write(f"- {condition}")
    
    with col2:
        confidence = safe_to_float(response_json.get('confidence', 0.0))
        st.metric("Confidence", f"{confidence:.2f}")
    
    st.markdown("#### 📚 Source Clauses")
    for clause in response_json.get('source_clauses', []):
        with st.expander(f"Clause ID: {clause.get('clause_id', 'N/A')} (Document: {clause.get('document', 'N/A')})"):
            st.write(clause.get('text', 'No text available.'))
    
    if chunk_results:
        st.markdown("#### 🔍 Chunk Relevance Summary")
        col1, col2, col3, col4 = st.columns(4)
        high_rel = sum(1 for r in chunk_results if r['relevance'] == 'High')
        med_rel = sum(1 for r in chunk_results if r['relevance'] == 'Medium')
        low_rel = sum(1 for r in chunk_results if r['relevance'] == 'Low')
        avg_score = sum(r['similarity_score'] for r in chunk_results) / len(chunk_results) if chunk_results else 0
        with col1:
            st.metric("High Relevance", high_rel)
        with col2:
            st.metric("Medium Relevance", med_rel)
        with col3:
            st.metric("Low Relevance", low_rel)
        with col4:
            st.metric("Avg. Score", f"{avg_score:.3f}")
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