import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.chunk_utils import load_chunks
from utils.llm_decision import make_decision_from_context
from utils.query_parser import parse_query

# Load model and index
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)

INDEX_PATH = "vector_store/faiss_index.bin"
CHUNK_PATH = "vector_store/chunks.pkl"

print("\n🔍 Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
chunks = load_chunks(CHUNK_PATH)

assert index.ntotal == len(chunks), "Mismatch between index and chunks"

# Enhanced semantic search with similarity scores
def search_similar_chunks_with_scores(user_query, top_k=5):
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

# Enhanced display function for chunks
def display_chunks_nicely(chunk_results):
    print("\n" + "="*80)
    print("📚 KNOWLEDGE BASE REFERENCES")
    print("="*80)
    
    for result in chunk_results:
        print(f"\n📖 Reference #{result['rank']} | Relevance: {result['relevance']} | Score: {result['similarity_score']:.3f}")
        print("-" * 60)
        
        # Show chunk content with better formatting
        chunk_text = result['chunk']
        if len(chunk_text) > 300:
            chunk_text = chunk_text[:300] + "..."
        
        # Add indentation for better readability
        formatted_chunk = '\n'.join(['    ' + line for line in chunk_text.split('\n')])
        print(f"{formatted_chunk}")
        print("-" * 60)

# Enhanced response formatting
def format_final_response(response, chunk_results):
    print("\n" + "="*80)
    print("🎯 FINAL DECISION & REASONING")
    print("="*80)
    
    try:
        parsed_response = json.loads(response)
        
        # Display main decision
        if 'decision' in parsed_response:
            print(f"\n✅ DECISION: {parsed_response['decision']}")
        
        # Display reasoning
        if 'reasoning' in parsed_response:
            print(f"\n🧠 REASONING:")
            reasoning = parsed_response['reasoning']
            formatted_reasoning = '\n'.join(['    ' + line for line in reasoning.split('\n')])
            print(f"{formatted_reasoning}")
        
        # Display confidence
        if 'confidence' in parsed_response:
            print(f"\n📊 CONFIDENCE: {parsed_response['confidence']}")
        
        # Display eligibility details if present
        if 'eligibility' in parsed_response:
            print(f"\n✔️ ELIGIBILITY:")
            eligibility = parsed_response['eligibility']
            if isinstance(eligibility, dict):
                for key, value in eligibility.items():
                    print(f"    • {key}: {value}")
            else:
                print(f"    {eligibility}")
        
        # Display any additional fields
        for key, value in parsed_response.items():
            if key not in ['decision', 'reasoning', 'confidence', 'eligibility']:
                print(f"\n📌 {key.upper()}:")
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"    • {k}: {v}")
                elif isinstance(value, list):
                    for item in value:
                        print(f"    • {item}")
                else:
                    print(f"    {value}")
                
    except json.JSONDecodeError:
        print(f"\n📝 RESPONSE:")
        formatted_response = '\n'.join(['    ' + line for line in response.split('\n')])
        print(f"{formatted_response}")
    
    # Show source summary
    print(f"\n📋 SOURCES SUMMARY:")
    high_rel = sum(1 for r in chunk_results if r['relevance'] == 'High')
    med_rel = sum(1 for r in chunk_results if r['relevance'] == 'Medium')
    low_rel = sum(1 for r in chunk_results if r['relevance'] == 'Low')
    
    print(f"    🔴 High Relevance: {high_rel} sources")
    print(f"    🟡 Medium Relevance: {med_rel} sources")
    print(f"    🔵 Low Relevance: {low_rel} sources")
    
    avg_score = sum(r['similarity_score'] for r in chunk_results) / len(chunk_results)
    print(f"    📈 Average Similarity Score: {avg_score:.3f}")

# Main processing function
def process_user_query(user_input):
    print(f"\n💬 User Query: {user_input}")
    print("="*50)
    
    # 1. Parse details
    parsed = parse_query(user_input)
    print(f"🔍 Parsed Query Details: {parsed}")
    
    if all(value is None for value in parsed.values()):
        print("❗ Couldn't extract meaningful information. Please try a better query.\n")
        return None
    
    # 2. Enhanced semantic search with scores
    chunk_results = search_similar_chunks_with_scores(user_input, top_k=5)
    
    # 3. Display chunks nicely
    display_chunks_nicely(chunk_results)
    
    # 4. Get LLM decision
    matched_chunks = [result['chunk'] for result in chunk_results]
    response = make_decision_from_context(user_input, matched_chunks)
    
    # 5. Format and display final response
    format_final_response(response, chunk_results)
    
    return response

# Enhanced main loop with better UI
def main():
    print("\n" + "="*80)
    print("🔍 DOCUMENT QUERY ASSISTANT")
    print("="*80)
    print("Ask questions about your documents. Type 'exit' or 'quit' to stop.")
    print("Examples:")
    print("  • What are the eligibility criteria for insurance?")
    print("  • What is the age limit for policy applications?")
    print("  • What documents are required for claims?")
    print("="*80)
    
    query_count = 0
    
    while True:
        query_count += 1
        print(f"\n🎯 Query #{query_count}")
        user_query = input("📝 Enter your query: ").strip()
        
        if user_query.lower() in ("exit", "quit", "q"):
            print("\n👋 Thank you for using the Document Query Assistant!")
            break
        
        if not user_query:
            print("❗ Please enter a valid query.")
            continue
        
        try:
            response = process_user_query(user_query)
            
            if not response:
                continue
            
            # Ask if user wants to continue
            print("\n" + "="*80)
            continue_choice = input("❓ Would you like to ask another question? (y/n): ").strip().lower()
            if continue_choice in ('n', 'no'):
                print("\n👋 Thank you for using the Document Query Assistant!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()
