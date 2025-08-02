import json
import re
from typing import List, Dict, Any
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

@lru_cache()
def get_embed_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

class DynamicDecisionEngine:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-8b-8192"
        )

    def query(self, user_query: str, context_chunks: List[str]) -> str:
        context_str = "\n".join(context_chunks)

        prompt = (
            f"You are a helpful assistant.\n"
            f"Use only the information from the provided context to answer.\n"
            f"If the context is not helpful, say you don't have enough information.\n"
            f"\nContext:\n{context_str}\n\nQuestion: {user_query}"
        )

        try:
            response = self.llm.invoke([{"role": "user", "content": prompt}])
            return response.content.strip()
        except Exception as e:
            return json.dumps({
                "prompt": user_query,
                "decision": "Error",
                "amount": "N/A",
                "justification": f"LLM error: {str(e)}",
                "confidence": "Low",
                "error": str(e)
            })

    def extract_rules_from_documents(self, chunks: List[str]) -> Dict[str, Any]:
        clean_chunks = [str(chunk.text) if hasattr(chunk, 'text') else str(chunk) for chunk in chunks]
        return {
            'document_patterns': self._learn_document_patterns(clean_chunks),
            'relationship_rules': self._learn_relationships(clean_chunks),
            'conditional_patterns': self._learn_conditional_patterns(clean_chunks),
            'entity_associations': self._learn_entity_associations(clean_chunks)
        }

    def _learn_document_patterns(self, chunks: List[str]) -> Dict[str, Any]:
        patterns = defaultdict(list)
        for chunk in chunks:
            sentences = re.split(r'[.!?]+', chunk)
            for sentence in sentences:
                sentence = sentence.strip().lower()
                if len(sentence) < 10: continue
                numbers = re.findall(r'\b\d+\b', sentence)
                if numbers:
                    for number in numbers:
                        context = sentence.replace(number, '[NUMBER]')
                        patterns['number_context'].append({'template': context, 'example_value': number, 'original_sentence': sentence})
                for cond_word in ['if', 'when', 'provided', 'subject to', 'unless', 'except']:
                    if cond_word in sentence:
                        patterns['conditional_patterns'].append({'trigger': cond_word, 'context': sentence, 'type': 'condition'})
                tokens = sentence.split()
                for i, word in enumerate(tokens):
                    if word in ['covered', 'eligible', 'included', 'excluded', 'required']:
                        context = tokens[max(0, i-3):min(len(tokens), i+4)]
                        patterns['coverage_patterns'].append({'keyword': word, 'context': ' '.join(context), 'full_sentence': sentence})
        return dict(patterns)

    def _learn_relationships(self, chunks: List[str]) -> Dict[str, List[str]]:
        relationships = defaultdict(list)
        for chunk in chunks:
            sentences = re.split(r'[.!?]+', chunk)
            for sentence in sentences:
                words = re.findall(r'\b\w+\b', sentence.lower())
                for conn_word in ['and', 'or', 'but', 'however', 'therefore', 'because', 'since']:
                    if conn_word in words:
                        idx = words.index(conn_word)
                        before = words[max(0, idx-3):idx]
                        after = words[idx+1:idx+4]
                        if before and after:
                            relationships[conn_word].append({'before': ' '.join(before), 'after': ' '.join(after), 'relationship_type': conn_word})
        return dict(relationships)

    def _learn_conditional_patterns(self, chunks: List[str]) -> List[Dict[str, Any]]:
        conditions = []
        for chunk in chunks:
            for match in re.finditer(r'if\s+([^,]+),?\s*then\s+([^.]+)', chunk.lower()):
                conditions.append({'condition': match.group(1).strip(), 'result': match.group(2).strip(), 'type': 'if_then', 'source': chunk[:100] + '...'})
            for match in re.finditer(r'when\s+([^,]+),?\s*([^.]+)', chunk.lower()):
                conditions.append({'trigger': match.group(1).strip(), 'result': match.group(2).strip(), 'type': 'when_then', 'source': chunk[:100] + '...'})
        return conditions

    def _learn_entity_associations(self, chunks: List[str]) -> Dict[str, Dict[str, int]]:
        associations = defaultdict(lambda: defaultdict(int))
        for chunk in chunks:
            words = re.findall(r'\b\w+\b', chunk.lower())
            for i, word in enumerate(words):
                for other in words[max(0, i-5):i] + words[i+1:i+6]:
                    if word != other:
                        associations[word][other] += 1
        return {w: {k: v for k, v in d.items() if v > 1} for w, d in associations.items() if d}

    def _detect_questions_in_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.endswith('?') and len(s) > 10]

    def _is_vague_query(self, user_input: str) -> bool:
        has_questions = '?' in user_input
        has_age = re.search(r'\d+[-\s]?(?:year|yr)', user_input, re.IGNORECASE)
        has_procedure = re.search(r'(surgery|treatment|operation|procedure|therapy|hospitalization)', user_input, re.IGNORECASE)
        has_location = re.search(r'in\s+[A-Z][a-z]+', user_input)
        has_policy_duration = re.search(r'\d+[-\s]?(?:month|year)s?\s*(?:policy|old)', user_input, re.IGNORECASE)
        if (has_age or has_procedure or has_location or has_policy_duration) and not has_questions:
            return True
        vague_patterns = [
            r'\d+\s*year.*?(surgery|treatment|procedure)',
            r'(surgery|procedure).*?\d+\s*month',
            r'\w+\s+(surgery|treatment).*?(covered|eligible)'
        ]
        return any(re.search(p, user_input, re.IGNORECASE) for p in vague_patterns)
