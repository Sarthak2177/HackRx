import json
import re
from typing import List, Dict, Any
from collections import defaultdict
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables for API key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class DynamicDecisionEngine:
    def __init__(self):
        self.learned_rules = {}
        self.document_patterns = defaultdict(list)

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
        """Extract questions from document text"""
        # Split by sentences and find those ending with question marks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        questions = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence.endswith('?') and len(sentence) > 10:
                questions.append(sentence)
        
        return questions

    def _is_vague_query(self, user_input: str) -> bool:
        """Determine if the input is a vague sentence rather than explicit questions"""
        # Check if input contains question marks
        has_questions = '?' in user_input
        
        # Check if it's a structured sentence with entities (age, procedure, location, etc.)
        has_age = re.search(r'\d+[-\s]?(?:year|yr)', user_input, re.IGNORECASE)
        has_procedure = re.search(r'(surgery|treatment|operation|procedure|therapy|hospitalization)', user_input, re.IGNORECASE)
        has_location = re.search(r'in\s+[A-Z][a-z]+', user_input)
        has_policy_duration = re.search(r'\d+[-\s]?(?:month|year)s?\s*(?:policy|old)', user_input, re.IGNORECASE)
        
        # If it has structured entities but no questions, it's likely vague
        if (has_age or has_procedure or has_location or has_policy_duration) and not has_questions:
            return True
        
        # Check for common vague patterns
        vague_patterns = [
            r'\d+\s*year.*?(surgery|treatment|procedure)',
            r'(surgery|procedure).*?\d+\s*month',
            r'\w+\s+(surgery|treatment).*?(covered|eligible)',
        ]
        
        for pattern in vague_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                return True
        
        return False

    def make_decision_from_context(self, user_input: str, parsed_query_details: Dict[str, Any], chunks: List[str]) -> str:
        if not chunks:
            return json.dumps({
                "prompt": user_input,
                "decision": "Cannot Determine",
                "amount": "N/A",
                "justification": "No relevant document clauses found to make a decision.",
                "confidence": "Low"
            })

        # Prepare context with numbered clauses
        context_with_lines = []
        for i, chunk in enumerate(chunks):
            lines = re.split(r'(?<=[.!?])\s+', chunk.strip())
            for j, line in enumerate(lines):
                if line.strip():
                    context_with_lines.append(f"Clause {i+1}.{j+1}: {line.strip()}")
        full_context = "\n\n".join(context_with_lines)

        # Extract questions from the document
        document_questions = []
        for chunk in chunks:
            document_questions.extend(self._detect_questions_in_text(chunk))

        # Determine the type of query
        is_vague_query = self._is_vague_query(user_input)
        explicit_questions = [q.strip() for q in re.split(r'(?<=[?])\s+', user_input) if '?' in q]
        
        # Handle different scenarios
        if explicit_questions:
            # User asked explicit questions
            return self._handle_explicit_questions(user_input, explicit_questions, parsed_query_details, full_context)
        elif is_vague_query:
            # User provided a vague sentence, analyze against all document questions
            return self._handle_vague_query(user_input, parsed_query_details, full_context, document_questions)
        elif document_questions:
            # No explicit query, answer all document questions
            return self._handle_document_questions(user_input, document_questions, full_context)
        else:
            # Fallback case
            return self._handle_general_query(user_input, parsed_query_details, full_context)

    def _handle_explicit_questions(self, user_input: str, questions: List[str], parsed_query_details: Dict[str, Any], full_context: str) -> str:
        """Handle explicit questions from user"""
        system_msg = """
You are an expert insurance policy assistant. You will receive explicit questions and policy clauses.

For each question, provide a JSON response with:
{
  "questions_analysis": [
    {
      "question": "...",
      "decision": "Approved | Rejected | Requires More Information | Not Applicable | Cannot Determine",
      "amount": "Specific amount if mentioned or N/A",
      "justification": "Detailed explanation referencing specific Clause X.Y with reasoning",
      "confidence": "High | Medium | Low",
      "referenced_clauses": ["Clause X.Y", "Clause A.B"]
    }
  ]
}

Always reference specific clauses and provide detailed justification.
"""

        user_msg = f"""
User Input: {user_input}

Explicit Questions:
{json.dumps(questions, indent=2)}

Parsed Query Details:
{json.dumps(parsed_query_details, indent=2) if any(parsed_query_details.values()) else "None"}

Policy Clauses:
{full_context}

Please analyze each question against the policy clauses and provide detailed decisions with justifications.
"""

        return self._call_llm(system_msg, user_msg, {"type": "json_object"})

    def _handle_vague_query(self, user_input: str, parsed_query_details: Dict[str, Any], full_context: str, document_questions: List[str]) -> str:
        """Handle vague queries by analyzing them against document context and questions"""
        system_msg = """
You are an expert insurance policy assistant. You received a vague query (like "46-year-old male, knee surgery in Pune, 3-month-old insurance policy") and need to analyze it comprehensively.

Provide a JSON response with:
{
  "scenario_analysis": {
    "scenario": "Brief description of the scenario",
    "decision": "Approved | Rejected | Requires More Information | Cannot Determine",
    "amount": "Coverage amount if determinable or N/A",
    "justification": "Detailed analysis referencing specific Clause X.Y with reasoning",
    "confidence": "High | Medium | Low",
    "referenced_clauses": ["Clause X.Y", "Clause A.B"]
  },
  "relevant_questions_answered": [
    {
      "question": "Relevant question from policy document",
      "answer": "Answer based on the scenario",
      "referenced_clause": "Clause X.Y"
    }
  ]
}

Analyze the vague query against all policy clauses and answer relevant questions that apply to the scenario.
"""

        user_msg = f"""
Vague Query: {user_input}

Parsed Details:
{json.dumps(parsed_query_details, indent=2)}

Document Questions (for reference):
{json.dumps(document_questions[:10], indent=2)}

Policy Clauses:
{full_context}

Analyze this scenario comprehensively and determine coverage, eligibility, and any other relevant aspects.
"""

        return self._call_llm(system_msg, user_msg, {"type": "json_object"})

    def _handle_document_questions(self, user_input: str, document_questions: List[str], full_context: str) -> str:
        """Handle case where user wants answers to document questions"""
        system_msg = """
You are an expert insurance policy assistant. Answer all the questions found in the policy document based on the policy clauses.

Provide a JSON response with:
{
  "questions_analysis": [
    {
      "question": "...",
      "answer": "Detailed answer based on policy clauses",
      "referenced_clauses": ["Clause X.Y", "Clause A.B"],
      "confidence": "High | Medium | Low"
    }
  ]
}

Answer each question comprehensively with proper clause references.
"""

        user_msg = f"""
User Request: {user_input}

Document Questions to Answer:
{json.dumps(document_questions, indent=2)}

Policy Clauses:
{full_context}

Please answer all the questions found in the document based on the policy clauses.
"""

        return self._call_llm(system_msg, user_msg, {"type": "json_object"})

    def _handle_general_query(self, user_input: str, parsed_query_details: Dict[str, Any], full_context: str) -> str:
        """Fallback for general queries"""
        system_msg = """
You are an expert insurance policy assistant. Respond with a JSON object:
{
  "prompt": "string",
  "decision": "Approved | Rejected | Requires More Information | Not Applicable | Cannot Determine",
  "amount": "string",
  "justification": "Detailed explanation with reference to specific Clause X.Y",
  "confidence": "High | Medium | Low",
  "referenced_clauses": ["Clause X.Y"]
}

Always provide detailed justification with specific clause references.
"""

        user_msg = f"""
Prompt: {user_input}

Parsed Query Details:
{json.dumps(parsed_query_details, indent=2) if any(parsed_query_details.values()) else "None"}

Policy Clauses:
{full_context}
"""

        return self._call_llm(system_msg, user_msg, {"type": "json_object"})

    def _call_llm(self, system_msg: str, user_msg: str, response_format) -> str:
        """Helper method to call the LLM with error handling"""
        try:
            chat_completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.3,
                response_format=response_format
            )
            return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            return json.dumps({
                "prompt": user_msg.split('\n')[0],
                "decision": "Error",
                "amount": "N/A",
                "justification": f"LLM error: {str(e)}",
                "confidence": "Low",
                "error": str(e)
            })