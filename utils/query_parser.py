import re
from typing import Dict, Any, List

def parse_query(user_query: str) -> Dict[str, Any]:
    """Enhanced query parser that handles both structured and vague queries"""
    parsed = {
        "age": None,
        "gender": None,
        "procedure": None,
        "location": None,
        "policy_duration_months": None,
        "policy_type": None,
        "medical_condition": None,
        "query_type": None,  # explicit_questions, vague_scenario, general
        "urgency": None,
        "family_status": None
    }

    # Determine query type
    if '?' in user_query:
        parsed["query_type"] = "explicit_questions"
    elif _is_vague_scenario(user_query):
        parsed["query_type"] = "vague_scenario"
    else:
        parsed["query_type"] = "general"

    # Extract age (various formats)
    age_patterns = [
        r"(\d+)[-\s]?year[-\s]?old",
        r"(\d+)[-\s]?yr[-\s]?old",
        r"age[:\s]*(\d+)",
        r"(\d+)\s*years?\s*of\s*age",
        r"(\d+)\s*years?"
    ]
    
    for pattern in age_patterns:
        age_match = re.search(pattern, user_query, re.IGNORECASE)
        if age_match:
            parsed["age"] = int(age_match.group(1))
            break

    # Extract gender
    gender_match = re.search(r"\b(male|female|man|woman|boy|girl)\b", user_query, re.IGNORECASE)
    if gender_match:
        gender = gender_match.group(1).lower()
        if gender in ['male', 'man', 'boy']:
            parsed["gender"] = "male"
        elif gender in ['female', 'woman', 'girl']:
            parsed["gender"] = "female"

    # Extract procedure/medical terms (expanded list)
    medical_terms = [
        'surgery', 'operation', 'procedure', 'treatment', 'therapy', 'transplant',
        'hospitalization', 'admission', 'consultation', 'diagnosis', 'examination',
        'check-?up', 'scan', 'imaging', 'x-?ray', 'mri', 'ct scan', 'ultrasound',
        'delivery', 'birth', 'pregnancy', 'maternity', 'cataract', 'cancer',
        'chemotherapy', 'radiation', 'dialysis', 'bypass', 'angioplasty',
        'knee replacement', 'hip replacement', 'joint replacement',
        'fracture', 'injury', 'accident', 'emergency', 'icu', 'intensive care',
        'dental', 'orthodontic', 'physiotherapy', 'rehabilitation'
    ]
    
    medical_pattern = r"\b(" + "|".join(medical_terms) + r")\b"
    proc_match = re.search(medical_pattern, user_query, re.IGNORECASE)
    if proc_match:
        parsed["procedure"] = proc_match.group(1).strip().lower()

    # Extract specific medical conditions
    condition_terms = [
        'diabetes', 'hypertension', 'heart disease', 'cardiac', 'kidney disease',
        'liver disease', 'arthritis', 'asthma', 'copd', 'stroke', 'paralysis',
        'mental health', 'depression', 'anxiety', 'epilepsy', 'migraine'
    ]
    
    condition_pattern = r"\b(" + "|".join(condition_terms) + r")\b"
    condition_match = re.search(condition_pattern, user_query, re.IGNORECASE)
    if condition_match:
        parsed["medical_condition"] = condition_match.group(1).strip().lower()

    # Extract location (improved pattern)
    location_patterns = [
        r"in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"at\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    ]
    
    for pattern in location_patterns:
        loc_match = re.search(pattern, user_query)
        if loc_match:
            parsed["location"] = loc_match.group(1)
            break

    # Extract policy duration (various formats)
    duration_patterns = [
        r"(\d+)[-\s]?(month|year)s?[-\s]?old\s*(?:policy|insurance)",
        r"(\d+)[-\s]?(month|year)s?\s*(?:policy|insurance)",
        r"policy\s*(?:of|for)?\s*(\d+)[-\s]?(month|year)s?",
        r"(\d+)[-\s]?(month|year)s?\s*(?:old|existing)\s*(?:policy|coverage)"
    ]
    
    for pattern in duration_patterns:
        pol_match = re.search(pattern, user_query, re.IGNORECASE)
        if pol_match:
            num = int(pol_match.group(1))
            unit = pol_match.group(2).lower()
            parsed["policy_duration_months"] = num * 12 if unit.startswith("year") else num
            break

    # Extract policy type
    policy_types = [
        'health insurance', 'medical insurance', 'life insurance', 'term insurance',
        'endowment', 'ulip', 'mediclaim', 'family floater', 'individual policy',
        'group insurance', 'senior citizen policy', 'critical illness'
    ]
    
    policy_pattern = r"\b(" + "|".join(policy_types) + r")\b"
    policy_match = re.search(policy_pattern, user_query, re.IGNORECASE)
    if policy_match:
        parsed["policy_type"] = policy_match.group(1).strip().lower()

    # Extract urgency indicators
    urgency_terms = ['emergency', 'urgent', 'immediate', 'asap', 'critical', 'severe']
    urgency_pattern = r"\b(" + "|".join(urgency_terms) + r")\b"
    urgency_match = re.search(urgency_pattern, user_query, re.IGNORECASE)
    if urgency_match:
        parsed["urgency"] = urgency_match.group(1).strip().lower()

    # Extract family status
    family_terms = ['family', 'spouse', 'wife', 'husband', 'children', 'dependent', 'parents']
    family_pattern = r"\b(" + "|".join(family_terms) + r")\b"
    family_match = re.search(family_pattern, user_query, re.IGNORECASE)
    if family_match:
        parsed["family_status"] = family_match.group(1).strip().lower()

    return parsed

def _is_vague_scenario(user_query: str) -> bool:
    """Determine if the query is a vague scenario description"""
    # Check for structured information without explicit questions
    has_age = re.search(r'\d+[-\s]?(?:year|yr)', user_query, re.IGNORECASE)
    has_procedure = re.search(r'(surgery|treatment|operation|procedure|therapy)', user_query, re.IGNORECASE)
    has_location = re.search(r'(?:in|at|from)\s+[A-Z][a-z]+', user_query)
    has_policy_duration = re.search(r'\d+[-\s]?(?:month|year)s?\s*(?:policy|old|insurance)', user_query, re.IGNORECASE)
    has_medical_condition = re.search(r'(diabetes|hypertension|heart|kidney|liver|arthritis)', user_query, re.IGNORECASE)
    
    # Count structured elements
    structured_elements = sum([
        bool(has_age), bool(has_procedure), bool(has_location), 
        bool(has_policy_duration), bool(has_medical_condition)
    ])
    
    # If it has 2 or more structured elements and no questions, it's likely vague
    if structured_elements >= 2 and '?' not in user_query:
        return True
    
    # Check for common vague sentence patterns
    vague_patterns = [
        r'\d+\s*year.*?(surgery|treatment|procedure)',
        r'(surgery|procedure).*?\d+\s*month',
        r'\w+\s+(surgery|treatment).*?(covered|eligible|policy)',
        r'(male|female).*?\d+.*?(surgery|treatment)',
        r'\d+.*?(policy|insurance).*?(surgery|treatment|procedure)'
    ]
    
    for pattern in vague_patterns:
        if re.search(pattern, user_query, re.IGNORECASE):
            return True
    
    return False

def extract_entities_summary(parsed_query: Dict[str, Any]) -> str:
    """Create a summary of extracted entities for display"""
    summary_parts = []
    
    if parsed_query.get("age"):
        summary_parts.append(f"Age: {parsed_query['age']}")
    
    if parsed_query.get("gender"):
        summary_parts.append(f"Gender: {parsed_query['gender'].title()}")
    
    if parsed_query.get("procedure"):
        summary_parts.append(f"Procedure: {parsed_query['procedure'].title()}")
    
    if parsed_query.get("medical_condition"):
        summary_parts.append(f"Condition: {parsed_query['medical_condition'].title()}")
    
    if parsed_query.get("location"):
        summary_parts.append(f"Location: {parsed_query['location']}")
    
    if parsed_query.get("policy_duration_months"):
        months = parsed_query['policy_duration_months']
        if months >= 12:
            years = months // 12
            remaining_months = months % 12
            if remaining_months:
                summary_parts.append(f"Policy Duration: {years}yr {remaining_months}mo")
            else:
                summary_parts.append(f"Policy Duration: {years} year{'s' if years > 1 else ''}")
        else:
            summary_parts.append(f"Policy Duration: {months} month{'s' if months > 1 else ''}")
    
    if parsed_query.get("policy_type"):
        summary_parts.append(f"Policy Type: {parsed_query['policy_type'].title()}")
    
    if parsed_query.get("urgency"):
        summary_parts.append(f"Urgency: {parsed_query['urgency'].title()}")
    
    if parsed_query.get("family_status"):
        summary_parts.append(f"Family: {parsed_query['family_status'].title()}")
    
    return " | ".join(summary_parts) if summary_parts else "No structured entities detected"