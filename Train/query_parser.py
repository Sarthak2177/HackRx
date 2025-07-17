import re

def parse_query(user_query):
    # Example: "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    parsed = {
        "age": None,
        "procedure": None,
        "location": None,
        "policy_duration_months": None
    }

    # Extract age
    age_match = re.search(r"(\d+)[ -]?year", user_query)
    if age_match:
        parsed["age"] = int(age_match.group(1))

    # Extract procedure
    proc_match = re.search(r"(surgery|treatment|operation|transplant|procedure|therapy|injury|fracture|hospitalization)", user_query, re.IGNORECASE)
    if proc_match:
        parsed["procedure"] = proc_match.group(0).strip().lower()

    # Extract location
    loc_match = re.search(r"in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", user_query)
    if loc_match:
        parsed["location"] = loc_match.group(1)

    # Extract policy duration
    pol_match = re.search(r"(\d+)[ -]?(month|year)", user_query)
    if pol_match:
        num = int(pol_match.group(1))
        unit = pol_match.group(2)
        parsed["policy_duration_months"] = num * 12 if unit == "year" else num

    return parsed
