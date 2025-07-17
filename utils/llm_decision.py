from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def make_decision_from_context(user_input, chunks):
    context = "\n---\n".join(chunks)

    prompt = f"""
You are an insurance policy expert.

Given the user's query and the policy clauses, you must choose ONLY ONE of the following replies:

- "Yes, [procedure] is covered under the policy."
- "No, [procedure] is not covered under the policy."
- "Cannot determine based on available policy clauses."

DO NOT add any explanation or justification. Just return the exact sentence that best applies.

User Query: {user_input}

Relevant Policy Clauses:
{context}

Respond now:
"""

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a strict decision-making assistant. Only output one of the allowed responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"
