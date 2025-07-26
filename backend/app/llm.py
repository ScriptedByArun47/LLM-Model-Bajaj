# query_llm.py
import json
import ollama
from prompts import MISTRAL_SYSTEM_PROMPT, build_mistral_prompt

def query_mistral_with_clauses(query, clauses):
    prompt = build_mistral_prompt(query, clauses)

    response = ollama.chat(
        model="llama3:8b",
        messages=[
            {"role": "system", "content": MISTRAL_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": prompt.strip()}
        ]
    )
    
    content = response['message']['content']

    try:
    # Try parsing JSON to confirm validity
        result = json.loads(content)
        return result
    except json.JSONDecodeError:
        return {
            "answer": "The document does not contain any clear or relevant clause to address the query. Please refer to the policy document directly or contact the insurer for further clarification."
        }


