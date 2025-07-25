# query_llm.py

import ollama
from prompts import MISTRAL_SYSTEM_PROMPT, build_mistral_prompt

def query_mistral_with_clauses(query, clauses):
    prompt = build_mistral_prompt(query, clauses)

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": MISTRAL_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": prompt.strip()}
        ]
    )
    
    return response['message']['content']
