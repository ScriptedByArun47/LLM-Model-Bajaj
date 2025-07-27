# llm.py
import json
import os
# import ollama # Remove this
import google.generativeai as genai # Import GenAI
from prompts import MISTRAL_SYSTEM_PROMPT_TEMPLATE, build_mistral_prompt # Note the change in import from MISTRAL_SYSTEM_PROMPT

# Configure Google Generative AI
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
genai_model = genai.GenerativeModel('gemini-pro') # Initialize the GenAI model

def query_mistral_with_clauses(query, clauses):
    prompt = build_mistral_prompt(query, clauses)

    try:
        # Use the GenAI model to generate content
        response = genai_model.generate_content(
            contents=[
                {"role": "user", "parts": [prompt]}
            ]
        )
        content = response.text.strip()

        # Try parsing JSON to confirm validity
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return result
    except json.JSONDecodeError:
        return {
            "answer": "The document does not contain any clear or relevant clause to address the query. Please refer to the policy document directly or contact the insurer for further clarification."
        }
    except Exception as e:
        print(f"Error calling GenAI API from llm.py: {e}")
        return {
            "answer": "Error",
            "supporting_clause": "None",
            "explanation": f"Error while calling LLM API: {str(e)}"
        }