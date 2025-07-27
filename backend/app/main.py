# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
from app.extract_clauses import extract_clauses_from_url
from app.prompts import MISTRAL_SYSTEM_PROMPT_TEMPLATE
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import requests
import asyncio
import sys
import os

# Import the Google GenAI library
import google.generativeai as genai

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Google Generative AI with your API key
# It's recommended to load this from an environment variable
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize the GenAI model
# You can choose a different model if 'gemini-pro' is not suitable
genai_model = genai.GenerativeModel('gemini-2.0-flash')


# ✅ Build FAISS index with clause embeddings
def build_faiss_index(clauses):
    texts = [c["clause"] for c in clauses]
    vectors = model.encode(texts)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, texts, vectors

# ✅ Query FAISS to get top-k relevant clauses
def get_top_clauses(question, index, texts, k=3):
    q_vector = model.encode([question])
    _, I = index.search(np.array(q_vector), k)
    return [texts[i] for i in I[0]]

# ✅ Async GenAI LLM call
async def call_genai_llm_async(prompt: str, timeout: int = 120) -> dict:
    try:
        # The prompt already includes system instructions and user query based on MISTRAL_SYSTEM_PROMPT_TEMPLATE
        # So, we can directly pass it as the user content to the GenAI model.
        # GenAI's chat structure typically handles alternating user/model roles.
        # For a single prompt, you can use generate_content directly.
        response = await asyncio.to_thread(
            genai_model.generate_content,
            contents=[
                {"role": "user", "parts": [prompt]}
            ]
        )
        raw_output = response.text.strip()
        
        # The prompt template expects a JSON output, so attempt to parse it
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_output)
    except Exception as e:
        print(f"Error calling GenAI API: {e}")
        return {
            "answer": "Error",
            "supporting_clause": "None",
            "explanation": f"Error while calling LLM API: {str(e)}"
        }

@app.post("/hackrx/run")
async def hackrx_run(req: HackRxRequest):
    doc_urls = req.documents if isinstance(req.documents, list) else [req.documents]

    # Step 1: Extract and index clauses
    all_clauses = []
    for url in doc_urls:
        all_clauses.extend(extract_clauses_from_url(url))

    index, clause_texts, _ = build_faiss_index(all_clauses)

    # Step 2: Prepare prompts concurrently
    async def process_question(q):
        top_clauses = get_top_clauses(q, index, clause_texts, k=3)
        formatted_clauses = "\n".join([f"{i+1}. {c}" for i, c in enumerate(top_clauses)])
        
        # Use the existing prompt template
        prompt = MISTRAL_SYSTEM_PROMPT_TEMPLATE.format(query=q, clauses=formatted_clauses)
        
        # Call the updated GenAI function
        return await call_genai_llm_async(prompt)

    results = await asyncio.gather(*[process_question(q) for q in req.questions])
    return {"answers": results}