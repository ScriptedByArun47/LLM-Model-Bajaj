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

# ✅ Async Ollama LLM call
async def call_mistral_llm_async(prompt: str, timeout: int = 60) -> dict:
    try:
        response = await asyncio.to_thread(requests.post,
            "http://localhost:11434/api/chat",
            json={
                "model": "mistral",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a legal assistant trained to process insurance policy documents."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            },
            timeout=timeout
        )
        response.raise_for_status()
        raw_output = response.json().get("message", {}).get("content", "").strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_output)
    except Exception as e:
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
        prompt = MISTRAL_SYSTEM_PROMPT_TEMPLATE.format(query=q, clauses=formatted_clauses)
        return await call_mistral_llm_async(prompt)

    results = await asyncio.gather(*[process_question(q) for q in req.questions])
    return {"answers": results}
