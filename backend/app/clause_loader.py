import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load clauses
CLAUSE_PATH = "app/data/clauses.json"
INDEX_DIR = "app/data/faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
TEXTS_PATH = os.path.join(INDEX_DIR, "texts.json")

if not os.path.exists(CLAUSE_PATH):
    raise FileNotFoundError(f"❌ Clause file not found: {CLAUSE_PATH}")

with open(CLAUSE_PATH, "r", encoding="utf-8") as f:
    clauses = json.load(f)

if not clauses:
    raise ValueError("❌ No clauses found in the JSON file.")

texts = [c['text'] for c in clauses]
embeddings = model.encode(texts, convert_to_numpy=True)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype("float32"))

# Save index and corresponding clause texts
os.makedirs(INDEX_DIR, exist_ok=True)
faiss.write_index(index, INDEX_PATH)

# Optional: save texts separately for retrieval
with open(TEXTS_PATH, "w", encoding="utf-8") as f:
    json.dump(texts, f, indent=2)

print("✅ FAISS index built and saved to:", INDEX_PATH)
print("📄 Clause texts saved to:", TEXTS_PATH)
