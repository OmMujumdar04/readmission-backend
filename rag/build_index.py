"""
Step 2 (revised): Build a FAISS vector index from the knowledge base.
Uses HuggingFace's free Inference API for embeddings instead of a local model,
to keep memory usage low on free-tier hosting.
Run this ONCE locally (and again anytime you edit the knowledge base).
"""

import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---- Config ----
KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "clinical_and_model_kb.md")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
os.makedirs(INDEX_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # same model, now called remotely

client = InferenceClient(token=os.environ.get("HF_TOKEN"))

# ---- Step 1: Load the knowledge base text ----
with open(KB_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Loaded knowledge base: {len(raw_text)} characters")

# ---- Step 2: Chunk it ----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
)
chunks = splitter.split_text(raw_text)
print(f"Split into {len(chunks)} chunks")

# ---- Step 3: Embed each chunk via the HF Inference API ----
def embed_text(text: str):
    result = client.feature_extraction(text, model=EMBEDDING_MODEL_NAME)
    return np.array(result, dtype=np.float32)

print("Embedding chunks via HuggingFace API (this calls the internet, may take a moment)...")
embeddings = np.array([embed_text(chunk) for chunk in chunks], dtype=np.float32)
print(f"Embedding shape: {embeddings.shape}")

# ---- Step 4: Build the FAISS index ----
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ---- Step 5: Save index + chunks ----
faiss.write_index(index, os.path.join(INDEX_DIR, "kb_index.faiss"))
with open(os.path.join(INDEX_DIR, "kb_chunks.pkl"), "wb") as f:
    pickle.dump(chunks, f)

print("Done. Saved index and chunks to rag/index/")