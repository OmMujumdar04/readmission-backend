"""
Step 2: Build a FAISS vector index from the knowledge base.
Run this ONCE locally (and again anytime you edit the knowledge base).
It saves the index to disk — the FastAPI app will just load it, not rebuild it.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# ---- Config ----
KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "clinical_and_model_kb.md")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
os.makedirs(INDEX_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free, runs locally

# ---- Step 1: Load the knowledge base text ----
with open(KB_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Loaded knowledge base: {len(raw_text)} characters")

# ---- Step 2: Chunk it ----
# chunk_size = max characters per chunk
# chunk_overlap = characters shared between consecutive chunks, so we don't
# lose context when a sentence gets cut at a chunk boundary
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "]  # try to split at headers/paragraphs first
)
chunks = splitter.split_text(raw_text)
print(f"Split into {len(chunks)} chunks")

# ---- Step 3: Embed each chunk ----
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
print(f"Embedding shape: {embeddings.shape}")  # (num_chunks, 384) for this model

# ---- Step 4: Build the FAISS index ----
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # simple exact nearest-neighbor search, fine at our scale
index.add(embeddings)

# ---- Step 5: Save index + the raw chunk text (we need text back after retrieval) ----
faiss.write_index(index, os.path.join(INDEX_DIR, "kb_index.faiss"))
with open(os.path.join(INDEX_DIR, "kb_chunks.pkl"), "wb") as f:
    pickle.dump(chunks, f)

print("Done. Saved index and chunks to rag/index/")