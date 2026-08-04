"""
Step 3 (revised): Retrieval function.
Uses HuggingFace's free Inference API for embeddings (no local model loaded),
to keep memory usage low on free-tier hosting.
"""

import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = InferenceClient(token=os.environ.get("HF_TOKEN"))

_index = faiss.read_index(os.path.join(INDEX_DIR, "kb_index.faiss"))
with open(os.path.join(INDEX_DIR, "kb_chunks.pkl"), "rb") as f:
    _chunks = pickle.load(f)


def embed_text(text: str):
    result = client.feature_extraction(text, model=EMBEDDING_MODEL_NAME)
    return np.array(result, dtype=np.float32)


def retrieve(query: str, top_k: int = 3):
    """
    Returns the top_k most relevant chunks of text for a given query.
    """
    query_embedding = embed_text(query).reshape(1, -1)
    distances, indices = _index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "text": _chunks[idx],
            "distance": float(distances[0][rank])
        })
    return results


if __name__ == "__main__":
    test_questions = [
        "What does A1C mean clinically?",
        "Why was this patient flagged high risk?",
        "What is the model's accuracy?",
    ]
    for q in test_questions:
        print(f"\nQuery: {q}")
        for r in retrieve(q, top_k=2):
            print(f"  [dist={r['distance']:.3f}] {r['text'][:120]}...")