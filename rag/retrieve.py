"""
Step 3: Retrieval function.
Given a user question, find the most relevant chunks from the knowledge base.
"""

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

INDEX_DIR = os.path.join(os.path.dirname(__file__), "index")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load once at import time (not on every request — loading is slow, searching is fast)
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
_index = faiss.read_index(os.path.join(INDEX_DIR, "kb_index.faiss"))
with open(os.path.join(INDEX_DIR, "kb_chunks.pkl"), "rb") as f:
    _chunks = pickle.load(f)


def retrieve(query: str, top_k: int = 3):
    """
    Returns the top_k most relevant chunks of text for a given query.
    """
    query_embedding = _model.encode([query], convert_to_numpy=True)

    # FAISS search returns two arrays: distances and indices of nearest neighbors
    distances, indices = _index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "text": _chunks[idx],
            "distance": float(distances[0][rank])  # lower = more similar
        })
    return results


# Quick manual test — run this file directly to sanity-check retrieval
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