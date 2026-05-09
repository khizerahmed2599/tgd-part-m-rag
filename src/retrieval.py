import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_PATH = "data/index.faiss"
METADATA_PATH = "data/index_metadata.json"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_retriever():
    """Load the FAISS index, metadata, and embedding model."""
    # 1. Load the FAISS index from INDEX_PATH using faiss.read_index()
    index = faiss.read_index(INDEX_PATH)
    # 2. Load the metadata list from METADATA_PATH
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    # 3. Load the SentenceTransformer model
    model = SentenceTransformer(MODEL_NAME)
    # 4. Return all three
    return index, metadata, model


def retrieve(query: str, index, metadata: list[dict], model, top_k: int = 5) -> list[dict]:
    """Find the top_k chunks most similar to the query."""
    # 1. Prefix the query with QUERY_PREFIX
    full_query = QUERY_PREFIX + query
    # 2. Embed the query (returns a numpy array of shape (1, 384))
    query_embedding = model.encode([full_query], show_progress_bar=False)
    # 3. Normalize it with faiss.normalize_L2 (in-place, no assignment)
    faiss.normalize_L2(query_embedding)
    # 4. Call index.search(query_embedding, top_k) — returns (scores, indices)
    scores, indices = index.search(query_embedding, top_k)
    
    # 5. Iterate over scores and indices using zip
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = metadata[idx]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(score),
        })
    
    # 6. Return the list of results
    return results


if __name__ == "__main__":
    index, metadata, model = load_retriever()
    print(f"Loaded index with {index.ntotal} vectors and {len(metadata)} metadata entries")

    query = "What handrail height is required on stairs?"
    # Other Queries tried
    # "What is the minimum width of a corridor for wheelchair access?"

    print(f"\nQuery: {query}\n")

    results = retrieve(query, index, metadata, model, top_k=5)

    for rank, r in enumerate(results, start=1):
        print(f"--- Rank {rank} | score: {r['score']:.3f} | {r['chunk_id']} | page {r['page']} ---")
        print(r["text"][:300])
        print()