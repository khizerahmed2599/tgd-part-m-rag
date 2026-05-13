import re
from rank_bm25 import BM25Okapi

RRF_K = 60  # Reciprocal Rank Fusion parameter

def tokenize(text:str) -> list[str]:
    """Simple whitespace and punctuation tokenizer.
    Converts to lowercase and splits on word boundaries."""
    return re.findall(r'\b\w+\b', text.lower())

def build_bm25_index(metadata: list[dict]) -> BM25Okapi:
    """Build a BM25 index from the chunk texts."""
    corpus = [tokenize(chunk["text"]) for chunk in metadata]
    return BM25Okapi(corpus)

def bm25_retrieve(query: str, bm25: BM25Okapi, 
                  metadata: list[dict], top_k: int) -> list[dict]:

    tokenized_query = tokenize(query)
    scores  = bm25.get_scores(tokenized_query)


    top_indices =  sorted(range(len(scores)),
                          key = lambda i: scores[i], 
                          reverse=True)[:top_k]
    
    results = []
    for idx in top_indices:
        chunk = metadata[idx]
        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(scores[idx])  # BM25 score as float
        })

    return results

def reciprocal_rank_fusion(ranked_lists: list[list[dict]], top_k: int,
                            k: int = RRF_K) -> list[dict]:
    
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {} # one full record per chunk_id for final output

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_data:
                chunk_data[cid] = result  # store first-seen copy
 
    # Sort all seen chunk_ids by RRF score
    sorted_ids = sorted(
        rrf_scores,
        key=lambda cid: rrf_scores[cid],
        reverse=True
    )
 
    fused = []
    for cid in sorted_ids[:top_k]:
        entry = chunk_data[cid].copy()
        entry["score"] = rrf_scores[cid]   # replace original score with RRF score
        fused.append(entry)
 
    return fused

if __name__ == "__main__":
    # Quick smoke test — run with: python src/hybrid.py
    # Verifies index builds and RRF produces sensible output.
 
    fake_metadata = [
        {"chunk_id": "p1_c0", "page": 1, "text": "The minimum corridor width should be 1200 mm."},
        {"chunk_id": "p2_c0", "page": 2, "text": "Ramps must have a gradient of no more than 1 in 20."},
        {"chunk_id": "p3_c0", "page": 3, "text": "Accessible WC dimensions should be 1500 x 2200 mm."},
    ]
 
    bm25 = build_bm25_index(fake_metadata)
    results = bm25_retrieve("corridor width minimum", bm25, fake_metadata, top_k=3)
 
    print("BM25 results for 'corridor width minimum':")
    for r in results:
        print(f"  {r['chunk_id']}  score={r['score']:.4f}  text={r['text'][:50]}")
 
    # Fake dense results (would come from retrieve() in practice)
    dense_results = [
        {"chunk_id": "p1_c0", "page": 1, "text": fake_metadata[0]["text"], "score": 0.85},
        {"chunk_id": "p3_c0", "page": 3, "text": fake_metadata[2]["text"], "score": 0.72},
        {"chunk_id": "p2_c0", "page": 2, "text": fake_metadata[1]["text"], "score": 0.60},
    ]
 
    fused = reciprocal_rank_fusion([dense_results, results], top_k=3)
    print("\nRRF fused results:")
    for r in fused:
        print(f"  {r['chunk_id']}  rrf_score={r['score']:.4f}")
 
    print("\nSmoke test passed.")