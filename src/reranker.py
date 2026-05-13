"""
Cross-encoder reranker for TGD Part M RAG.

Provides a second-stage reranking step on top of the bi-encoder
(BGE-small) first-stage retrieval.

Why two stages?
    The bi-encoder (BGE-small) embeds the query and each chunk
    independently, then compares them via cosine similarity. This is
    fast but loses interaction signal — the model never sees the query
    and a specific chunk together.

    The cross-encoder reads (query, chunk_text) as a single input and
    produces a relevance score that captures how well this specific
    chunk answers this specific question. Much more accurate, but
    too slow to run over the full 630-chunk corpus.

    Solution: bi-encoder retrieves top 20 candidates quickly,
    cross-encoder reranks those 20 precisely. Best of both.
"""
from sentence_transformers import CrossEncoder

# Small, fast cross-encoder trained on MS MARCO passage reranking.
# No GPU required. ~80 MB download.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker() -> CrossEncoder:
    """Load the cross-encoder model.

    Call once at startup — not per query. The model is ~80 MB
    and takes 2-3 seconds to load. Loaded once, reused for every query.
    """
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, results: list[dict], model: CrossEncoder,
           top_k: int) -> list[dict]:
    """Reorder retrieved chunks using the cross-encoder.

    Args:
        query: the plain user question — NO BGE query prefix.
               The cross-encoder was not trained with prefixes.
        results: output from retrieve(). Each dict has at minimum
                 chunk_id, page, text, and score (bi-encoder).
        model: loaded CrossEncoder from load_reranker().
        top_k: how many results to return after reranking.

    Returns:
        top_k results, reordered by cross-encoder score.
        Each result gains a 'rerank_score' field.
        The original bi-encoder score is preserved as 'retrieval_score'.
        'score' is set to the rerank_score so downstream code that
        reads 'score' automatically uses the better signal.
    """
    if not results:
        return results

    # Build (query, chunk_text) pairs — the cross-encoder's input format.
    # It reads both together in one forward pass, unlike the bi-encoder.
    pairs = [(query, r["text"]) for r in results]

    # Score all pairs in one batch call.
    # Returns a numpy array of logit scores — higher = more relevant.
    # Scale is different from cosine similarity (can be negative or > 1).
    scores = model.predict(pairs)

    # Attach scores and preserve the original retrieval score for debugging.
    for result, rerank_score in zip(results, scores):
        result["retrieval_score"] = result.pop("score")   # preserve bi-encoder score
        result["rerank_score"] = float(rerank_score)
        result["score"] = float(rerank_score)              # 'score' now = reranker signal

    # Sort by reranker score descending, return top_k.
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]

if __name__ == "__main__":

    # Quick test to verify the reranker is working.
    # Run `python src/reranker.py` and check the output. This only runs when you execute this file directly, not when imported the functions above.
    model = load_reranker()
    test_query = "What is the capital of France?"
    test_results = [
        {"chunk_id": 1, "text": "Paris is the capital of France.", "score": 0.8},
        {"chunk_id": 2, "text": "Berlin is the capital of Germany.", "score": 0.7},
        {"chunk_id": 3, "text": "Madrid is the capital of Spain.", "score": 0.6},
    ]
    reranked = rerank(test_query, test_results, model, top_k=3)
    for r in reranked:
        print(f"Chunk ID: {r['chunk_id']}, Rerank Score: {r['rerank_score']:.4f}, Text: {r['text']}")