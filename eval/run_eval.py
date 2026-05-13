import argparse
from importlib.metadata import metadata
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
 
import faiss
from sentence_transformers import SentenceTransformer

# We will need to import the retrival.py module. 
from src.retrieval import MODEL_NAME, retrieve

from dotenv import load_dotenv
load_dotenv()
from langfuse import Langfuse

# Colling the reranker, which is a 
# cross-encoder that reads the query and chunk text together and produces a relevance score.

from src.reranker import load_reranker, rerank

from src.hybrid import bm25_retrieve, build_bm25_index, reciprocal_rank_fusion

def load_questions(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
    

def load_retriever(index_path: str, metadata_path: str) -> dict:
    index = faiss.read_index(index_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    model = SentenceTransformer(MODEL_NAME)
    reranker = load_reranker()
    bm25=build_bm25_index(metadata)
    return {
        "index": index,
        "metadata": metadata,
        "model": model,
        "reranker": reranker,
        "bm25": bm25,
    }

def validate_ground_truth(questions: list[dict], metadata: list[dict]) -> None:
    valid_ids = {c["chunk_id"] for c in metadata}
    bad = []
    for q in questions:
        for cid in q.get("relevant_chunk_ids", []):
            if cid not in valid_ids:
                bad.append((q["id"], cid))
    if bad:
        msg = "\n  ".join(f"{qid}: missing chunk_id {cid!r}" for qid, cid in bad)
        raise ValueError(f"Ground truth references chunks not in index:\n  {msg}")

# Peer-query execution and scoring functions 
def run_single_query(question: str, retriever: dict, top_k: int) -> list[dict]:
    candidates = retrieve(
        question,
        retriever["index"],
        retriever["metadata"],
        retriever["model"],
        top_k=top_k*2,  # Retrieve more candidates for reranking
    )

    sparse = bm25_retrieve(
            question, 
            retriever["bm25"], 
            retriever["metadata"], 
            top_k=top_k*2)
    
    results = reciprocal_rank_fusion([sparse, candidates], top_k=top_k)

    return [
        {"chunk_id": r["chunk_id"], "score": r["score"], "rank": i + 1}
        for i, r in enumerate(results)
    ]

def score_query(retrieved: list[dict], relevant_ids: list[str]) -> dict:
    # → {"hit": 1, "recall": 0.67, "rr": 0.2, "max_score": 0.84}
    retrieved_ids = [r["chunk_id"] for r in retrieved]
    relevant = set(relevant_ids)
    max_score = max((r["score"] for r in retrieved), default=0.0)
 
    if not relevant:
        return {"hit": 0, "recall": 0.0, "rr": 0.0,
                "max_score": max_score, "n_relevant": 0}
    # Hit Rate: 1 if ANY relevant chunk appears in retrieved, else 0
    hits = relevant.intersection(retrieved_ids)
    hit = 1 if hits else 0

    # Recall: fraction of relevant chunks that were retrieved
    recall = len(hits) / len(relevant)
 
    # Reciprocal Rank: 1 / rank of the FIRST relevant chunk (0 if none)
    rr = 0.0
    for r in retrieved:
        if r["chunk_id"] in relevant:
            rr = 1.0 / r["rank"]
            break
 
    return {"hit": hit, "recall": recall, "rr": rr,
            "max_score": max_score, "n_relevant": len(relevant)}

def aggregate(per_query: list[dict]) -> dict:
    """Group per-query results by category and compute averages."""
    by_cat = defaultdict(list)
    for r in per_query:
        by_cat[r["category"]].append(r)
 
    agg = {}
    for cat, rows in by_cat.items():
        if cat == "in_scope":
            agg[cat] = {
                "n": len(rows),
                "hit_rate": mean(r["hit"] for r in rows),
                "recall": mean(r["recall"] for r in rows),
                "mrr": mean(r["rr"] for r in rows),
                "mean_max_score": mean(r["max_score"] for r in rows),
            }
        else:
            # off_topic / in_scope_unanswerable — only score distribution
            scores = [r["max_score"] for r in rows]
            agg[cat] = {
                "n": len(rows),
                "mean_max_score": mean(scores),
                "min_max_score": min(scores),
                "max_max_score": max(scores),
            }
    return agg

def get_git_sha() -> str:
    """Short git SHA, or 'nogit' if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return sha.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def save_results(metrics: dict, raw: list[dict], out_dir: str,
                 config: dict) -> str:
    """Write timestamped JSON: config + aggregated metrics + raw per-query."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sha = get_git_sha()
    path = Path(out_dir) / f"{timestamp}_{sha}.json"
 
    payload = {
        "timestamp": timestamp,
        "git_sha": sha,
        "config": config,
        "aggregated": metrics,
        "per_query": raw,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path)

def print_summary(metrics: dict, top_k: int) -> None:
    """Compact terminal summary."""
    print(f"\n{'=' * 60}")
    print(f"Eval Summary (top_k={top_k})")
    print(f"{'=' * 60}")
 
    if "in_scope" in metrics:
        m = metrics["in_scope"]
        print(f"\nin_scope (n={m['n']}):")
        print(f"  Hit Rate @ {top_k}: {m['hit_rate']:.3f}")
        print(f"  Recall   @ {top_k}: {m['recall']:.3f}")
        print(f"  MRR              : {m['mrr']:.3f}")
        print(f"  Mean max score   : {m['mean_max_score']:.3f}")
 
    for cat in ("off_topic", "in_scope_unanswerable"):
        if cat in metrics:
            m = metrics[cat]
            print(f"\n{cat} (n={m['n']}):")
            print(f"  Max score (mean / min / max): "
                  f"{m['mean_max_score']:.3f} / "
                  f"{m['min_max_score']:.3f} / "
                  f"{m['max_max_score']:.3f}")
            


def print_failures(per_query: list[dict]) -> None:
    """Show in_scope questions where NO relevant chunk was retrieved.
    These are the ones to dig into after a run."""
    misses = [r for r in per_query
              if r["category"] == "in_scope" and r["hit"] == 0]
    if not misses:
        print("\nNo in-scope misses. (Either retrieval is great or k is too big.)")
        return
    print(f"\n{'=' * 60}")
    print(f"In-scope misses: {len(misses)}")
    print(f"{'=' * 60}")
    for r in misses:
        print(f"\n  {r['id']}: {r['question']}")
        print(f"     expected: {r['relevant_chunk_ids']}")
        top3 = [x['chunk_id'] for x in r['retrieved'][:3]]
        print(f"     got top-3: {top3}")

def main():
    
    parser = argparse.ArgumentParser(description="Retrieval eval harness.")
    parser.add_argument("--questions", default="eval/questions.jsonl",
                        help="Path to questions JSONL file.")
    parser.add_argument("--index", default="data/index.faiss",
                        help="Path to FAISS index file.")
    parser.add_argument("--metadata", default="data/index_metadata.json",
                        help="Path to index metadata JSON file.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of chunks to retrieve per query.")
    parser.add_argument("--out-dir", default="eval/results",
                        help="Where to write the results JSON.")
    args = parser.parse_args()
    
    print(f"Loading questions from {args.questions}")
    questions = load_questions(args.questions)
    print(f"  -> {len(questions)} questions")
 
    print(f"Loading retriever (model={MODEL_NAME})")
    retriever = load_retriever(args.index, args.metadata)
    print(f"  -> {len(retriever['metadata'])} chunks in index")
 
    print("Validating ground truth against index...")
    validate_ground_truth(questions, retriever["metadata"])
    print("  -> OK, all chunk_ids resolve")
 
    print(f"\nRunning {len(questions)} queries at top_k={args.top_k}...")

    per_query = []   # collect detailed results for each query, to be written out and aggregated
    git_sha = get_git_sha()
    session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{git_sha}" # unique session ID for this run, combining timestamp and git SHA
    # computed once
    run_tags = [ # these tags will be attached to every trace for easy filtering in the UI
        f"top_k:{args.top_k}",
        f"model:{MODEL_NAME.split('/')[-1]}",  # strip the BAAI/ prefix
        f"git_sha:{get_git_sha()}",
        ]
    
    langfuse = Langfuse() # Initialize Langfuse client
    assert langfuse.auth_check(), "Langfuse client failed to authenticate"
    for q in questions:
        trace_tags = run_tags + [f"category:{q['category']}"] # we also tag by question category (in_scope / off_topic / in_scope_unanswerable)
        trace = langfuse.trace(
                        name=f"eval_{q['id']}",
                        input=q['question'],
                        session_id=session_id,
                        tags=trace_tags,
                        )
        retrieved = run_single_query(q["question"], retriever, args.top_k)
        scores = score_query(retrieved, q.get("relevant_chunk_ids", []))
        
        # NEW — attach scores to the trace
        if q["category"] == "in_scope":
            trace.score(name="hit", value=scores["hit"])
            trace.score(name="recall", value=scores["recall"])
            trace.score(name="rr", value=scores["rr"])
        trace.score(name="max_score", value=scores["max_score"])
        trace_tags = run_tags + [f"category:{q['category']}"]
        
        trace.update(output={
                            "retrieved": retrieved,  # the list of {chunk_id, score, rank}
                             "expected": q.get("relevant_chunk_ids", []),
                         })
        per_query.append({
            "id": q["id"],
            "trace_id": trace.id,
            "question": q["question"],
            "category": q["category"],
            "relevant_chunk_ids": q.get("relevant_chunk_ids", []),
            "retrieved": retrieved,
            **scores,
        })
 
    metrics = aggregate(per_query)
 
    config = {
        "top_k": args.top_k,
        "embedding_model": MODEL_NAME,
        "questions_file": args.questions,
        "index_file": args.index,
    }
    out_path = save_results(metrics, per_query, args.out_dir, config)
 
    print_summary(metrics, args.top_k)
    print_failures(per_query)
    print(f"\nResults written to: {out_path}")

    langfuse.flush()
 
 
if __name__ == "__main__":
    main()