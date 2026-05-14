import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean
 
from dotenv import load_dotenv
 
load_dotenv()
 
from google import genai
from google.genai import types
 
from src.retrieval import retrieve, MODEL_NAME
from src.reranker import load_reranker, rerank
from src.hybrid import build_bm25_index, bm25_retrieve, reciprocal_rank_fusion
from src.generate import generate_answer
from eval.run_eval import load_retriever

JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-3-flash-preview")


FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI assistant's answer is faithful to \
the retrieved chunks it was given.
 
QUESTION:
{question}
 
RETRIEVED CHUNKS:
{chunks}
 
GENERATED ANSWER:
{answer}
 
A claim is SUPPORTED if it can be directly traced to information in one of the chunks.
A claim is NOT SUPPORTED if it:
- Adds facts not present in any chunk
- Makes quantitative claims (measurements, distances, percentages) not found in chunks
- Contradicts information in the chunks
- Generalises beyond what the chunks say
 
If the answer says it cannot answer from the provided information, that is a \
FAITHFUL response — score it 1.0 and note it as a correct refusal.
 
Respond ONLY with a valid JSON object, no markdown, no explanation outside the JSON:
{{
  "faithfulness": <float 0.0 to 1.0>,
  "unsupported_claims": [<list of specific unsupported claims, empty if none>],
  "explanation": "<one sentence explaining the score>",
  "is_refusal": <true if the answer declined to answer, false otherwise>
}}"""

REFUSAL_PROMPT = """\
You are evaluating whether an AI assistant correctly refused to answer a question.
 
QUESTION: {question}
CATEGORY: {category}
 
GENERATED RESPONSE:
{answer}
 
For "off_topic": the system should refuse because the question is not about \
TGD Part M building accessibility regulations.
 
For "in_scope_unanswerable": the system should refuse because the question \
is about building regulations but TGD Part M does not cover it. \
The system must NOT guess or use general knowledge.
 
A refusal is CORRECT if the system declined to answer and gave a reason.
A refusal is INCORRECT if the system attempted to answer (even partially) \
or refused without explanation.
 
Respond ONLY with a valid JSON object, no markdown:
{{
  "refused_correctly": <true or false>,
  "explanation": "<one sentence explaining your verdict>"
}}"""


def load_judge() -> genai.Client:
    """Initialise the Gemini judge client using the separate judge API key."""
    key = os.getenv("GEMINI_JUDGE_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_JUDGE_API_KEY not set. "
            "Add it to .env and to GitHub Secrets."
        )
    return genai.Client(api_key=key)

def format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks for inclusion in the judge prompt."""
    formatted = []
    for i, c in enumerate(chunks, 1):
        formatted.append(
            f"[{i}] (chunk_id: {c['chunk_id']}, page {c['page']})\n{c['text']}"
        )
    return "\n\n".join(formatted)


def parse_judge_json(raw: str) -> dict | None:
    """Extract and parse JSON from judge response.
 
    The judge is instructed to return only JSON, but LLMs sometimes
    wrap it in markdown backticks. Strip those before parsing.
    Returns None if parsing fails — caller handles the fallback.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    

def judge_faithfulness(question: str,chunks: list[dict], answer: str, judge: genai.Client) -> dict:
    """Ask the judge to score answer faithfulness.
 
    Returns a dict with faithfulness (float), unsupported_claims (list),
    explanation (str), is_refusal (bool). On judge failure, returns a
    sentinel dict with faithfulness=-1 so failures are visible in results.
    """
    prompt = FAITHFULNESS_PROMPT.format(
        question=question,
        chunks=format_chunks(chunks),
        answer=answer,
    )
    try:
        response = judge.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        parsed = parse_judge_json(response.text)
        if parsed is None:
            return {
                "faithfulness": -1,
                "unsupported_claims": [],
                "explanation": f"Judge returned unparseable response: {response.text[:200]}",
                "is_refusal": False,
                "judge_error": True,
            }
        return {
            "faithfulness": float(parsed.get("faithfulness", -1)),
            "unsupported_claims": parsed.get("unsupported_claims", []),
            "explanation": parsed.get("explanation", ""),
            "is_refusal": bool(parsed.get("is_refusal", False)),
            "judge_error": False,
        }
    except Exception as e:
        return {
            "faithfulness": -1,
            "unsupported_claims": [],
            "explanation": f"Judge call failed: {str(e)}",
            "is_refusal": False,
            "judge_error": True,
        }
 
 
def judge_refusal(
    question: str,
    category: str,
    answer: str,
    judge: genai.Client,
) -> dict:
    """Ask the judge to score whether the system correctly refused.
 
    Returns a dict with refused_correctly (bool) and explanation (str).
    """
    prompt = REFUSAL_PROMPT.format(
        question=question,
        category=category,
        answer=answer,
    )
    try:
        response = judge.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        parsed = parse_judge_json(response.text)
        if parsed is None:
            return {
                "refused_correctly": False,
                "explanation": f"Judge returned unparseable response: {response.text[:200]}",
                "judge_error": True,
            }
        return {
            "refused_correctly": bool(parsed.get("refused_correctly", False)),
            "explanation": parsed.get("explanation", ""),
            "judge_error": False,
        }
    except Exception as e:
        return {
            "refused_correctly": False,
            "explanation": f"Judge call failed: {str(e)}",
            "judge_error": True,
        }

def run_single_query(
    question: str,
    retriever: dict,
    top_k: int,
) -> tuple[list[dict], str]:
    """Run full pipeline for one question: retrieve → generate.
 
    Returns (chunks, answer). Identical pipeline to the API server.
    """
    # Stage 1: hybrid retrieval
    dense = retrieve(
        question,
        retriever["index"],
        retriever["metadata"],
        retriever["model"],
        top_k=top_k * 2,
    )
    sparse = bm25_retrieve(
        question,
        retriever["bm25"],
        retriever["metadata"],
        top_k=top_k * 2,
    )
    candidates = reciprocal_rank_fusion([dense, sparse], top_k=top_k * 2)
 
    # Stage 2: rerank
    chunks = rerank(question, candidates, retriever["reranker"], top_k=top_k)
 
    # Stage 3: generate
    answer = generate_answer(question, chunks)
 
    return chunks, answer


def aggregate_gen(per_query: list[dict]) -> dict:
    """Compute generation metrics grouped by category.
 
    in_scope:
        mean_faithfulness     — average judge faithfulness score (0-1)
        pct_high_faithfulness — % of answers scoring >= 0.8
        n_judge_errors        — failed judge calls (excluded from averages)
 
    off_topic + in_scope_unanswerable:
        refusal_accuracy — fraction correctly refused
    """
    in_scope = [r for r in per_query if r["category"] == "in_scope"]
    refusal_cats = [
        r for r in per_query
        if r["category"] in ("off_topic", "in_scope_unanswerable")
    ]
 
    agg: dict = {}
 
    if in_scope:
        valid = [
            r for r in in_scope
            if not r["judge_result"].get("judge_error")
            and r["judge_result"]["faithfulness"] >= 0
        ]
        errors = len(in_scope) - len(valid)
        scores = [r["judge_result"]["faithfulness"] for r in valid]
        agg["in_scope"] = {
            "n": len(in_scope),
            "n_judge_errors": errors,
            "mean_faithfulness": mean(scores) if scores else 0.0,
            "pct_high_faithfulness": (
                sum(1 for s in scores if s >= 0.8) / len(scores)
            ) if scores else 0.0,
        }
 
    if refusal_cats:
        valid_r = [
            r for r in refusal_cats
            if not r["judge_result"].get("judge_error")
        ]
        correct = sum(
            1 for r in valid_r if r["judge_result"].get("refused_correctly")
        )
        agg["refusal"] = {
            "n": len(refusal_cats),
            "n_judge_errors": len(refusal_cats) - len(valid_r),
            "refusal_accuracy": correct / len(valid_r) if valid_r else 0.0,
        }
 
    return agg


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"
 
 
def save_gen_results(
    metrics: dict,
    raw: list[dict],
    out_dir: str,
    config: dict,
) -> str:
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

def print_gen_summary(metrics: dict, judge_model: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"Generation Eval Summary (judge: {judge_model})")
    print(f"{'=' * 60}")
 
    if "in_scope" in metrics:
        m = metrics["in_scope"]
        print(f"\nin_scope (n={m['n']}, errors={m['n_judge_errors']}):")
        print(f"  Mean faithfulness:    {m['mean_faithfulness']:.3f}")
        print(f"  % high faithfulness:  {m['pct_high_faithfulness']:.1%}")
 
    if "refusal" in metrics:
        m = metrics["refusal"]
        print(f"\nRefusal categories (n={m['n']}, errors={m['n_judge_errors']}):")
        print(f"  Refusal accuracy:     {m['refusal_accuracy']:.3f}")
 
 
def print_gen_failures(per_query: list[dict]) -> None:
    """Print questions with low faithfulness or incorrect refusals."""
    low_faith = [
        r for r in per_query
        if r["category"] == "in_scope"
        and not r["judge_result"].get("judge_error")
        and r["judge_result"].get("faithfulness", 1.0) < 0.7
    ]
    wrong_refusals = [
        r for r in per_query
        if r["category"] in ("off_topic", "in_scope_unanswerable")
        and not r["judge_result"].get("judge_error")
        and not r["judge_result"].get("refused_correctly", True)
    ]
 
    if low_faith:
        print(f"\n{'=' * 60}")
        print(f"Low faithfulness (< 0.70): {len(low_faith)}")
        print(f"{'=' * 60}")
        for r in low_faith:
            jr = r["judge_result"]
            print(f"\n  {r['id']}: {r['question'][:60]}")
            print(f"     faithfulness={jr['faithfulness']:.3f}")
            print(f"     explanation: {jr['explanation']}")
            if jr.get("unsupported_claims"):
                print(f"     unsupported: {jr['unsupported_claims'][:2]}")
 
    if wrong_refusals:
        print(f"\n{'=' * 60}")
        print(f"Incorrect refusals: {len(wrong_refusals)}")
        print(f"{'=' * 60}")
        for r in wrong_refusals:
            print(f"\n  {r['id']} ({r['category']}): {r['question'][:60]}")
            print(f"     {r['judge_result']['explanation']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation eval harness.")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--index", default="data/index.faiss")
    parser.add_argument("--metadata", default="data/index_metadata.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-dir", default="eval/gen_results")
    args = parser.parse_args()
 
    # Load retriever
    print(f"Loading retriever (model={MODEL_NAME})")
    retriever = load_retriever(args.index, args.metadata)
    print(f"  -> {len(retriever['metadata'])} chunks in index")
 
    # Load judge
    print(f"Loading judge (model={JUDGE_MODEL})")
    judge = load_judge()
    print("  -> Judge client initialised")
 
    # Load questions
    print(f"Loading questions from {args.questions}")
    with open(args.questions, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"  -> {len(questions)} questions")
 
    # Run pipeline + judge for each question
    print(f"\nRunning {len(questions)} questions...")
    per_query = []
 
    for i, q in enumerate(questions, 1):
        print(f"  [{i:2d}/{len(questions)}] {q['id']}...", end=" ", flush=True)
 
        # Run pipeline
        chunks, answer = run_single_query(q["question"], retriever, args.top_k)
 
        # Judge the answer
        if q["category"] == "in_scope":
            judge_result = judge_faithfulness(q["question"], chunks, answer, judge)
            score_summary = f"faithfulness={judge_result['faithfulness']:.3f}"
        else:
            judge_result = judge_refusal(q["question"], q["category"], answer, judge)
            score_summary = f"refused_correctly={judge_result['refused_correctly']}"
 
        if judge_result.get("judge_error"):
            print(f"JUDGE ERROR — {judge_result['explanation'][:60]}")
        else:
            print(score_summary)
 
        per_query.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "answer": answer,
            "chunks_retrieved": [
                {"chunk_id": c["chunk_id"], "page": c["page"]}
                for c in chunks
            ],
            "judge_result": judge_result,
        })
 
    # Aggregate and save
    metrics = aggregate_gen(per_query)
    config = {
        "top_k": args.top_k,
        "retrieval_model": MODEL_NAME,
        "judge_model": JUDGE_MODEL,
        "questions_file": args.questions,
    }
    out_path = save_gen_results(metrics, per_query, args.out_dir, config)
 
    print_gen_summary(metrics, JUDGE_MODEL)
    print_gen_failures(per_query)
    print(f"\nResults written to: {out_path}")
 
 
if __name__ == "__main__":
    main()