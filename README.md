# TGD Part M RAG

A retrieval-augmented generation system over Ireland's Technical Guidance
Document M (Access and Use, 2022) — the regulation that governs
disability-access compliance in Irish buildings.

## What it does

Given a question about Part M (e.g., *"What is the minimum corridor
width for wheelchair access?"*), the system:

1. Embeds the question and retrieves the top-5 most relevant clauses
   from the regulation using BGE-small + FAISS.
2. Sends the clauses + question to Gemini 2.5 Flash with a constrained
   prompt that requires inline citations.
3. Returns a grounded answer that cites chunk IDs and page numbers.

If the regulation doesn't contain the answer, the system says so
explicitly rather than fabricating one. Off-topic questions are
politely refused.

## Architecture

See [`docs/REFERENCE.md`](docs/REFERENCE.md) for a full technical
walkthrough of every stage and the design decisions behind it.

## Status

- [x] Baseline pipeline (extract → chunk → embed → retrieve → generate)
- [x] Refusal pathway (off-topic and unanswerable questions)
- [x] Evaluation harness — with Baseline at top_k=10: Hit Rate: 0.88, Recall: 0.81, MRR: 0.58
  3 in-scope misses diagnose:
    - q07: WC/toilet vocabulary sensitivity (paraphrase pair caught it)
    - q08: chunk dilution + regulatory context conflation
    - q15: chunk dilution on rise/going spec
  Off-topic and in-scope-unanswerable score distributions overlap,
  refusal logic must come from prompt not threshold.
- [ ] Reranking, hybrid search, structure-aware chunking — measured against eval set

## Setup

(your existing setup steps)

## Running

```bash