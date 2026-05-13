# TGD Part M RAG

![Eval](https://github.com/khizerahmed2599/tgd-part-m-rag/actions/workflows/eval.yml/badge.svg)

A retrieval-augmented generation system over Ireland's Technical Guidance Document M (Access and Use, 2022) — the regulation that governs disability-access compliance in Irish buildings.

Built to demonstrate production-engineering practices around an LLM application: evaluation harness, observability, and measured improvement against ground truth.

## What it does

Given a question about Part M (e.g., *"What is the minimum corridor width for wheelchair access?"*), the system:

1. Embeds the question and retrieves the top-k most relevant clauses from the regulation using BGE-small + FAISS
2. Sends the clauses + question to Gemini 2.5 Flash with a constrained prompt that requires inline citations
3. Returns a grounded answer that cites chunk IDs and page numbers

If the regulation doesn't contain the answer, the system says so explicitly rather than fabricating one. Off-topic questions are politely refused.

## Baseline evaluation

Measured against 33 hand-curated questions (25 in-scope + 5 off-topic + 3 in-scope-unanswerable), each tagged with ground-truth chunk IDs:

| Metric | Value | Meaning |
|--------|-------|---------|
| Hit Rate @ 10 | 0.88 | At least one relevant chunk in top-10 |
| Recall @ 10 | 0.81 | Fraction of relevant chunks retrieved |
| MRR | 0.58 | First relevant chunk at rank ~1.7 on average |

Three diagnostic in-scope misses surfaced specific failure modes:

- **q07 (WC vs toilet)** — vocabulary sensitivity between regulator jargon and everyday language. Justifies hybrid search (BM25 + dense).
- **q08 (passing places spacing)** — chunk dilution plus an adversarial near-miss conflating internal corridors with external access routes. Justifies a cross-encoder reranker.
- **q15 (step rise and going)** — pure chunk dilution where the canonical answer scores lower than dense topical neighbours.

Off-topic and in-scope-unanswerable score distributions overlap (max scores 0.46–0.67 vs 0.65–0.67), confirming refusal logic must come from the prompt rather than a similarity threshold.

Full findings: [`docs/eval_findings_day1.docx`](docs/eval_findings_day1.docx)

## Architecture

```
Build-time pipeline (one-time, re-runs only when the regulation updates):

  TGD_Part_M.pdf
      │
      ▼
  extract_pdf.py     pypdf → 1-indexed page texts
      │
      ▼
  chunk_text.py      600-char windows, 100-char overlap → 630 chunks
      │
      ▼
  build_index.py     BGE-small embeddings → FAISS IndexFlatIP
                     → data/index.faiss + data/index_metadata.json

Query-time pipeline (per user question):

  User query
      │
      ▼
  retrieve.py        embed → normalize → FAISS top-k search
      │
      ▼
  generate.py        Gemini + system instruction + cited clauses
      │
      ▼
  Grounded answer with [chunk_id, page] citations
```

Concept-level documentation, design decisions, and known failure modes: [`docs/baseline_walkthrough.docx`](docs/baseline_walkthrough.docx).

## Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| PDF extraction | `pypdf` | Simple, sufficient for baseline; revisit with `pdfplumber` if tables become a bottleneck |
| Chunking | Fixed-size character (600/100) | Deliberately dumb baseline; structure-aware chunking is on the roadmap |
| Embeddings | `BAAI/bge-small-en-v1.5` | 384-dim, CPU-friendly, strong baseline for retrieval |
| Vector index | FAISS `IndexFlatIP` | Exact search at 630 vectors; graduate to HNSW if corpus grows |
| Generation | Gemini 2.5 Flash, `temperature=0.1` | Low randomness for factual Q&A over regulations |
| Observability | LangFuse v2 | Per-query traces with scores, tags, sessions |
| Eval harness | Custom Python | 33-question ground-truth set with multi-truth chunk IDs |

## Observability

Every eval run is traced in LangFuse. Each question becomes a trace with:

- **Input:** question text
- **Output:** retrieved chunks (chunk_id, score, rank) + expected ground truth
- **Scores:** `hit`, `recall`, `rr`, `max_score`
- **Tags:** `category:*`, `top_k:*`, `model:*`, `git_sha:*`
- **Session:** one eval run = one session, all 33 traces grouped

This makes it possible to compare runs over time (e.g., before/after adding a reranker) and drill into specific question failures.

![LangFuse trace view](docs/img/langfuse_traces.png)

## Docker

The eval harness and (placeholder) API server are containerized via a multi-mode Dockerfile. The image:

- Bakes the BGE-small model into the image to avoid runtime downloads
- Layers ordered so code edits trigger ~10s rebuilds, not full dependency reinstalls
- Secrets injected at runtime via `--env-file .env` — never baked into the image
- Eval results inside the container match local exactly (Hit Rate 0.880, Recall 0.807, MRR 0.576)

Run the eval inside the container:

```bash
docker build -t tgd-part-m-rag .
docker run --rm --env-file .env tgd-part-m-rag
```

This is the same image GitHub Actions will run on every PR (Day 4 of the sprint).

## Roadmap

- [x] Baseline pipeline (extract → chunk → embed → retrieve → generate)
- [x] Refusal pathway (off-topic and unanswerable questions)
- [x] Evaluation harness with 33-question ground-truth set
- [x] Observability via LangFuse (per-query traces, scored metrics, session grouping)
- [x] Containerization (Docker)
- [x] CI/CD with regression-gated eval (GitHub Actions)
- [ ] Measured improvement: hybrid search OR cross-encoder reranker — chosen based on eval evidence

## Setup

```bash
# Clone
git clone https://github.com/khizerahmed2599/tgd-part-m-rag.git
cd tgd-part-m-rag

# Virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# Dependencies
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env to fill in:
#   GEMINI_API_KEY=...
#   LANGFUSE_PUBLIC_KEY=...
#   LANGFUSE_SECRET_KEY=...
#   LANGFUSE_HOST=https://cloud.langfuse.com
```

## Running

```bash
# 1. Build the index (one-time; assumes data/TGD_Part_M.pdf is present)
python -m src.extract_pdf
python -m src.chunk_text
python -m src.build_index

# 2. Ask a question
python -m src.main "What is the minimum corridor width for wheelchair access?"

# 3. Run the eval harness
python -m eval.run_eval                  # default top_k=10
python -m eval.run_eval --top-k 5        # vary parameters

# OR: run the eval inside Docker (matches CI exactly)
docker build -t tgd-part-m-rag .
docker run --rm --env-file .env tgd-part-m-rag
```

Results are written to `eval/results/<timestamp>_<git_sha>.json` and traced in LangFuse.

## Repo structure

```
.
├── src/                  Pipeline code: extract, chunk, build_index, retrieve, generate
├── data/                 PDF + index artifacts (large files gitignored)
├── eval/
│   ├── questions.jsonl   33-question ground-truth set
│   ├── run_eval.py       Eval harness — Hit Rate, Recall, MRR per category
│   └── results/          Timestamped eval results (committed for regression tracking)
├── docs/
│   ├── baseline_walkthrough.docx   System architecture + design decisions
│   ├── eval_findings_day1.docx     What the eval measured and what to fix
│   └── img/                        README screenshots
├── Dockerfile                      Multi-mode image (eval default, api stub)
├── docker-entrypoint.sh            Routes `eval` vs `api` modes
├── .dockerignore                   Excludes venv, secrets, build artifacts
├── .env.example
└── requirements.txt 
