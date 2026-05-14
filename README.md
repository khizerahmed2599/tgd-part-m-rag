# TGD Part M RAG

![Eval](https://github.com/khizerahmed2599/tgd-part-m-rag/actions/workflows/eval.yml/badge.svg)

A retrieval-augmented generation system over Ireland's Technical Guidance Document M (Access and Use, 2022) — the regulation that governs disability-access compliance in Irish buildings.

Built to demonstrate production-engineering practices around an LLM application: evaluation harness, measured improvement, observability, containerization, regression-gated CI/CD, and a FastAPI server.

## What it does

Given a question about Part M (e.g., *"What is the minimum corridor width for wheelchair access?"*), the system:

1. Retrieves the top-k most relevant clauses using two-stage hybrid retrieval (BGE-small dense + BM25 sparse → Reciprocal Rank Fusion → cross-encoder reranker)
2. Sends the ranked clauses + question to Gemini 2.5 Flash with a constrained prompt that requires inline citations
3. Returns a grounded answer that cites chunk IDs and page numbers

If the regulation does not contain the answer, the system says so explicitly rather than fabricating one. Off-topic questions are politely refused.

## Evaluation

### Retrieval — measured against 33 hand-curated questions

| Metric | Baseline | After reranker | After hybrid search |
|--------|----------|----------------|---------------------|
| Hit Rate @ 10 | 0.880 | 0.920 | **0.960** |
| Recall @ 10 | 0.807 | 0.840 | **0.860** |
| MRR | 0.576 | 0.676 | **0.647** |
| In-scope misses | 3 | 2 | **1** |

The one remaining miss (q15, rise and going dimensions) is caused by chunk dilution — the answer is buried inside a long mixed-content chunk. Structure-aware chunking is the documented fix.


### Generation — LLM-as-a-Judge

Evaluated using `gemini-3-flash-preview` as judge, separate from the `gemini-2.5-flash` answerer to reduce self-evaluation bias:

| Metric | Value | Meaning |
|--------|-------|---------|
| Faithfulness | 1.000 | Every claim in every answer is supported by retrieved chunks |
| Refusal accuracy | 1.000 | All off-topic and unanswerable questions correctly refused |

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
      ├─ Stage 1a: retrieve.py   BGE-small dense → top-20 by cosine similarity
      ├─ Stage 1b: hybrid.py     BM25 sparse → top-20 by keyword score
      └─ Stage 1c: hybrid.py     Reciprocal Rank Fusion → top-20 fused candidates
      │
      ▼
  Stage 2: reranker.py           Cross-encoder → reordered top-10
      │
      ▼
  generate.py                    Gemini 2.5 Flash + cited clauses
      │
      ▼
  Grounded answer with [chunk_id, page] citations

API:

  src/api.py     FastAPI — POST /query, GET /health, OpenAPI docs at /docs
```

## Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| PDF extraction | `pypdf` | Simple, sufficient; revisit with `pdfplumber` if tables become a bottleneck |
| Chunking | Fixed-size character (600/100) | Deliberate baseline; structure-aware chunking is on the roadmap |
| Dense embeddings | `BAAI/bge-small-en-v1.5` | 384-dim, CPU-friendly, strong retrieval baseline |
| Sparse retrieval | `BM25Okapi` (rank_bm25) | Exact keyword matching; complements dense for vocabulary gaps |
| Fusion | Reciprocal Rank Fusion (k=60) | Merges ranked lists by rank position, avoiding score-scale incompatibility |
| Vector index | FAISS `IndexFlatIP` | Exact search at 630 vectors; graduate to HNSW if corpus grows |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reads query + chunk together; more accurate than bi-encoder comparison |
| Generation | Gemini 2.5 Flash, `temperature=0.1` | Low randomness for factual compliance Q&A |
| Generation eval judge | Gemini 3 Flash Preview | LLM-as-a-judge for faithfulness and refusal accuracy scoring |
| API | FastAPI + uvicorn | Pydantic validation, startup loading, auto-generated OpenAPI docs |
| Observability | LangFuse v2 | Per-query traces with scores, tags, sessions |

## Observability

Every retrieval eval run is traced in LangFuse. Each question becomes a trace with:

- **Input:** question text
- **Output:** retrieved chunks (chunk_id, score, rank) + expected ground truth
- **Scores:** `hit`, `recall`, `rr`, `max_score`
- **Tags:** `category:*`, `top_k:*`, `model:*`, `git_sha:*`
- **Session:** one eval run = one session, all 33 traces grouped

This makes it possible to compare runs over time and drill into specific question failures.

![LangFuse trace view](docs/img/langfuse_traces.png)

## Docker

Three run modes from one image:

```bash
docker build -t tgd-part-m-rag .

docker run --rm --env-file .env tgd-part-m-rag                      # retrieval eval (default)
docker run --rm --env-file .env tgd-part-m-rag gen-eval             # generation eval
docker run --rm --env-file .env -p 8000:8000 tgd-part-m-rag api    # API server
```

The image pre-downloads BGE-small and cross-encoder models so there are no runtime HuggingFace downloads. Secrets are injected at runtime via `--env-file`, never baked into the image.

## CI/CD

GitHub Actions runs on every push to main and every PR:

1. Builds the Docker image
2. Runs retrieval eval — fails if Hit Rate, Recall, or MRR drops beyond tolerance vs `eval/baseline.json`
3. Runs generation eval — fails if faithfulness or refusal accuracy drops vs `eval/gen_baseline.json`
4. Uploads both result sets as downloadable artifacts

The retrieval gate is hard-blocking. The generation gate uses `continue-on-error: true` so judge API unavailability never blocks the retrieval gate.

## Roadmap

- [x] Baseline pipeline (extract → chunk → embed → retrieve → generate)
- [x] Refusal pathway (off-topic and unanswerable questions)
- [x] Retrieval evaluation harness — 33-question ground-truth set
- [x] Observability via LangFuse (per-query traces, scored metrics, session grouping)
- [x] Containerization (Docker, three run modes)
- [x] CI/CD with regression-gated retrieval and generation eval (GitHub Actions)
- [x] Two-stage retrieval — cross-encoder reranker + BM25 hybrid search
- [x] FastAPI server with `/health` and `/query` endpoints + OpenAPI docs
- [x] Generation eval — LLM-as-a-judge (faithfulness + refusal accuracy)

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
# Edit .env:
#   GEMINI_API_KEY=...              answerer model
#   GEMINI_JUDGE_API_KEY=...        judge model for generation eval
#   GEMINI_JUDGE_MODEL=...          e.g. gemini-3-flash-preview
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

# 3. Run the retrieval eval
python -m eval.run_eval                  # default top_k=10
python -m eval.run_eval --top-k 5

# 4. Run the generation eval
python -m eval.run_gen_eval

# 5. Start the API server
uvicorn src.api:app --reload
# → GET  localhost:8000/health
# → POST localhost:8000/query   body: {"question": "...", "top_k": 10}
# → GET  localhost:8000/docs    interactive Swagger UI

# OR run any mode in Docker (matches CI exactly)
docker run --rm --env-file .env tgd-part-m-rag                      # retrieval eval
docker run --rm --env-file .env tgd-part-m-rag gen-eval             # generation eval
docker run --rm --env-file .env -p 8000:8000 tgd-part-m-rag api    # API server
```

Results land in `eval/results/` (retrieval) or `eval/gen_results/` (generation), timestamped by git SHA.

## Repo structure

```
.
├── src/
│   ├── extract_pdf.py           PDF text extraction
│   ├── chunk_text.py            Fixed-size chunking
│   ├── build_index.py           FAISS index construction
│   ├── retrieval.py             Dense retrieval (BGE-small + FAISS)
│   ├── hybrid.py                BM25 sparse retrieval + Reciprocal Rank Fusion
│   ├── reranker.py              Cross-encoder reranking
│   ├── generate.py              Gemini generation with cited answer
│   └── api.py                   FastAPI server (/health, /query)
├── data/                        PDF + index artifacts (large files gitignored)
├── eval/
│   ├── questions.jsonl          33-question ground-truth set
│   ├── run_eval.py              Retrieval eval — Hit Rate, Recall, MRR
│   ├── check_regression.py      Retrieval regression gate
│   ├── baseline.json            Retrieval metrics baseline for CI
│   ├── run_gen_eval.py          Generation eval — faithfulness + refusal accuracy
│   ├── check_gen_regression.py  Generation regression gate
│   ├── gen_baseline.json        Generation metrics baseline for CI
│   ├── results/                 Timestamped retrieval eval results
│   └── gen_results/             Timestamped generation eval results
├── .github/workflows/eval.yml   CI — retrieval + generation gates on every PR
├── Dockerfile                   Multi-mode image (eval / gen-eval / api)
├── docker-entrypoint.sh         Routes run mode at container start
├── .dockerignore
├── .env.example
└── requirements.txt
```

