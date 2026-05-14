"""
TGD Part M RAG — FastAPI application.

Two endpoints:
    GET  /health  — liveness check, confirms the server is running
                    and the retriever is loaded
    POST /query   — submit a question, get a cited answer
"""
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()   # load .env before any imports that need API keys

from src.retrieval import retrieve, MODEL_NAME
from src.reranker import load_reranker, rerank
from src.hybrid import build_bm25_index, bm25_retrieve, reciprocal_rank_fusion
from eval.run_eval import load_retriever   # reuse your existing loader
from src.generate import generate_answer          # your existing generation function


# ------------------------------------------------------------------
# Pydantic models — the shape of requests and responses
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,                          # ... means required, no default
        min_length=5,
        max_length=500,
        description="The compliance question to ask about TGD Part M.",
        examples=["What is the minimum corridor width for wheelchair access?"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Default 10, max 20.",
        examples=[10],
    )


class ChunkReference(BaseModel):
    chunk_id: str
    page: int
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    chunks_used: list[ChunkReference]
    retrieval_ms: float = Field(description="Time taken for retrieval in milliseconds.")


class HealthResponse(BaseModel):
    status: str
    retriever_loaded: bool
    model: str


# ------------------------------------------------------------------
# Lifespan — runs at startup and shutdown
# ------------------------------------------------------------------

# This dict is a simple way to share state between the lifespan
# and the request handlers. FastAPI's app.state works too.
state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup. Release at shutdown."""
    print("Loading retriever...")
    state["retriever"] = load_retriever(
        index_path="data/index.faiss",
        metadata_path="data/index_metadata.json",
    )
    print(f"Retriever loaded. {len(state['retriever']['metadata'])} chunks in index.")
    yield   # server runs here — everything between yield and the end is shutdown
    print("Shutting down.")
    state.clear()


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="TGD Part M RAG",
    description=(
        "Retrieval-augmented generation over Ireland's Technical Guidance "
        "Document M (Access and Use, 2022). Answers building accessibility "
        "compliance questions with cited clauses. "
        "Observability via LangFuse — every query is traced."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Liveness and readiness check",
)
def health():
    """Returns 200 if the server is running and the retriever is loaded.
    Returns 503 if the retriever failed to load at startup.
    """
    loaded = "retriever" in state
    if not loaded:
        raise HTTPException(
            status_code=503,
            detail="Retriever not loaded. Check server logs.",
        )
    return HealthResponse(
        status="ok",
        retriever_loaded=loaded,
        model=MODEL_NAME,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["rag"],
    summary="Ask a question about TGD Part M",
)
def query(request: QueryRequest):
    """Submit a question about Ireland's building accessibility regulation.

    Returns a cited answer grounded in the regulation text.
    If the question is off-topic or unanswerable from the regulation,
    returns an explicit refusal rather than a fabricated answer.
    """
    if "retriever" not in state:
        raise HTTPException(
            status_code=503,
            detail="Retriever not available.",
        )

    retriever = state["retriever"]

    # Stage 1: hybrid retrieval (dense + BM25 → RRF fusion)
    t0 = time.perf_counter()
    dense = retrieve(
        request.question,
        retriever["index"],
        retriever["metadata"],
        retriever["model"],
        top_k=request.top_k * 2,
    )
    sparse = bm25_retrieve(request.question, retriever["bm25"], retriever["metadata"], top_k=request.top_k * 2)
    candidates = reciprocal_rank_fusion([dense, sparse], top_k=request.top_k * 2)

    # Stage 2: rerank
    results = rerank(request.question, candidates, retriever["reranker"], top_k=request.top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Stage 3: generate answer
    answer = generate_answer(request.question, results)

    return QueryResponse(
        question=request.question,
        answer=answer,
        chunks_used=[
            ChunkReference(
                chunk_id=r["chunk_id"],
                page=r["page"],
                score=round(r["score"], 4),
            )
            for r in results
        ],
        retrieval_ms=round(retrieval_ms, 2),
    )