# TGD Part M RAG — Technical Reference

A retrieval-augmented generation system over Ireland's *Technical Guidance
Document M (Access and Use, 2022)*. This document explains what every part
of the system does and *why* it does it that way, as a lookup reference
for future work.

> **What this document is.** A reference for re-orienting yourself to the
> codebase later. Not a tutorial — it assumes you've already built each
> stage. Read straight through once; come back to specific sections as
> needed.

---

## Table of contents

1. [What is RAG?](#what-is-rag)
2. [Why retrieval, not context-stuffing?](#why-retrieval-not-context-stuffing)
3. [The pipeline at a glance](#the-pipeline-at-a-glance)
4. [Stage 1 — PDF extraction](#stage-1--pdf-extraction)
5. [Stage 2 — Chunking](#stage-2--chunking)
6. [Stage 3 — Embeddings & FAISS index](#stage-3--embeddings--faiss-index)
7. [Stage 4 — Retrieval](#stage-4--retrieval)
8. [Stage 5 — Generation (planned)](#stage-5--generation-planned)
9. [Operations](#operations)
10. [Failure modes catalog](#failure-modes-catalog)
11. [Improvement roadmap](#improvement-roadmap)
12. [Glossary](#glossary)

---

## What is RAG?

**Retrieval-Augmented Generation** is a pattern for getting an LLM to
answer questions using information it wasn't trained on, without
fine-tuning or retraining the model.

The LLM doesn't know about TGD Part M. We have a 180-page PDF that does.
RAG bridges the gap: at query time, we *retrieve* the relevant pieces of
the PDF and *give them to the LLM* as part of the prompt. The LLM uses
those pieces (the "context") to generate an answer.

There are three ways to make an LLM "know" something it didn't learn:

| Approach        | What it is                                                                  | When to use                                            |
|-----------------|-----------------------------------------------------------------------------|--------------------------------------------------------|
| **Pre-train**   | Train a foundation model from scratch on your data. Months, $millions.      | Almost never; only frontier labs do this.              |
| **Fine-tune**   | Adjust a pre-trained model's weights using your data. Hours-days, $hundreds.| Style adaptation, tone, narrow domains where retrieval can't reach. |
| **Retrieve**    | Look up relevant info at query time and put it in the prompt. Minutes, ~free.| Most knowledge-intensive applications. **What we're doing.** |

Retrieval wins for our case because:

- **Freshness.** When TGD Part M is updated, we re-index. No retraining.
- **Auditability.** Every answer can cite the chunks it came from. For
  compliance work this is mandatory.
- **Cost.** Embedding 630 chunks costs cents. Fine-tuning costs hundreds.
- **Control.** If a chunk is wrong, you fix the chunk. With fine-tuning,
  bad data is permanent until the next training run.

---

## Why retrieval, not context-stuffing?

A fair question: modern LLMs (Gemini 1.5, Claude 3.5, GPT-4o) have very
long context windows. Why not just dump the whole 180-page document into
the prompt and let the model figure it out?

Three reasons:

**1. Cost and latency scale with context.** Every query re-processes the
entire document. At thousands of queries per day this is expensive and
slow. Retrieval sends ~3,000 tokens (the chunks + question) instead of
80,000+ (the whole doc).

**2. Long context degrades quality ("lost in the middle").** Even when
models *can* handle long context, performance drops on information located
in the middle of the input. A small focused context outperforms a large
diluted one.

**3. Auditability.** When the system says *"corridors must be 1200 mm wide
per Section 1.3.3.3 (page 65),"* a regulator can verify the citation. If
the LLM "knows" it because we dumped 180 pages in, there's no audit trail.

For compliance use cases, **(3) is decisive.** Retrieval isn't a
workaround for limited context — it's the architecturally correct choice.

---

## The pipeline at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│                       BUILD-TIME (once)                         │
└─────────────────────────────────────────────────────────────────┘

  TGD_Part_M.pdf
        │
        ▼
  ┌────────────────┐
  │ extract_pdf.py │   pypdf reads PDF → list of {page, text}
  └────────────────┘
        │
        ▼
  ┌────────────────┐
  │ chunk_text.py  │   600-char windows w/ 100-char overlap
  └────────────────┘   → data/chunks.json (630 chunks)
        │
        ▼
  ┌────────────────┐
  │ build_index.py │   BGE-small embeds each chunk → 384-dim
  └────────────────┘   → data/index.faiss + index_metadata.json

┌─────────────────────────────────────────────────────────────────┐
│                  QUERY-TIME (every question)                    │
└─────────────────────────────────────────────────────────────────┘

  User query: "How wide should a corridor be?"
        │
        ▼
  ┌────────────────┐
  │   retrieve.py  │   Embed query → FAISS top-k → chunks + scores
  └────────────────┘
        │
        ▼
  ┌────────────────┐
  │  generate.py   │   (Stage 5) Gemini receives query + chunks
  └────────────────┘   → grounded answer with citations
```

Each stage produces an artifact in `data/` consumed by the next stage.
All artifacts are gitignored — they are *derived*, not source. The PDF
itself is also gitignored (licensing).

---

## Stage 1 — PDF extraction

**File.** `src/extract_pdf.py`
**Library.** `pypdf`
**Output (in-memory).** `list[{"page": int, "text": str}]`

### What it does

Opens the PDF, walks every page, calls `page.extract_text()`, and returns
a list of dicts pairing 1-indexed page numbers with extracted text. Pages
that return `None` (some diagram-only pages) are coerced to empty strings
so downstream code doesn't crash.

### Why PDFs are hard

A PDF is **not a structured document**. It's a sequence of instructions
for rendering glyphs at coordinates on pages. Headings, paragraphs,
tables, footnotes — none of those are explicitly marked in the file
format. They're visual conventions.

Text extraction is therefore *guessing*: the library reads the rendering
instructions and tries to reconstruct reading order from glyph positions.
Different libraries make different guesses:

| Library     | Speed | Tables | Multi-column | Notes                          |
|-------------|-------|--------|--------------|--------------------------------|
| `pypdf`     | Fast  | Poor   | Poor         | Simple, our baseline           |
| `pdfplumber`| Slow  | Good   | OK           | Best for tables                |
| `PyMuPDF`   | Fast  | OK     | Good         | Strong general-purpose option  |
| OCR (`tesseract`) | Slow | n/a | n/a       | Required for scanned PDFs      |

We use `pypdf` for the baseline because it's simple and ships in pure
Python. If table extraction becomes important, we'll evaluate alternatives.

### Page-numbering ambiguity

Three different "page numbers" exist for any PDF:

1. **PDF page number** — physical sheet position. Always 1, 2, 3, ...
2. **Document page number** — what's printed in the page footer. May
   skip, restart, or use roman numerals (front matter).
3. **Index position** — Python's 0-indexed list position.

We use **PDF page numbers, 1-indexed**. This is consistent and
deterministic but doesn't always match the page numbers a user sees in
the printed document. This will need reconciliation when citations are
shown to end users. It's the same source of truth as your previous
system's "page-offset" bug — track it carefully.

### Example output

For TGD Part M:

```python
[
    {"page": 1,   "text": "Building Regulations\nTechnical Guidance Document M\n2022\n..."},
    {"page": 2,   "text": "© Government of Ireland 2022"},
    {"page": 3,   "text": "Contents\nIntroduction ........... 1\n..."},
    ...
    {"page": 65,  "text": "1.3.3.3 Corridors and passageways\nA corridor or passageway should be wide enough..."},
    ...
    {"page": 180, "text": "gov.ie/housing\nDepartment of Housing, Local Government and Heritage"},
]
```

180 dicts. ~270,000 total characters.

### Known limitations

- Tables and multi-column layouts may extract with mangled reading order.
- Diagrams come through as nothing (we get an empty string).
- Front matter and back matter (cover, copyright, TOC, references) are
  treated like content.
- Whitespace artifacts (margin gaps, line spacing) flow through as
  newlines and runs of spaces.

These are deliberately *not* fixed in the baseline. They become measurable
problems once the eval harness exists; that's when we fix them.

---

## Stage 2 — Chunking

**File.** `src/chunk_text.py`
**Output.** `data/chunks.json`

### Why we chunk

A single embedding for 180 pages is useless — it averages too many topics
into one vector. A single embedding per word is also useless — there's no
context. The unit of embedding has to be roughly *one coherent idea*.

For regulations, that's a clause or a paragraph. As a baseline, we
approximate this with **fixed-character chunks**: take 600 characters,
slide forward, repeat.

### The size-vs-context tradeoff

| Chunk size  | Risk                                                            |
|-------------|-----------------------------------------------------------------|
| 100 chars   | Too small — single sentences lose surrounding context.          |
| 500–800     | Sweet spot for most prose. Our default: 600.                    |
| 2000+ chars | Too big — embedding becomes a blurry average of multiple topics.|

### How overlap works

A 500-char chunk with 100-char overlap means the *next* chunk starts at
character 400, not 500. Visually:

```
Page text:
[───────────────────────────── 1500 chars ─────────────────────────────]

Chunk 0: chars 0–500
[──────── 500 ────────]

Chunk 1: chars 400–900     (overlaps 100 chars with chunk 0)
                [──────── 500 ────────]

Chunk 2: chars 800–1300
                                [──────── 500 ────────]

Chunk 3: chars 1200–1500   (truncated at end)
                                                [── 300 ──]
```

The window advances by `chunk_size - overlap` per step. Information that
straddles a chunk boundary is preserved in *both* neighbors — so it's
retrievable from either one. Cost: storage and compute (the overlap
region gets embedded twice). Benefit: robustness against boundary effects.

### Chunk ID convention

Each chunk gets a stable, human-readable ID:

```
p<page>_c<index>     e.g.   p72_c1   = page 72, second chunk on that page
```

The index resets per page. This matters for debugging — when retrieval
returns the wrong chunk, the ID tells you *exactly* where in the PDF
to look. A flat global counter (`chunk_437`) tells you nothing about
location.

### Output structure

```python
[
    {"chunk_id": "p1_c0", "page": 1,  "text": "..."},
    {"chunk_id": "p2_c0", "page": 2,  "text": "..."},
    {"chunk_id": "p72_c0", "page": 72, "text": "..."},
    {"chunk_id": "p72_c1", "page": 72, "text": "..."},
    ...
]
```

For TGD Part M with `chunk_size=600, overlap=100`: ~630 chunks total.

### Known limitations

- **Mid-word cuts.** A 600-char window doesn't respect word, sentence,
  or clause boundaries. The chunker may split `1200 mm` between chunks.
- **Page-boundary resets.** A clause that crosses a page break gets
  fragmented because the chunker resets per page.
- **No structural awareness.** Section headers, clause numbers, and
  hierarchy aren't preserved in chunks beyond what naturally fits in
  the window.
- **Front matter is chunked too.** Cover, TOC, and references all become
  retrievable chunks even though they're useless for compliance queries.

---

## Stage 3 — Embeddings & FAISS index

**File.** `src/build_index.py`
**Model.** `BAAI/bge-small-en-v1.5`
**Output.** `data/index.faiss`, `data/index_metadata.json`

### What an embedding is (recap)

An embedding model maps text to a fixed-size vector of numbers. For
BGE-small: text in → 384 numbers out. The numbers themselves are not
human-interpretable. What matters is the *geometry*: texts about similar
topics produce numerically similar vectors.

```
"corridor width 1200 mm"      →  [0.12, -0.45,  0.78, ...,  0.03]
"minimum passageway dimension"→  [0.10, -0.43,  0.81, ...,  0.05]   ← close
"VAT registration thresholds" →  [-0.62, 0.31, -0.04, ..., -0.55]   ← far
```

Each text becomes one point in a 384-dimensional space. Search becomes
"find the nearest points." We can't visualize 384 dimensions, but the
math works the same as in 2D or 3D:

```
       ●  "corridor width 1200 mm"
     ●  "minimum passageway dimension"
   ●  "wheelchair accessible route"
            ●  "stair handrail height"
                       ●  "WC dimensions"


       ●  "VAT registration thresholds"
                ●  "tax compliance dates"
```

(Imagined as 2D. The clusters represent semantic neighborhoods.)

### Why normalize?

We want **cosine similarity** (angle between vectors), not Euclidean
distance. Cosine compares *direction* (meaning) and ignores *magnitude*
(which correlates with text length and other noise).

FAISS doesn't have a built-in cosine index, but it has `IndexFlatIP`
(inner product = dot product). The trick:

```
If a and b are unit vectors (magnitude 1):
    cosine(a, b)  ==  a · b
```

So we normalize every vector to unit length, use `IndexFlatIP`, and
the inner product *is* the cosine similarity. Standard pattern.

### Why `IndexFlatIP` and not something fancier?

FAISS offers many index types with different speed/accuracy tradeoffs:

| Index           | Speed       | Recall  | When to use                       |
|-----------------|-------------|---------|-----------------------------------|
| `IndexFlatIP`   | Slow at scale | 100%    | <100K vectors. **Our baseline.**  |
| `IndexIVFFlat`  | Fast        | ~95%    | Millions of vectors.              |
| `IndexHNSW`     | Very fast   | ~98%    | Production, billions of vectors.  |

For 630 vectors, exact brute-force search is instant. Premature
optimization to a smarter index would just hide the real performance
characteristics. We'll graduate to `IndexHNSW` if and when corpus size
demands it.

### Why two output files?

```
data/index.faiss            ← just the 384-dim vectors
data/index_metadata.json    ← chunk_id, page, text — same order as vectors
```

FAISS stores **only numbers**. It doesn't know about your text. When
search returns "row 47," you look up `metadata[47]` to get the actual
chunk content.

**Critical invariant: the order must match exactly.** If you reshuffle
one without the other, retrieval silently returns wrong text — same
chunk_id, wrong content, no error. This was the failure mode in the
previous compliance system. It manifests as "retrieval scores look fine,
but answers are nonsensical."

```
            FAISS index                metadata list
            ┌────────────┐             ┌──────────────────┐
       Row 0│ [v0_0...v0_383]        │ {p1_c0, page=1, "..."}
       Row 1│ [v1_0...v1_383]        │ {p1_c1, page=1, "..."}
       Row 2│ [v2_0...v2_383]        │ {p2_c0, page=2, "..."}
       ...                            ...
       Row N│ [vN_0...vN_383]        │ {p180_c0, page=180,"..."}
            └────────────┘             └──────────────────┘
                  ↑                              ↑
                  └──── must align by index ─────┘
```

### Known limitations

- **One model per index.** Embeddings from BGE-small and embeddings from
  another model are not comparable. Mixing them silently corrupts results.
  `MODEL_NAME` is defined as a constant; **never override it ad-hoc**.
- **Re-indexing is required when chunks change.** A change to chunking
  parameters means stale FAISS index. There's no automatic invalidation.

---

## Stage 4 — Retrieval

**File.** `src/retrieve.py`
**Output.** `list[{"chunk_id", "page", "text", "score"}]`

### Asymmetric models and the query prefix

BGE was trained with separate query and passage modes. To retrieve
correctly, the **query** must be prefixed:

```python
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
```

Passages (the chunks at index time) get no prefix. The model's training
distinguished the two roles — skipping the prefix at query time degrades
results measurably.

This is documented in the BGE model card. Other embedding models have
different conventions:

| Model family            | Query prefix needed?                         |
|-------------------------|----------------------------------------------|
| BGE (`BAAI/bge-*`)      | Yes — `"Represent this sentence..."`         |
| E5 (`intfloat/e5-*`)    | Yes — `"query: "` for queries, `"passage: "` for chunks |
| OpenAI `text-embedding-3-*` | No (symmetric)                          |
| Cohere `embed-v3`       | Yes — different `input_type` parameter       |

If you change models, **read the model card**. Wrong prefix = silently
broken retrieval.

### The retrieval flow

```
query  ──►  prefix  ──►  embed  ──►  normalize  ──►  faiss.search(k)
                                                          │
                                                          ▼
                                             (scores, indices)
                                                          │
                                                          ▼
                                          metadata[idx] for each idx
                                                          │
                                                          ▼
                                  list of {chunk_id, page, text, score}
```

`index.search(query_vec, k)` returns two arrays of shape `(1, k)`:

- `scores[0]` — k similarity scores, sorted descending
- `indices[0]` — k row numbers into the index/metadata

### Reading scores

| Score range | Interpretation                                              |
|-------------|-------------------------------------------------------------|
| 0.85 – 1.00 | Very confident. Strong semantic match.                       |
| 0.65 – 0.85 | Confident. Almost always relevant.                           |
| 0.45 – 0.65 | Uncertain. Could be relevant or off-topic.                   |
| 0.25 – 0.45 | Weak. Likely paraphrase mismatch or chunking issue.          |
| < 0.25      | Probably not in corpus, or embedding model lacks the domain. |

**Scores are necessary but not sufficient.** A high score means *the
model is confident the query and chunk are about the same thing.* It
does **not** guarantee correctness. Three failure modes:

1. **Confident wrong.** Top result has score 0.78 and is about the
   wrong topic. The embedding model has decided two things are similar
   that, semantically, aren't. Diagnose by reading the chunks.

2. **Unconfident right.** The right chunk is at rank 5 with score 0.62.
   A reranker would fix this. Diagnose by manually finding the
   ground-truth chunk and checking its score.

3. **Silently empty.** All scores are below your threshold (e.g., 0.35),
   so the system filters everything and returns "no relevant clauses
   found." This was the old system's bug. Diagnose by *always* logging
   raw scores before any threshold filter.

### Example: real query output

Query: *"What is the minimum width of a corridor for wheelchair access?"*

| Rank | Score | Chunk ID | Page | Snippet                                          |
|------|-------|----------|------|--------------------------------------------------|
| 1    | 0.844 | p37_c1   | 37   | "passing places for wheelchair users 2000×1800"  |
| 2    | 0.838 | p74_c0   | 74   | "long corridors over 20 m, minimum 1200 mm"      |
| 3    | 0.832 | p37_c3   | 37   | "passing places, 25 m max spacing"               |
| 4    | 0.809 | p72_c0   | 72   | "1.3.3.3 Corridors and passageways"              |
| 5    | 0.808 | p72_c1   | 72   | "**unobstructed clear width should be at least 1200 mm**" |

Notes on this output:

- Scores are tightly clustered (0.81–0.84). The model is confident about
  all five.
- The *canonical answer* (1200 mm clear width, p72_c1) is at **rank 5**.
  The top-ranked chunks are about adjacent topics (passing places).
- This is the **chunk-dilution problem**: the canonical chunk has more
  surrounding context that dilutes its embedding.
- A reranker model (e.g., `bge-reranker-base`) would re-score the top-K
  using a more careful comparison and likely promote `p72_c1` to rank 1.

This is exactly why we need an eval harness — to *measure* this kind of
near-miss systematically rather than spotting it by luck.

---

## Stage 5 — Generation (planned)

The final stage takes `query + retrieved_chunks` and asks an LLM
(Gemini, in our case) to compose a grounded answer.

### Prompt structure

This where the magic lies, the model understands based on the instructions given to it. There are further details on how to design this mentioned.

```
SYSTEM:
You are an assistant for compliance with TGD Part M (Ireland's
disability access regulations). Answer using only the provided
clauses. Cite the clause and page for every claim. If the clauses
do not contain the answer, say so explicitly — do not guess.

USER:
[CLAUSES]
1. (page 72, p72_c1) The unobstructed clear width should be at
   least 1200 mm. Elements such as columns, radiators...
2. (page 72, p72_c0) 1.3.3.3 Corridors and passageways. A corridor
   or passageway should be wide enough...
... (k clauses)

[QUESTION]
What is the minimum width of a corridor for wheelchair access?
```

### Key design decisions (TBD)

- **Citation format.** `[page X, chunk_id Y]` inline, or footnote-style?
- **"I don't know" handling.** The system must refuse to answer when the
  retrieved chunks don't support a claim. This is non-trivial — LLMs
  often hallucinate confidence. Explicit instruction + low temperature
  helps but doesn't guarantee.
- **Output format.** Free text, structured JSON, or both?
- **Refusal of non-Part-M questions.** "What's the capital of France?"
  should not invoke retrieval and should not be answered. Detection
  is a separate problem (query routing).

### Faithfulness measurement

Once generation is in place, the metrics that matter are:

- **Citation accuracy.** Does every claim have a chunk that supports it?
- **Faithfulness.** Are the citations *actually* from the retrieved
  chunks (not hallucinated)?
- **Answer relevance.** Does the answer address the question?
- **Refusal rate on out-of-scope queries.**

Tools: RAGAS, TruLens, or a custom harness. To be specified when
Stage 5 is implemented.

---

## Operations

### Folder layout

```
tgd-part-m-rag/
├── data/
│   ├── README.md                   ← how to obtain the PDF
│   ├── tgd_part_m_2022.pdf         ← gitignored
│   ├── chunks.json                 ← gitignored (derived)
│   ├── index.faiss                 ← gitignored (derived)
│   └── index_metadata.json         ← gitignored (derived)
├── docs/
│   └── REFERENCE.md                ← this document
├── src/
│   ├── extract_pdf.py
│   ├── chunk_text.py
│   ├── build_index.py
│   ├── retrieve.py
│   └── generate.py                 ← Stage 5
├── tests/                          ← (planned)
├── .env                            ← gitignored
├── .env.example
├── .gitignore
├── notes.md                        ← personal learning notes
├── README.md
└── requirements.txt
```

### Rebuild from scratch

```bash
# 1. Clone and set up environment
git clone https://github.com/khizerahmed2599/tgd-part-m-rag.git
cd tgd-part-m-rag
python -m venv venv
venv\Scripts\Activate.ps1                # Windows
source venv/bin/activate                 # macOS/Linux
pip install -r requirements.txt

# 2. Place the PDF
# Download per data/README.md, save as data/tgd_part_m_2022.pdf

# 3. Set up secrets
cp .env.example .env                     # then edit GEMINI_API_KEY

# 4. Build artifacts
python src/chunk_text.py                 # → data/chunks.json
python src/build_index.py                # → data/index.faiss + metadata

# 5. Test retrieval
python src/retrieve.py
```

### Constants reference

| Constant         | Defined in              | Value                                                          |
|------------------|--------------------------|----------------------------------------------------------------|
| `MODEL_NAME`     | `build_index.py`, `retrieve.py` | `BAAI/bge-small-en-v1.5`                                |
| `QUERY_PREFIX`   | `retrieve.py`            | `"Represent this sentence for searching relevant passages: "`  |
| `CHUNKS_PATH`    | `chunk_text.py`, `build_index.py` | `data/chunks.json`                                    |
| `INDEX_PATH`     | `build_index.py`, `retrieve.py` | `data/index.faiss`                                      |
| `METADATA_PATH`  | `build_index.py`, `retrieve.py` | `data/index_metadata.json`                              |
| `chunk_size`     | `chunk_text.py`          | `600`                                                          |
| `overlap`        | `chunk_text.py`          | `100`                                                          |

> **Eventual refactor.** These are duplicated across files. Once a 6th
> file appears, lift them into `src/config.py` and import. Doing it now
> is premature; doing it never is technical debt.

### Reproducibility checklist

- [ ] `requirements.txt` pinned to exact versions
- [ ] `MODEL_NAME` constant referenced everywhere (no string literals)
- [ ] PDF version recorded somewhere (filename includes year)
- [ ] Index rebuilt from scratch produces identical retrieval results
      (sanity check)
- [ ] No absolute paths in source

---

## Failure modes catalog

A reference for diagnosing strange behavior. Each entry: what it looks
like, how to confirm, how to fix.

### 1. Page-numbering offset

**Symptom.** Citations point to wrong pages — off by 1 or 2 from what
the user expects.

**Root cause.** Confusion between PDF page (physical), document page
(printed footer), and Python list index (0-based).

**Diagnose.** Pick a chunk, find its content in the PDF manually, compare
to the `page` field.

**Fix.** Use 1-indexed PDF pages consistently. Front matter is acknowledged
to have mismatched document page numbers; this is a known limitation.

---

### 2. FAISS index drift (model mismatch)

**Symptom.** Scores look reasonable (0.4–0.7) but retrieved chunks are
unrelated to the query. Or worse: scores are extremely low (<0.3) and
the system filters everything.

**Root cause.** The embedding model used at *index time* differs from
the one used at *query time*. The vectors are in incompatible "languages."

**Diagnose.** Print `MODEL_NAME` from both `build_index.py` and
`retrieve.py`. They must be byte-identical.

**Fix.** Define `MODEL_NAME` once, in a shared module if necessary.
Re-index whenever it changes.

---

### 3. TOC and front-matter pollution

**Symptom.** Top retrieval results include the table of contents,
copyright page, or back cover. High keyword overlap, no semantic value.

**Root cause.** Chunker indexes every page, including non-content pages.

**Diagnose.** Check chunk IDs of top results. `p1_c0`–`p10_c0` are
suspicious for most documents.

**Fix (post-baseline).** Pre-process to detect and skip front matter
(by content patterns, page ranges, or section markers). Measure impact
on eval set.

---

### 4. Whitespace pollution

**Symptom.** Some chunks contain mostly spaces and newlines with a few
fragments of header text. Chunks have low information density.

**Root cause.** PDF margins, headers, and inter-line spacing flow into
extracted text as whitespace runs.

**Diagnose.** Look at chunks with the lowest `len(text.strip())` values.

**Fix (post-baseline).** Normalize whitespace during extraction — collapse
consecutive whitespace, strip page-header artifacts.

---

### 5. Chunk dilution

**Symptom.** The canonical answer chunk ranks 4th or 5th instead of 1st.
Adjacent topical chunks rank higher.

**Root cause.** The canonical chunk contains the answer plus surrounding
context that dilutes the embedding. Shorter, denser chunks score higher.

**Diagnose.** Manually identify the ground-truth chunk. Check its rank
and score. If score is reasonable but rank is low, this is dilution.

**Fix (post-baseline).** Add a reranker (e.g., `bge-reranker-base`) that
takes top-K and re-orders using cross-encoder scoring. Or use
contextual retrieval (Anthropic-style chunk prefixing).

---

### 6. Confident wrong retrieval

**Symptom.** All top results have high scores (0.7+) but they're
about the wrong topic.

**Root cause.** The embedding model has its own notion of similarity
that doesn't match yours. Common with domain jargon ("WC" vs "toilet").

**Diagnose.** Read the top chunks. Try query variants ("WC", "toilet",
"sanitary accommodation"). If different phrasings retrieve different
chunks, the embedding model is sensitive to vocabulary.

**Fix (post-baseline).** Hybrid search — combine BGE (semantic) with
BM25 (keyword) using reciprocal rank fusion. Or query expansion.

---

### 7. Silent empty retrieval

**Symptom.** Every query returns "no results" or empty string answers.

**Root cause.** Score threshold filters out all results. Often combined
with a model-mismatch bug that crushes all scores.

**Diagnose.** Remove the threshold temporarily. Print raw top-5 with
scores. If scores are 0.1–0.3, retrieval is fundamentally broken.

**Fix.** Diagnose the underlying issue (model mismatch, wrong index file,
bad embeddings). Don't lower the threshold blindly — that hides bugs.

---

## Improvement roadmap

In order of priority. Each step's success is measured against the
**eval harness** (the next step after baseline).

### 1. Eval harness — *the next thing*

Hand-craft 25–30 questions with ground-truth chunk IDs. Build a
script that runs each question through retrieval and computes:

- **Recall@k** — does the right chunk appear in the top-k?
- **MRR** (Mean Reciprocal Rank) — average of `1/rank` for the right chunk.
- **Hit rate** — fraction of queries with the right chunk in top-k.

Without this, every "improvement" below is guesswork.

### 2. Whitespace normalization & front-matter detection

Cheap fixes likely to help a lot. Measure impact, document the gain.

### 3. Structure-aware chunking

Detect clause boundaries (e.g., "1.3.3.3 ...") and chunk by clause
rather than fixed character count. Preserve headers as metadata.

### 4. Reranking

Add a cross-encoder reranker on top-20. Should fix the chunk-dilution
problem documented above.

### 5. Hybrid search (BM25 + dense)

Combine semantic and keyword retrieval via reciprocal rank fusion.
Fixes terminology mismatches (WC vs toilet).

### 6. Contextual retrieval

Per Anthropic's blog: prefix each chunk with a short summary of where
it fits in the document, then embed. Improves retrieval significantly
on hierarchical documents.

### 7. Query decomposition / agentic retrieval

For complex queries spanning multiple clauses, break the query into
sub-queries, retrieve each, and synthesize. Optionally let the LLM
decide *when* to retrieve again.

### 8. GraphRAG for cross-references

Regulations have explicit cross-references ("subject to §1.3.3.3").
Build a graph of clause-to-clause references and traverse it during
retrieval.

---

## Glossary

**BM25** — A classical keyword-based ranking function used in
search engines for decades. Exact-match and term-frequency-based.
Complementary to dense embeddings.

**Chunk** — A small piece of source text (typically a paragraph or
~500 chars) that gets independently embedded and indexed.

**Cosine similarity** — Measure of similarity between two vectors based
on the angle between them. Invariant to magnitude. Range: -1 (opposite)
to 1 (identical direction).

**Cross-encoder** — A model that takes two pieces of text *together*
and outputs a similarity score. More accurate than dense embeddings
(bi-encoders) but slower. Used for reranking.

**Dense retrieval** — Retrieval using neural embeddings (vectors of
floats). Captures semantic similarity.

**Embedding** — A fixed-size numerical vector representation of text,
produced by a learned model. For BGE-small: 384 dimensions.

**FAISS** — Library by Facebook for efficient similarity search over
dense vectors. Stores numbers; not text.

**Inner product** — `a · b = Σ aᵢbᵢ`. Equals cosine similarity when
both vectors are unit length.

**MRR (Mean Reciprocal Rank)** — Evaluation metric. For each query,
take `1/rank` of the first correct result; average across queries.
Range: 0 to 1.

**Recall@k** — Fraction of queries where the correct result appears
in the top k retrieved items.

**Reranker** — A model that reorders an initial top-k retrieval by
applying more careful (and expensive) scoring per candidate.

**Sparse retrieval** — Retrieval using bag-of-words / token-frequency
methods (BM25, TF-IDF). Captures exact-keyword similarity.

**Top-k retrieval** — Returning the k most similar items to a query.
Typical k: 3–10.

**Vector index** — A data structure (FAISS, Chroma, Pinecone, etc.)
optimized for nearest-neighbor search over dense vectors.

---

*Document version: 1.0. Updated as the system grows.*