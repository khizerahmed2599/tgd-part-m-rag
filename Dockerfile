# ----------------------------------------------------------------------------
# TGD Part M RAG — eval + API image
# Two run modes:
#   docker run --env-file .env tgd-part-m-rag                 # eval (default)
#   docker run --env-file .env -p 8000:8000 tgd-part-m-rag api
# ----------------------------------------------------------------------------

FROM python:3.12-slim

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer - only rebuilds if requirements.txt changes or if we add new files)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Pre-download the embedding model so it's baked inot the image
# # (otherwise the first container run pulls 30MB from HuggingFace, slow + needs network)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Pre-downlodad the reranker model
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

#Copy application code (last, becuase this is what changes most often)
COPY src/ ./src/
COPY eval/ ./eval/
COPY data/ ./data/

#Expose API port (only needed for API mode, but doesn't hurt in eval mode)
EXPOSE 8000

#Entry script handels both modes - eval(default) and API (if "api" arg is passed)
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]

# Default mode: eval
# Override with: docker run ... tgd-part-m-rag api
CMD ["eval"] 