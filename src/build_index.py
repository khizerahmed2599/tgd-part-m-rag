import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "data/index.faiss"
METADATA_PATH = "data/index_metadata.json"


def build_index():
    """Embed all chunks and build a FAISS index for cosine similarity search."""
    # 1. Load chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"\n📦 Loaded {len(chunks)} chunks from {CHUNKS_PATH}")
    
    # 2. Load model
    model = SentenceTransformer(MODEL_NAME)
    
    # 3. Embed all chunk texts in one batch
    chunk_texts = [c["text"] for c in chunks]
    print(f"\n🔤 Embedding {len(chunk_texts)} texts with {MODEL_NAME}...")
    embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=32)
    print(f"✅ Embedding complete")
    
    # 4. Normalize embeddings to unit length
    faiss.normalize_L2(embeddings)

    # 5. Build IndexFlatIP with dim = embedding dim
    dim = embeddings.shape[1]
    print(f"📐 Embedding dimension: {dim}")
    
    index = faiss.IndexFlatIP(dim)
    
    # 6. Add vectors to index
    index.add(embeddings)
    print(f"📊 Index total vectors: {index.ntotal}")
    
    # 7. Save index and metadata
    faiss.write_index(index, INDEX_PATH)
    print(f"💾 Saved index to {INDEX_PATH}")
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved metadata to {METADATA_PATH}")
    
    return len(chunks), dim, index.ntotal

if __name__ == "__main__":
    print("=" * 60)
    print("Building FAISS Index...")
    print("=" * 60)
    
    num_chunks, embedding_dim, total_vectors = build_index()
    
    print("\n" + "=" * 60)
    print("✨ INDEX BUILD SUMMARY")
    print("=" * 60)
    print(f"Total chunks processed:  {num_chunks}")
    print(f"Embedding dimension:     {embedding_dim}")
    print(f"Vectors in index:        {total_vectors}")
    print(f"Index saved to:          {INDEX_PATH}")
    print(f"Metadata saved to:       {METADATA_PATH}")
    print("=" * 60)