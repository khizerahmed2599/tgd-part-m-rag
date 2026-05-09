import json

from extract_pdf import extract_pages


def chunk_pages(pages: list[dict], chunk_size: int = 600, overlap: int = 100) -> list[dict]:
    """Split each page's text into overlapping character-based chunks with stable IDs."""
    all_chunks = []
    step_size = chunk_size - overlap

    for page in pages:
        text = page["text"]
        chunk_index = 0
        start = 0
        while start < len(text):
            chunk_text = text[start:start + chunk_size]
            if chunk_text.strip():
                all_chunks.append({
                    "chunk_id": f"p{page['page']}_c{chunk_index}",
                    "page": page["page"],
                    "text": chunk_text,
                })
                chunk_index += 1
            start += step_size

    return all_chunks


if __name__ == "__main__":
    pdf_path = r"data/tgd_part_m_2022.pdf"
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, chunk_size=600, overlap=100)

    print(f"Total chunks: {len(chunks)}")
    avg_length = sum(len(c["text"]) for c in chunks) / len(chunks)
    print(f"Average chunk length: {avg_length:.0f} chars")
    print("\n--- First 3 chunks ---")
    for c in chunks[:3]:
        print(f"\n[{c['chunk_id']} | page {c['page']}]")
        print(c["text"])
        print("-" * 60)

    # Save chunks to JSON file
    output_path = "data/chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(chunks)} chunks to {output_path}")