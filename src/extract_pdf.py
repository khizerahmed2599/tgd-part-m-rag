from pypdf import PdfReader

def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF, preserving page numbers."""
    reader = PdfReader(pdf_path)
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""   # Handle cases where text extraction might return None       
        pages.append({"page": index, "text": text})  
    return pages

if __name__ == "__main__":
    pdf_path = "C:\\Users\\BIYABANK\\Desktop\\KhizerCstuff\\CVs\\Interview Pre\\TGD_M RAG\\data\\TGDM_PDF_2022.pdf" 
    extracted_pages = extract_pages(pdf_path)
    print(f"Total pages: {len(extracted_pages)}")
    total_chars = 0
    for page in extracted_pages:
        total_chars += len(page["text"])
    print(f"Total characters: {total_chars}") # Print the first 200 characters of each page's text

    first_500 = extracted_pages[4]["text"][:500]  # Index 4 because 0-indexed
    print(first_500)

    empty_count = sum(1 for p in extracted_pages if not p["text"].strip())
    print(f"Empty pages: {empty_count}")
