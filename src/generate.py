import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from retrieval import load_retriever, retrieve

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = """You are a compliance assistant for TGD Part M, Ireland's
Technical Guidance Document on Access and Use of buildings for people
with disabilities.

Answer the user's question using ONLY the clauses provided below. Rules:

1. For every claim you make, cite the supporting clause inline using
   the format [chunk_id, page X]. Multiple citations are fine.
2. If the clauses do not contain enough information to answer the
   question, say so explicitly. Do not fall back on general knowledge.
3. Be concise. Quote exact figures (e.g., "1200 mm") directly from the
   clauses where relevant.
4. If the question is not about TGD Part M / building accessibility,
   politely refuse and explain that you only answer questions about
   this regulation."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build the user-message prompt from a query and retrieved chunks."""
    # 1. Format each chunk as: [N] (chunk_id: ..., page X) <text>
    formatted_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        formatted = f"[{i}] (chunk_id: {chunk['chunk_id']}, page {chunk['page']})\n{chunk['text']}"
        formatted_chunks.append(formatted)

    # 2. Combine into a single string with sections: CLAUSES, QUESTION
    clauses_section = "CLAUSES:\n" + "\n\n".join(formatted_chunks)
    question_section = f"QUESTION:\n{query}"
    prompt = f"{clauses_section}\n\n{question_section}"    
    # 3. Return the formatted string
    return prompt



def generate_answer(query: str, chunks: list[dict]) -> str:
    """Send query + chunks to Gemini and return the generated answer."""
    # 1. Build the prompt
    prompt = build_prompt(query, chunks)
    # 2. Initialize the genai client
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # 3. Call client.models.generate_content with system instruction and low temperature
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
        ),
    )
    # 4. Return the response text
    return response.text


if __name__ == "__main__":
    index, metadata, model = load_retriever()
    query = "What is the minimum width of a corridor for wheelchair access?"

    #Checking if refusdal works for out-of-scope question
    # query = "What is the capital of France?"
    # query = "What's the maximum allowable noise level in office spaces?"

    print(f"Question: {query}\n")

    chunks = retrieve(query, index, metadata, model, top_k=5)
    print(f"Retrieved {len(chunks)} chunks. Top scores: {[round(c['score'], 3) for c in chunks]}\n")

    answer = generate_answer(query, chunks)
    print("Answer:")
    print(answer)