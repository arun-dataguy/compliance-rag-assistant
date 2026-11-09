from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

RAW_DIR = Path("data/raw")
DB_DIR = "data/chroma"
COLLECTION = "compliance_docs"
CHUNK_WORDS = 300    # words per chunk
OVERLAP_WORDS = 50   # words shared between neighbouring chunks


def load_pdfs():
    """Read every PDF and return its text page by page, keeping the source and page number."""
    pages = []
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = " ".join((page.extract_text() or "").split())  # tidy whitespace
            if text:
                pages.append({"source": pdf_path.name, "page": page_num, "text": text})
    return pages


def chunk_pages(pages):
    """Split each page into overlapping chunks of words."""
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for p in pages:
        words = p["text"].split()
        for start in range(0, len(words), step):
            piece = words[start:start + CHUNK_WORDS]
            if start > 0 and len(piece) < 30:  # skip tiny leftover pieces
                break
            chunks.append({
                "id": f"{p['source']}-p{p['page']}-w{start}",
                "text": " ".join(piece),
                "source": p["source"],
                "page": p["page"],
            })
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


def main():
    pages = load_pdfs()
    chunks = chunk_pages(pages)
    pdf_count = len({p["source"] for p in pages})
    print(f"Read {pdf_count} PDFs, {len(pages)} pages, made {len(chunks)} chunks")

    print("Creating embeddings (first run downloads a small model)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(
        [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)  # start fresh on each run
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    batch = 1000
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        collection.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=embeddings[i:i + batch].tolist(),
            metadatas=[{"source": c["source"], "page": c["page"]} for c in part],
        )
    print(f"Stored {collection.count()} chunks in {DB_DIR}")

    # Quick check: search for a sample question
    question = "What are the lawful bases for processing personal data?"
    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    print(f"\nTop matches for: {question}")
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        print(f"  {meta['source']} (page {meta['page']}), similarity {1 - dist:.2f}")


if __name__ == "__main__":
    main()