import sys

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

DB_DIR = "data/chroma"
COLLECTION = "compliance_docs"
TOP_K = 4  # number of passages to give the model
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = """You are a UK data protection compliance assistant.
Answer the question using ONLY the numbered sources provided.
After each sentence, cite the source number in square brackets, for example [1].
If the sources do not contain the answer, reply exactly:
"I can't find this in the documents provided."
Do not use any outside knowledge."""


def retrieve(question, embedder, collection, k=TOP_K):
    """Find the k passages most similar in meaning to the question."""
    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
    r = collection.query(query_embeddings=q_emb, n_results=k)
    return [
        {"text": doc, "source": meta["source"], "page": meta["page"], "similarity": 1 - dist}
        for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
    ]


def build_messages(question, passages):
    """Number the passages and put them in front of the question."""
    sources = "\n\n".join(
        f"[{i}] ({p['source']}, page {p['page']})\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
    ]


def main():
    question = " ".join(sys.argv[1:]) or "How long does an organisation have to respond to a subject access request?"

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    generator = pipeline("text-generation", model=LLM_MODEL)

    passages = retrieve(question, embedder, collection)
    messages = build_messages(question, passages)
    output = generator(messages, max_new_tokens=300, do_sample=False)
    answer = output[0]["generated_text"][-1]["content"]

    print(f"\nQuestion: {question}\n")
    print(f"Answer:\n{answer}\n")
    print("Sources:")
    for i, p in enumerate(passages, start=1):
        print(f"  [{i}] {p['source']}, page {p['page']} (similarity {p['similarity']:.2f})")


if __name__ == "__main__":
    main()