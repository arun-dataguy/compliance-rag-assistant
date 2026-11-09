import csv
import re
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from ask import COLLECTION, DB_DIR, LLM_MODEL, build_messages, retrieve

QUESTIONS_FILE = Path("eval/questions.csv")
RESULTS_FILE = Path("eval/results.csv")
REFUSAL = "can't find this in the documents"


def score(row, answer, passages):
    """Score one answer against the expected result."""
    answer_lower = answer.lower()
    refused = REFUSAL in answer_lower.replace("cannot", "can't")
    cited_numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    cited_sources = {passages[n - 1]["source"] for n in cited_numbers if 1 <= n <= len(passages)}
    retrieved_sources = {p["source"] for p in passages}

    if row["answerable"].strip().lower() == "no":
        return {
            "correct": refused,
            "retrieval_hit": "",
            "cited": "",
            "cited_correct_source": "",
            "refused": refused,
        }

    keywords = [k.strip().lower() for k in row["expected_keywords"].split("|") if k.strip()]
    keywords_found = all(k in answer_lower for k in keywords)
    return {
        "correct": keywords_found and not refused,
        "retrieval_hit": row["expected_source"] in retrieved_sources,
        "cited": bool(cited_numbers),
        "cited_correct_source": row["expected_source"] in cited_sources,
        "refused": refused,
    }


def pct(values):
    values = [v for v in values if v != ""]
    return f"{100 * sum(values) / len(values):.0f}% ({sum(values)}/{len(values)})" if values else "n/a"


def main():
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("Loading models...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    generator = pipeline("text-generation", model=LLM_MODEL)

    results = []
    start = time.time()
    for i, row in enumerate(rows, start=1):
        passages = retrieve(row["question"], embedder, collection)
        output = generator(build_messages(row["question"], passages), max_new_tokens=300, do_sample=False)
        answer = output[0]["generated_text"][-1]["content"]
        s = score(row, answer, passages)
        results.append({**row, "answer": answer, **s})
        print(f"[{i}/{len(rows)}] {'PASS' if s['correct'] else 'FAIL'}  {row['question']}")

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    yes = [r for r in results if r["answerable"].strip().lower() == "yes"]
    no = [r for r in results if r["answerable"].strip().lower() == "no"]
    print(f"\n===== Results ({len(results)} questions, {time.time() - start:.0f}s) =====")
    print(f"Answer accuracy (answerable):        {pct([r['correct'] for r in yes])}")
    print(f"Retrieval hit rate:                  {pct([r['retrieval_hit'] for r in yes])}")
    print(f"Answers with citations:              {pct([r['cited'] for r in yes])}")
    print(f"Cited the correct document:          {pct([r['cited_correct_source'] for r in yes])}")
    print(f"Wrongly refused (answerable):        {pct([r['refused'] for r in yes])}")
    print(f"Correctly refused (unanswerable):    {pct([r['correct'] for r in no])}")
    print(f"\nFull answers saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()