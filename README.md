# compliance-rag-assistant
Compliance RAG Assistant

Compliance teams spend hours searching long regulatory documents for answers. General-purpose chatbots can’t be trusted for this, because they may invent rules that don’t exist.

This project is an assistant that answers questions about UK compliance documents, such as ICO guidance on UK GDPR and FCA Consumer Duty guidance, using only the content of those documents. Every answer cites the exact passage it came from, so users can check it. If the answer isn’t in the documents, the assistant says so instead of guessing.

How it works

Regulatory PDFs are split into chunks and turned into embeddings.
For each question, the most relevant chunks are retrieved from a vector database.
A small open-source language model writes an answer from those chunks only, with citations.
A test set of questions measures answer accuracy, citation accuracy and how often the model invents information, so prompt changes are judged on evidence.

Tech stack: Python, Hugging Face Transformers (Qwen2.5), sentence-transformers, ChromaDB, FastAPI, Streamlit, GitHub Actions.

Status: In progress.
