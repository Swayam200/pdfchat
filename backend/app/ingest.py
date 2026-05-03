# backend/app/ingest.py

import sys
from pathlib import Path

from app.pdf_loader import load_pdf
from app.chunker import chunk_documents
from app.embeddings import store_embeddings

def ingest_pdf(pdf_path: str):
    """
    Full ingestion pipeline: load PDF → chunk → embed → store.
    """
    print(f"[1/3] Loading PDF: {pdf_path}")
    documents = load_pdf(pdf_path)
    print(f"      Loaded {len(documents)} pages")
    
    print(f"[2/3] Chunking documents...")
    chunked_docs = chunk_documents(documents)
    print(f"      Created {len(chunked_docs)} chunks")
    
    print(f"[3/3] Embedding and storing in ChromaDB...")
    vector_store = store_embeddings(chunked_docs)
    print(f"      Stored in ./chroma_store")
    
    # Test: query to prove it worked
    print(f"\n[TEST] Retrieving top 2 similar chunks to query 'what is this about?'")
    results = vector_store.similarity_search("what is this about?", k=2)
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content[:100]}...")
        print(f"      (source: {doc.metadata})\n")

if __name__ == "__main__":
    # Example: python -m app.ingest sample.pdf
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingest <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    ingest_pdf(pdf_path)