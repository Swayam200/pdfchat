# backend/app/embeddings.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from typing import List
import os

def get_embeddings_model():
    """
    Initialize the embedding model (runs locally, no API key needed).
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def store_embeddings(documents: List[Document], persist_directory: str = "./chroma_store"):
    """
    Convert documents to embeddings and store in ChromaDB.
    
    Args:
        documents: List of chunked Document objects
        persist_directory: Where to save the vector database on disk
        
    Returns:
        The Chroma vector store object
    """
    embeddings_model = get_embeddings_model()
    
    # Create or load ChromaDB with persistence
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        persist_directory=persist_directory
    )
    
    # Critical: persist to disk
    vector_store.persist()
    
    return vector_store