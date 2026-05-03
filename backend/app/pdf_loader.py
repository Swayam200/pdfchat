from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return a list of Document objects.

    Args:
        file_path (str): The path to the PDF file.
    
    Returns:
        List[Document]: A list of Document objects containing the text and metadata from the PDF.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
