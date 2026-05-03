from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """
    Splits documents into overlapping chunks.

    Args:
        documents (List[Document]): The list of documents to be chunked.
        chunk_size (int): The maximum size of each chunk. Default is 1000 characters.
        chunk_overlap (int): The number of characters to overlap between chunks. Default is 200 characters.

    Returns:
        List[Document]: A list of chunked documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""] # Trying to split on paragraphs first, then lines, then spaces, and finally characters
        )
    
    chunked_docs = splitter.split_documents(documents)
    return chunked_docs