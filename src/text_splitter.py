from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from src.utils import logger

class DocumentSplitter:
    """
    Splits text documents into smaller chunks using RecursiveCharacterTextSplitter.
    Supports configuring chunk size and overlap at runtime.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits lists of documents into smaller, overlapping chunks.
        """
        if not documents:
            logger.warning("Empty document list passed for splitting.")
            return []

        logger.info(
            f"Splitting {len(documents)} documents (chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )

        try:
            chunks = self.splitter.split_documents(documents)
            logger.info(f"Generated {len(chunks)} text chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise e
