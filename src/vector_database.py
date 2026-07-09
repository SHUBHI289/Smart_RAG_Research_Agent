import os
import shutil
from typing import List, Optional, Union
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document
from src.utils import logger

# Base directory for vector storage
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_ROOT_DIR = os.path.join(BASE_DIR, "vector_store")

class VectorStoreManager:
    """
    Manages vector databases (FAISS and ChromaDB).
    Partitions indices based on both DB type and embedding model name to avoid conflicts.
    """
    
    def __init__(self, db_type: str = "FAISS", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.db_type = db_type.upper()
        if self.db_type not in ["FAISS", "CHROMADB"]:
            logger.warning(f"Unsupported db_type '{db_type}'. Defaulting to FAISS.")
            self.db_type = "FAISS"

        # Sanitize model name for directory paths
        sanitized_model_name = embedding_model_name.replace("/", "_").replace("\\", "_")
        
        self.persist_dir = os.path.join(DB_ROOT_DIR, self.db_type.lower(), sanitized_model_name)
        os.makedirs(self.persist_dir, exist_ok=True)
        
        logger.info(f"VectorStoreManager initialized for {self.db_type} at {self.persist_dir}")

    def create_or_update_vectorstore(self, documents: List[Document], embeddings: Embeddings) -> Union[FAISS, Chroma]:
        """
        Creates a new vector store or updates the existing one.
        """
        if not documents:
            logger.warning("No documents supplied to create or update vectorstore.")
            raise ValueError("Document list cannot be empty.")

        logger.info(f"Creating/updating {self.db_type} index with {len(documents)} document chunks.")
        try:
            if self.db_type == "FAISS":
                faiss_index_file = os.path.join(self.persist_dir, "index.faiss")
                if os.path.exists(faiss_index_file):
                    logger.info("Loading existing FAISS index to append new documents.")
                    db = FAISS.load_local(
                        self.persist_dir, 
                        embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    db.add_documents(documents)
                else:
                    logger.info("Building a new FAISS index.")
                    db = FAISS.from_documents(documents, embeddings)
                
                db.save_local(self.persist_dir)
                logger.info(f"FAISS index successfully saved to {self.persist_dir}")
                return db
                
            elif self.db_type == "CHROMADB":
                logger.info("Initializing ChromaDB.")
                db = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=embeddings
                )
                db.add_documents(documents)
                if hasattr(db, "persist"):
                    db.persist()
                logger.info(f"ChromaDB index successfully saved/updated at {self.persist_dir}")
                return db
                
        except Exception as e:
            logger.error(f"Error while building/updating {self.db_type} database: {str(e)}")
            raise e

    def get_vectorstore(self, embeddings: Embeddings) -> Optional[Union[FAISS, Chroma]]:
        """
        Loads the existing vector store index from disk. Returns None if it doesn't exist.
        """
        try:
            if self.db_type == "FAISS":
                faiss_index_file = os.path.join(self.persist_dir, "index.faiss")
                if not os.path.exists(faiss_index_file):
                    logger.warning(f"No FAISS index found at {self.persist_dir}")
                    return None
                db = FAISS.load_local(
                    self.persist_dir, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                return db
                
            elif self.db_type == "CHROMADB":
                if not os.path.exists(self.persist_dir) or not os.listdir(self.persist_dir):
                    logger.warning(f"No ChromaDB index files found at {self.persist_dir}")
                    return None
                db = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=embeddings
                )
                return db
        except Exception as e:
            logger.error(f"Error loading {self.db_type} vector database: {str(e)}")
            return None

    def clear_vectorstore(self):
        """
        Removes the index files from disk.
        """
        logger.info(f"Clearing vector store directory: {self.persist_dir}")
        try:
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir)
                os.makedirs(self.persist_dir, exist_ok=True)
                logger.info(f"Successfully cleared vector store at {self.persist_dir}")
            else:
                logger.warning(f"Directory {self.persist_dir} did not exist.")
        except Exception as e:
            logger.error(f"Error deleting vector store files: {str(e)}")
            raise e
            
    @staticmethod
    def clear_all_stores():
        """
        Deletes all vector stores under DB_ROOT_DIR.
        """
        logger.info(f"Clearing all vector stores under {DB_ROOT_DIR}")
        try:
            if os.path.exists(DB_ROOT_DIR):
                shutil.rmtree(DB_ROOT_DIR)
                os.makedirs(DB_ROOT_DIR, exist_ok=True)
                logger.info("Successfully cleared all vector stores.")
        except Exception as e:
            logger.error(f"Failed to clear all vector stores: {str(e)}")
            raise e
