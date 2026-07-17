import os
from typing import Optional
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.utils import logger


class EmbeddingManager:
    """
    Manages embedding model initialization.
    Supports Google GenAI gemini-embedding-2 and HuggingFace models.
    """
    
    SUPPORTED_MODELS = {
        "google-genai/gemini-embedding-2": "models/gemini-embedding-2",
        "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5"
    }

    def __init__(self, model_name: str = "google-genai/gemini-embedding-2", api_key: Optional[str] = None):
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model '{model_name}' is not in supported list. "
                f"Defaulting to 'google-genai/gemini-embedding-2'."
            )
            model_name = "google-genai/gemini-embedding-2"
            
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embeddings = None

    def get_embeddings(self):
        """
        Initializes and returns the Embeddings instance.
        """
        if self._embeddings is not None:
            return self._embeddings

        logger.info(f"Initializing embedding model: {self.model_name}")
        
        if self.model_name == "google-genai/gemini-embedding-2":
            try:
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-2",
                    google_api_key=self.api_key,
                    request_options={"timeout": 300.0}
                )
                logger.info("Google GenAI Embeddings initialized successfully.")
                return self._embeddings
            except Exception as e:
                logger.error(f"Error initializing Google GenAI Embeddings: {str(e)}")
                raise e
        else:
            try:
                model_kwargs = {"device": self.device}
                encode_kwargs = {"normalize_embeddings": True}
                
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                logger.info("HuggingFace Embedding model initialized successfully.")
                return self._embeddings
            except Exception as e:
                logger.error(f"Error initializing HuggingFace embedding model {self.model_name}: {str(e)}")
                raise e
