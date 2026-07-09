import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils import logger




class EmbeddingManager:
    """
    Manages embedding model initialization.
    Supports sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, and BAAI/bge-base-en-v1.5.
    """
    
    SUPPORTED_MODELS = {
        "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5"
    }

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model '{model_name}' is not in supported list. "
                f"Defaulting to 'sentence-transformers/all-MiniLM-L6-v2'."
            )
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embeddings = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Initializes and returns the HuggingFaceEmbeddings instance.
        """
        if self._embeddings is not None:
            return self._embeddings

        logger.info(f"Initializing embedding model: {self.model_name} on device: {self.device}")
        try:
            model_kwargs = {"device": self.device}
            encode_kwargs = {"normalize_embeddings": True}
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            logger.info("Embedding model initialized successfully.")
            return self._embeddings
        except Exception as e:
            logger.error(f"Error initializing embedding model {self.model_name}: {str(e)}")
            raise e
