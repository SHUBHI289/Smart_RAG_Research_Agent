import os
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from src.utils import logger

class RagasEvaluator:
    """
    Handles evaluation of RAG responses using RAGAS.
    """

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.ragas_llm = None
        self.ragas_embeddings = None
        self._initialize_wrappers()

    def _initialize_wrappers(self):
        """
        Wraps LangChain LLM and Embeddings for compatibility with RAGAS.
        """
        try:
            logger.info("Initializing RAGAS wrappers for LLM and Embeddings.")
            self.ragas_llm = LangchainLLMWrapper(langchain_llm=self.llm)
            self.ragas_embeddings = LangchainEmbeddingsWrapper(embeddings=self.embeddings)
            
            # Attach model and embeddings to RAGAS metrics
            faithfulness.llm = self.ragas_llm
            answer_relevancy.llm = self.ragas_llm
            answer_relevancy.embeddings = self.ragas_embeddings
            logger.info("RAGAS wrappers initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to wrap LangChain components for RAGAS: {str(e)}")

    def evaluate_response(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        Evaluates a single question, answer, and retrieved contexts tuple.
        Returns a dictionary of scores.
        """
        if not question or not answer or not contexts:
            logger.warning("Empty parameters passed to evaluate_response. Skipping evaluation.")
            return {"faithfulness": 0.0, "answer_relevancy": 0.0}

        if self.ragas_llm is None or self.ragas_embeddings is None:
            logger.warning("RAGAS wrappers were not initialized. Skipping evaluation.")
            return {"faithfulness": 0.0, "answer_relevancy": 0.0}

        logger.info(f"Running RAGAS evaluation for: question='{question[:30]}...'")
        
        try:
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts]
            }
            dataset = Dataset.from_dict(data)

            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
                raise_exceptions=False
            )
            
            scores = {
                "faithfulness": float(result.get("faithfulness", 0.0)),
                "answer_relevancy": float(result.get("answer_relevancy", 0.0))
            }
            
            logger.info(f"RAGAS evaluation scores: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {str(e)}")
            return {"faithfulness": 0.0, "answer_relevancy": 0.0}
