import os
from typing import List, Dict, Any, Tuple
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.retrievers import BM25Retriever
from src.utils import logger

class HybridRetriever:
    """
    Implements Hybrid Retrieval by combining Semantic Vector Search
    (from FAISS or ChromaDB) and Keyword Search (BM25) using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, vectorstore, top_k: int = 4):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.bm25_retriever = None
        self._all_documents = []
        self._initialize_bm25()

    def _initialize_bm25(self):
        """
        Extracts all documents from the vector store and builds the BM25 index.
        """
        if self.vectorstore is None:
            return

        try:
            logger.info("Extracting documents from vector store to build BM25 index.")
            if isinstance(self.vectorstore, FAISS):
                self._all_documents = list(self.vectorstore.docstore._dict.values())
            elif isinstance(self.vectorstore, Chroma):
                result = self.vectorstore.get(include=["documents", "metadatas"])
                self._all_documents = []
                for text, metadata in zip(result.get("documents", []), result.get("metadatas", [])):
                    self._all_documents.append(Document(page_content=text, metadata=metadata))
            
            if self._all_documents:
                logger.info(f"Building BM25Retriever with {len(self._all_documents)} document chunks.")
                self.bm25_retriever = BM25Retriever.from_documents(self._all_documents)
                self.bm25_retriever.k = self.top_k
            else:
                logger.warning("No documents found in vector store to build BM25.")
        except Exception as e:
            logger.error(f"Failed to initialize BM25 retriever: {str(e)}")

    def _normalize_score(self, score: float, db_type: str) -> float:
        """
        Normalizes vector store distances/scores into a 0.0 - 1.0 similarity score range.
        """
        try:
            score = float(score)
            if db_type == "FAISS":
                similarity = 1.0 - (score / 2.0)
                return max(0.0, min(1.0, similarity))
            elif db_type == "CHROMADB":
                similarity = 1.0 / (1.0 + score)
                return max(0.0, min(1.0, similarity))
            return 0.5
        except Exception as e:
            logger.error(f"Error normalizing score: {str(e)}")
            return 0.5

    def retrieve(self, query: str, search_type: str = "Hybrid") -> List[Dict[str, Any]]:
        """
        Retrieves top-k documents using Semantic, Keyword, or Hybrid search.
        """
        if self.vectorstore is None:
            logger.warning("Retrieve called but vectorstore is None.")
            return []

        search_type = search_type.capitalize()
        logger.info(f"Retrieving top {self.top_k} documents for query='{query}' using {search_type} search.")

        # 1. Semantic Search
        semantic_results: List[Tuple[Document, float]] = []
        try:
            db_name = "FAISS" if isinstance(self.vectorstore, FAISS) else "CHROMADB"
            semantic_results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        except Exception as e:
            logger.error(f"Error during semantic vector search: {str(e)}")

        # 2. Keyword Search (BM25)
        keyword_results: List[Document] = []
        if self.bm25_retriever is not None and search_type in ["Keyword", "Hybrid"]:
            try:
                keyword_results = self.bm25_retriever.get_relevant_documents(query)
            except Exception as e:
                logger.error(f"Error during BM25 keyword search: {str(e)}")

        if search_type == "Semantic":
            formatted_results = []
            for doc, raw_score in semantic_results:
                norm_score = self._normalize_score(raw_score, db_name)
                formatted_results.append({
                    "document": doc,
                    "score": norm_score,
                    "type": "Semantic",
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 1)
                })
            return formatted_results[:self.top_k]

        elif search_type == "Keyword":
            formatted_results = []
            for doc in keyword_results:
                formatted_results.append({
                    "document": doc,
                    "score": 1.0,
                    "type": "Keyword",
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 1)
                })
            return formatted_results[:self.top_k]

        else:  # Hybrid search (RRF)
            RRF_K = 60
            rrf_scores: Dict[str, float] = {}
            doc_lookup: Dict[str, Document] = {}
            doc_semantic_score: Dict[str, float] = {}

            # Process Semantic Ranks
            for rank, (doc, raw_score) in enumerate(semantic_results):
                doc_id = doc.page_content
                doc_lookup[doc_id] = doc
                norm_score = self._normalize_score(raw_score, db_name)
                doc_semantic_score[doc_id] = norm_score
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + (rank + 1)))

            # Process Keyword Ranks
            for rank, doc in enumerate(keyword_results):
                doc_id = doc.page_content
                doc_lookup[doc_id] = doc
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + (rank + 1)))

            # Sort docs by RRF score descending
            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            
            formatted_results = []
            for doc_id, rrf_score in sorted_docs[:self.top_k]:
                doc = doc_lookup[doc_id]
                score = doc_semantic_score.get(doc_id, 0.70)
                formatted_results.append({
                    "document": doc,
                    "score": score,
                    "type": "Hybrid",
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 1)
                })
                
            return formatted_results
