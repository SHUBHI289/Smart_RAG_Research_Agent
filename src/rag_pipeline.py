import os
from typing import List, Dict, Any, Generator, Tuple, Optional
from langchain.docstore.document import Document
from src.utils import logger, count_tokens
from src.text_splitter import DocumentSplitter
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingManager
from src.vector_database import VectorStoreManager
from src.retriever import HybridRetriever
from src.llm import GeminiLLMManager
from src.prompts import PROMPT_QA, PROMPT_CONVERSATIONAL
from src.memory import ConversationMemoryManager
from src.evaluator import RagasEvaluator

class RAGPipeline:
    """
    Orchestrates the entire RAG pipeline:
    - Document Ingestion (loading, splitting, embedding, vector store creation)
    - Querying (retrieval, prompt construction, memory handling, LLM generation, streaming)
    - Evaluation (RAGAS faithfulness & answer relevancy metrics)
    - Metadata & Statistics tracking
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.memory_manager = ConversationMemoryManager(window_size=5)
        self.loader = DocumentLoader()

    def ingest_sources(
        self, 
        paths_or_urls: List[str], 
        db_type: str = "FAISS", 
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Loads documents, splits into chunks, computes embeddings, and stores in the chosen vector database.
        Returns document statistics.
        """
        logger.info(f"Starting ingestion of {len(paths_or_urls)} sources using {db_type} and {embedding_model}")
        
        all_docs = []
        errors = []
        
        for source in paths_or_urls:
            try:
                docs = self.loader.load_source(source)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load source '{source}': {str(e)}")
                errors.append(f"Failed to load {source}: {str(e)}")

        if not all_docs:
            error_msg = "No documents could be loaded. " + "; ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Split documents
        splitter = DocumentSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(all_docs)
        
        # Initialize Embeddings
        embedding_manager = EmbeddingManager(model_name=embedding_model)
        embeddings = embedding_manager.get_embeddings()
        
        # Save to Vector Database
        vdb_manager = VectorStoreManager(db_type=db_type, embedding_model_name=embedding_model)
        vdb_manager.create_or_update_vectorstore(chunks, embeddings)
        
        # Compute stats
        total_pages = sum(1 for doc in all_docs if doc.metadata.get("file_type") == "pdf") or len(all_docs)
        total_chunks = len(chunks)
        total_tokens = sum(count_tokens(chunk.page_content) for chunk in chunks)
        
        logger.info("Ingestion completed successfully.")
        return {
            "total_sources": len(paths_or_urls),
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "errors": errors
        }

    def clear_database(self, db_type: str, embedding_model: str):
        """
        Clears the specific vector database index on disk.
        """
        vdb_manager = VectorStoreManager(db_type=db_type, embedding_model_name=embedding_model)
        vdb_manager.clear_vectorstore()
        logger.info(f"Database {db_type} for model {embedding_model} cleared.")

    def query(
        self,
        question: str,
        db_type: str = "FAISS",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        search_type: str = "Hybrid",
        top_k: int = 4,
        temperature: float = 0.2,
        use_history: bool = True
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, float]]:
        """
        Executes a synchronous query on the RAG pipeline.
        Returns a tuple: (answer, retrieved_sources, evaluation_scores)
        """
        # Load Vector Store
        embedding_manager = EmbeddingManager(model_name=embedding_model)
        embeddings = embedding_manager.get_embeddings()
        
        vdb_manager = VectorStoreManager(db_type=db_type, embedding_model_name=embedding_model)
        vectorstore = vdb_manager.get_vectorstore(embeddings)
        
        if vectorstore is None:
            raise ValueError(
                f"The {db_type} vector database for embedding model '{embedding_model}' has not been built yet. "
                f"Please upload documents and click 'Build Knowledge Base' first."
            )
            
        # Retrieve Sources
        retriever = HybridRetriever(vectorstore=vectorstore, top_k=top_k)
        retrieved_sources = retriever.retrieve(question, search_type=search_type)
        
        if not retrieved_sources:
            return "I couldn't find this information in the uploaded documents.", [], {"faithfulness": 0.0, "answer_relevancy": 0.0}

        # Build Context String
        context_str = "\n\n".join(
            f"[Source: {s['source']} | Page: {s['page']} | Similarity: {s['score']:.2f}]\n{s['document'].page_content}"
            for s in retrieved_sources
        )
        
        # Load Gemini LLM
        llm_manager = GeminiLLMManager(api_key=self.api_key, temperature=temperature)
        llm = llm_manager.get_llm()
        
        # Choose Prompt and Arguments
        if use_history:
            history_str = self.memory_manager.get_history_string()
            prompt_input = PROMPT_CONVERSATIONAL.format(
                context=context_str,
                chat_history=history_str,
                question=question
            )
        else:
            prompt_input = PROMPT_QA.format(
                context=context_str,
                question=question
            )
            
        # Run LLM
        logger.info("Invoking LLM for response generation.")
        try:
            response = llm.invoke(prompt_input)
            answer = response.content
        except Exception as e:
            logger.error(f"Gemini LLM invocation failed: {str(e)}")
            raise RuntimeError(f"Error during response generation: {str(e)}")
            
        # Save Interaction to Memory
        if use_history:
            self.memory_manager.add_interaction(question, answer)
            
        # Run Evaluation (RAGAS)
        eval_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0}
        try:
            evaluator = RagasEvaluator(llm=llm, embeddings=embeddings)
            contexts = [s["document"].page_content for s in retrieved_sources]
            eval_scores = evaluator.evaluate_response(question, answer, contexts)
        except Exception as e:
            logger.error(f"RAGAS evaluation execution failed: {str(e)}")
            
        return answer, retrieved_sources, eval_scores

    def query_stream(
        self,
        question: str,
        db_type: str = "FAISS",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        search_type: str = "Hybrid",
        top_k: int = 4,
        temperature: float = 0.2,
        use_history: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams response chunks from Gemini LLM.
        """
        # Load Vector Store
        embedding_manager = EmbeddingManager(model_name=embedding_model)
        embeddings = embedding_manager.get_embeddings()
        
        vdb_manager = VectorStoreManager(db_type=db_type, embedding_model_name=embedding_model)
        vectorstore = vdb_manager.get_vectorstore(embeddings)
        
        if vectorstore is None:
            raise ValueError(
                f"The {db_type} vector database for embedding model '{embedding_model}' has not been built yet. "
                f"Please upload documents and click 'Build Knowledge Base' first."
            )
            
        # Retrieve Sources
        retriever = HybridRetriever(vectorstore=vectorstore, top_k=top_k)
        retrieved_sources = retriever.retrieve(question, search_type=search_type)
        
        yield {"type": "sources", "data": retrieved_sources}
        
        if not retrieved_sources:
            yield {"type": "chunk", "text": "I couldn't find this information in the uploaded documents."}
            yield {"type": "evaluation", "scores": {"faithfulness": 0.0, "answer_relevancy": 0.0}}
            return

        # Build Context String
        context_str = "\n\n".join(
            f"[Source: {s['source']} | Page: {s['page']} | Similarity: {s['score']:.2f}]\n{s['document'].page_content}"
            for s in retrieved_sources
        )
        
        # Load Gemini LLM
        llm_manager = GeminiLLMManager(api_key=self.api_key, temperature=temperature)
        llm = llm_manager.get_llm()
        
        # Choose Prompt and Arguments
        if use_history:
            history_str = self.memory_manager.get_history_string()
            prompt_input = PROMPT_CONVERSATIONAL.format(
                context=context_str,
                chat_history=history_str,
                question=question
            )
        else:
            prompt_input = PROMPT_QA.format(
                context=context_str,
                question=question
            )
            
        # Stream response
        logger.info("Streaming response from Gemini LLM.")
        full_answer_list = []
        try:
            for chunk in llm.stream(prompt_input):
                chunk_text = chunk.content
                full_answer_list.append(chunk_text)
                yield {"type": "chunk", "text": chunk_text}
        except Exception as e:
            logger.error(f"Gemini LLM streaming failed: {str(e)}")
            raise RuntimeError(f"Error during response streaming: {str(e)}")
            
        full_answer = "".join(full_answer_list)
        
        # Save Interaction to Memory
        if use_history:
            self.memory_manager.add_interaction(question, full_answer)
            
        # Run Evaluation (RAGAS)
        eval_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0}
        try:
            evaluator = RagasEvaluator(llm=llm, embeddings=embeddings)
            contexts = [s["document"].page_content for s in retrieved_sources]
            eval_scores = evaluator.evaluate_response(question, full_answer, contexts)
        except Exception as e:
            logger.error(f"RAGAS evaluation failed in streaming pipeline: {str(e)}")
            
        yield {"type": "evaluation", "scores": eval_scores}
