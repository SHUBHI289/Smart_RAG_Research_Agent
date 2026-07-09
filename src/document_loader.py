import os
import requests
from typing import List
from urllib.parse import urlparse
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from src.utils import logger

class DocumentLoader:
    """
    Handles routing and loading for various file types (PDF, TXT) and Web URLs.
    Transforms raw files into standard LangChain Document chunks.
    """

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """
        Loads PDF documents using PyPDFLoader.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")
            
        logger.info(f"Loading PDF file: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Enrich metadata
            file_name = os.path.basename(file_path)
            for doc in docs:
                doc.metadata["source"] = file_name
                doc.metadata["file_type"] = "pdf"
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"] + 1
                else:
                    doc.metadata["page"] = 1
                    
            logger.info(f"Successfully loaded {len(docs)} pages from PDF: {file_name}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load PDF file {file_path}: {str(e)}")
            raise ValueError(f"Invalid or corrupted PDF file: {str(e)}")

    @staticmethod
    def load_txt(file_path: str) -> List[Document]:
        """
        Loads plain text files using TextLoader.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TXT file not found at: {file_path}")
            
        logger.info(f"Loading TXT file: {file_path}")
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            
            file_name = os.path.basename(file_path)
            for doc in docs:
                doc.metadata["source"] = file_name
                doc.metadata["file_type"] = "txt"
                doc.metadata["page"] = 1
                
            logger.info(f"Successfully loaded TXT file: {file_name}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load TXT file {file_path}: {str(e)}")
            raise ValueError(f"Error reading text file: {str(e)}")

    @staticmethod
    def load_url(url: str) -> List[Document]:
        """
        Scrapes and loads web pages using WebBaseLoader.
        """
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL string: '{url}'")

        logger.info(f"Loading content from URL: {url}")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            
            response = requests.head(url, headers=headers, timeout=10)
            if response.status_code >= 400:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

            loader = WebBaseLoader(url)
            loader.requests_kwargs = {"headers": headers, "timeout": 15}
            docs = loader.load()
            
            for doc in docs:
                doc.metadata["source"] = url
                doc.metadata["file_type"] = "web"
                doc.metadata["page"] = 1
                
            logger.info(f"Successfully loaded website content from {url}")
            return docs
        except requests.exceptions.RequestException as re:
            logger.error(f"Network error accessing URL {url}: {str(re)}")
            raise ValueError(f"Network error accessing web page: {str(re)}")
        except Exception as e:
            logger.error(f"Failed to parse Web URL {url}: {str(e)}")
            raise ValueError(f"Failed to load website content: {str(e)}")

    def load_source(self, path_or_url: str) -> List[Document]:
        """
        Identifies the source format and loads using the correct strategy.
        """
        if path_or_url.startswith(("http://", "https://")):
            return self.load_url(path_or_url)
        elif path_or_url.lower().endswith(".pdf"):
            return self.load_pdf(path_or_url)
        elif path_or_url.lower().endswith(".txt"):
            return self.load_txt(path_or_url)
        else:
            raise ValueError(f"Unsupported file type or protocol: {path_or_url}")
