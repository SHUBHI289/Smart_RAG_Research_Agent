import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils import logger

class GeminiLLMManager:
    """
    Manages connection to the Google Gemini LLM via LangChain.
    Attempts to use 'gemini-2.5-flash' first, and falls back to 'gemini-1.5-flash'
    if the API key or system doesn't support the 2.5 version.
    """

    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.2):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.temperature = temperature
        self.primary_model = "gemini-2.5-flash"
        self.fallback_model = "gemini-1.5-flash"
        self._llm = None

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        Initializes and returns the ChatGoogleGenerativeAI instance.
        """
        if self._llm is not None:
            return self._llm

        if not self.api_key:
            logger.error("GOOGLE_API_KEY environment variable or argument is missing.")
            raise ValueError(
                "Google Gemini API Key is missing. "
                "Please configure it in the .env file or enter it in the sidebar."
            )

        logger.info(f"Initializing Gemini LLM wrapper (Primary: {self.primary_model})")
        
        try:
            self._llm = ChatGoogleGenerativeAI(
                model=self.primary_model,
                google_api_key=self.api_key,
                temperature=self.temperature,
                convert_system_message_to_human=True,
                max_retries=2
            )
            logger.info(f"Initialized ChatGoogleGenerativeAI with model={self.primary_model}")
            return self._llm
        except Exception as e:
            logger.warning(
                f"Failed to initialize primary model '{self.primary_model}' due to: {str(e)}. "
                f"Falling back to '{self.fallback_model}'."
            )
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=self.fallback_model,
                    google_api_key=self.api_key,
                    temperature=self.temperature,
                    convert_system_message_to_human=True,
                    max_retries=2
                )
                logger.info(f"Successfully fallback initialized model={self.fallback_model}")
                return self._llm
            except Exception as fe:
                logger.error(f"Failed fallback initialization of model '{self.fallback_model}': {str(fe)}")
                raise fe

    def check_api_key_validity(self) -> bool:
        """
        Runs a minor test query to verify if the API key is working.
        """
        if not self.api_key:
            return False
        try:
            llm = self.get_llm()
            llm.invoke("Hi")
            return True
        except Exception as e:
            logger.error(f"Gemini API test connection failed: {str(e)}")
            self._llm = None
            return False
