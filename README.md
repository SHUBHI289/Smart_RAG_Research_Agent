# Smart Research Assistant (RAG-Based Knowledge System)

The **Smart Research Assistant** is a production-ready Retrieval-Augmented Generation (RAG) application that allows users to upload document collections (PDF, TXT) and web page URLs to build a dynamic knowledge base. Users can ask natural language questions and receive accurate, context-grounded responses powered by Google's Gemini 2.5 Flash, complete with highlighted source citations, similarity scores, and automated RAGAS evaluation.

---

## 📌 Features
- **Multi-Source Loading**: Seamless parsing of PDFs (using `PyPDFLoader`), Text files (`TextLoader`), and Web URLs (`WebBaseLoader`).
- **Flexible Chunking**: Fully configurable chunk sizing and overlap settings.
- **Dynamic Embedding Models**: Choice of HuggingFace embeddings (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `bge-base-en-v1.5`) running on CPU/GPU.
- **Hybrid Retrieval System**: Combines Reciprocal Rank Fusion (RRF) on BM25 keyword search and semantic vector similarity search.
- **Dual Vector Stores**: Supports switching dynamically between **FAISS** and **ChromaDB**.
- **Context Grounding & Prompt Enforcement**: Restricts hallucinations; yields "I couldn't find this information in the uploaded documents" if context doesn't contain answers.
- **Multi-turn Chat Memory**: Remembers conversational context for up to 5 turns.
- **Conversational Analytics**: Displays page numbers, similarity scores, highlighted contexts, and knowledge base stats.
- **Automatic RAGAS Evaluation**: Measures Faithfulness and Answer Relevancy using the Gemini API.
- **Export Formats**: Exporter for downloading conversations as formatted `.txt` or `.pdf` files.
- **Responsive UI**: Streamlit interface styled using premium typography and HSL-tailored colors.

---

## 🛠️ Technology Stack
- **Programming Language**: Python
- **Frontend**: Streamlit
- **RAG Framework**: LangChain & LangChain-Community
- **LLM**: Google Gemini 2.5 Flash (`langchain-google-genai` / `google-generativeai`)
- **Embeddings**: Sentence-Transformers via HuggingFace
- **Vector DBs**: FAISS & ChromaDB
- **Evaluation**: RAGAS (Faithfulness and Answer Relevancy)
- **Document Exporters**: ReportLab (PDF) & Python I/O (TXT)

---

## ⚙️ System Workflow
1. **Upload Documents / URLs**: User uploads PDFs, TXTs, or submits URLs via the Streamlit sidebar.
2. **Load & Extract**: System routes the document to the corresponding loader (`PyPDFLoader`, `TextLoader`, `WebBaseLoader`).
3. **Split Text**: Content is segmented into overlapping chunks using `RecursiveCharacterTextSplitter`.
4. **Generate Embeddings**: HuggingFace models embed the text chunks.
5. **Index Vectors**: Store vectors and text chunks into **FAISS** or **ChromaDB**.
6. **Query & Retrieval**: User submits a query; a hybrid search (semantic vector + BM25 keyword) retrieves relevant chunks, fused using RRF.
7. **Answer Generation**: Google Gemini processes the question and retrieved context to produce a grounded response.
8. **RAGAS Evaluation**: Evaluates faithfulness and answer relevancy metrics.
9. **Display**: Visualizes answers, citations, scores, and evaluation metrics in Streamlit tabs.

---

## 📂 Folder Structure

```text
Smart-Research-Assistant/
├── app.py                  # Main Streamlit UI Entrypoint
├── requirements.txt        # Package dependencies
├── README.md               # Documentation
├── .env.example            # Environment variables example
├── uploads/                # Directory for temporary file uploads
├── data/                   # General data store
├── vector_store/           # Directory where FAISS & ChromaDB persist files
├── logs/                   # Log file output folder (app.log)
└── src/
    ├── document_loader.py  # PDF, TXT, Web Scraper loaders
    ├── text_splitter.py    # Document splitter
    ├── embeddings.py       # HuggingFace Embedding Manager
    ├── vector_database.py  # FAISS & ChromaDB indices handling
    ├── retriever.py        # BM25 + Semantic Hybrid retrieval (RRF)
    ├── llm.py              # Google Gemini connection & fallback
    ├── prompts.py          # Custom system prompt templates
    ├── rag_pipeline.py     # Main pipeline orchestration
    ├── evaluator.py        # RAGAS Faithfulness & Relevancy
    ├── memory.py           # Conversation window memory
    └── utils.py            # PDF/TXT export & token counting helpers
```

---

## 🔑 Environment Variables

To run the application, create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Open `.env` and enter your Gemini API Key:
```text
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

## 🚀 Running Instructions

### 1. Prerequisites
Ensure you have Python 3.9, 3.10, or 3.11 installed.

### 2. Installation
Clone or navigate to the project directory and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Streamlit application:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 RAGAS Evaluation Details

For a live, production chatbot, we evaluate:
1. **Faithfulness**: Validates that all factual claims made in the answer can be traced directly back to the retrieved contexts.
2. **Answer Relevancy**: Validates that the generated response aligns with the topic/intent of the question.

*Note: Since live user queries do not contain predefined ground truth reference answers, context precision/accuracy are not evaluated. This prevents skewing metrics or throwing runtime exceptions.*

---

## 🔮 Future Enhancements
- **Multi-modal RAG**: Load and parse charts, images, and tables from PDFs.
- **Reranking**: Integrate Cohere or BGE rerankers to improve top-k accuracy before passing to the LLM.
- **Local LLM integration**: Support running local models using Ollama or LlamaCpp.
