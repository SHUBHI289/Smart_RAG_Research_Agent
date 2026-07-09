import os
import sys
import asyncio

# Suppress Proactor connection reset noise on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure there is an event loop in the current thread (ScriptRunner.scriptThread)
try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import streamlit as tf
from dotenv import load_dotenv



# Set page config FIRST before imports that might call streamlit methods
tf.set_page_config(
    page_title="Smart Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

import pandas as pd
from typing import List, Dict, Any
from src.utils import (
    logger,
    save_uploaded_file, 
    clear_uploads_directory, 
    export_chat_to_txt, 
    export_chat_to_pdf
)
from src.rag_pipeline import RAGPipeline

# Define Custom CSS for Premium Design
CUSTOM_CSS = """
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Title and Header customization */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .subtitle {
        font-size: 1.15rem;
        color: var(--text-color);
        opacity: 0.8;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    /* Styled Metric Cards */
    .metric-card {
        background-color: rgba(150, 150, 150, 0.08);
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        border-top: 1px solid rgba(150, 150, 150, 0.1);
        border-right: 1px solid rgba(150, 150, 150, 0.1);
        border-bottom: 1px solid rgba(150, 150, 150, 0.1);
    }
    .metric-title {
        font-size: 0.825rem;
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-color);
        margin-top: 5px;
    }
    
    /* Source Highlight */
    .source-block {
        background-color: rgba(150, 150, 150, 0.05);
        border: 1px solid rgba(150, 150, 150, 0.15);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .source-header {
        font-weight: 600;
        color: var(--text-color);
        border-bottom: 1px solid rgba(150, 150, 150, 0.15);
        padding-bottom: 8px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
    }
    .source-body {
        font-size: 0.95rem;
        color: var(--text-color);
        opacity: 0.9;
        line-height: 1.6;
        white-space: pre-wrap;
    }
</style>
"""
tf.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "messages" not in tf.session_state:
    tf.session_state.messages = []
if "last_sources" not in tf.session_state:
    tf.session_state.last_sources = []
if "last_eval" not in tf.session_state:
    tf.session_state.last_eval = {"faithfulness": 0.0, "answer_relevancy": 0.0}
if "doc_stats" not in tf.session_state:
    tf.session_state.doc_stats = None
if "api_key" not in tf.session_state:
    tf.session_state.api_key = os.environ.get("GOOGLE_API_KEY", "")

# ----------------- SIDEBAR -----------------
tf.sidebar.markdown("""
<div style="text-align: center; padding-bottom: 15px;">
    <h1 style="font-size: 2.2rem; margin: 0; padding: 0;">🔍</h1>
    <h3 style="margin-top: 5px; font-weight: 700; color: var(--text-color);">Configuration Panel</h3>
</div>
""", unsafe_allow_html=True)

# API Key input
api_key_input = tf.sidebar.text_input(
    "Google Gemini API Key",
    value=tf.session_state.api_key,
    type="password",
    help="Enter your Google Gemini API Key. If set in the .env file, it will be loaded automatically."
)
if api_key_input != tf.session_state.api_key:
    tf.session_state.api_key = api_key_input
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

# Instantiate RAG pipeline
pipeline = RAGPipeline(api_key=tf.session_state.api_key)

# Document Uploads
tf.sidebar.header("1. Ingestion Sources")
uploaded_files = tf.sidebar.file_uploader(
    "Upload Files (PDF, TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

url_input = tf.sidebar.text_area(
    "Add Website URLs (One per line)", 
    placeholder="https://example.com/paper\nhttps://example.com/policy"
)

# Vector Database and Embedding Configuration
tf.sidebar.header("2. RAG Settings")
db_choice = tf.sidebar.selectbox(
    "Vector Database",
    options=["FAISS", "ChromaDB"],
    index=0
)

embedding_choice = tf.sidebar.selectbox(
    "Embedding Model",
    options=[
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5"
    ],
    index=0
)

# Retrieval Search Mode
search_choice = tf.sidebar.selectbox(
    "Search Type",
    options=["Hybrid", "Semantic", "Keyword"],
    index=0
)

# Advanced Configuration Expandable
with tf.sidebar.expander("Advanced Chunking & Model Tuning", expanded=False):
    chunk_size = tf.slider("Chunk Size", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = tf.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
    top_k = tf.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=4, step=1)
    temperature = tf.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

# Indexing Button
build_kb = tf.sidebar.button("⚙️ Build Knowledge Base", use_container_width=True)

# Database Control Buttons
col_clear_db, col_clear_chat = tf.sidebar.columns(2)
with col_clear_db:
    clear_db = tf.button("🗑️ Clear DB", use_container_width=True, help="Wipes the selected vector store directory.")
with col_clear_chat:
    clear_chat = tf.button("🧹 Clear Chat", use_container_width=True, help="Clears conversation memory and screen.")

# Process Database Clears
if clear_db:
    try:
        pipeline.clear_database(db_choice, embedding_choice)
        tf.session_state.doc_stats = None
        tf.sidebar.success(f"Cleared {db_choice} for {embedding_choice} successfully!")
    except Exception as e:
        tf.sidebar.error(f"Error: {str(e)}")

if clear_chat:
    pipeline.memory_manager.clear()
    tf.session_state.messages = []
    tf.session_state.last_sources = []
    tf.session_state.last_eval = {"faithfulness": 0.0, "answer_relevancy": 0.0}
    tf.sidebar.success("Chat history cleared!")

# Build Knowledge Base Logic
if build_kb:
    if not tf.session_state.api_key:
        tf.sidebar.error("Error: Please provide a Google Gemini API Key first.")
    else:
        paths_or_urls = []
        if uploaded_files:
            for file in uploaded_files:
                try:
                    file_path = save_uploaded_file(file)
                    paths_or_urls.append(file_path)
                except Exception as e:
                    tf.sidebar.error(f"Failed to process {file.name}: {str(e)}")

        if url_input.strip():
            urls = [url.strip() for url in url_input.split("\n") if url.strip()]
            paths_or_urls.extend(urls)

        if not paths_or_urls:
            tf.sidebar.warning("Please upload files or input URLs to index.")
        else:
            with tf.sidebar.status("🔄 Ingesting documents & building index...", expanded=True) as status:
                try:
                    stats = pipeline.ingest_sources(
                        paths_or_urls=paths_or_urls,
                        db_type=db_choice,
                        embedding_model=embedding_choice,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    tf.session_state.doc_stats = stats
                    clear_uploads_directory()
                    status.update(label="✅ Knowledge Base built successfully!", state="complete", expanded=False)
                    tf.sidebar.success(f"Indexed {stats['total_chunks']} chunks from sources.")
                except Exception as e:
                    status.update(label="❌ Indexing failed!", state="error", expanded=True)
                    tf.sidebar.error(f"Error: {str(e)}")


# ----------------- MAIN WINDOW -----------------

tf.markdown('<div class="main-title">Smart Research Assistant</div>', unsafe_allow_html=True)
tf.markdown('<div class="subtitle">Search, analyze, and query document collections securely via RAG & Gemini</div>', unsafe_allow_html=True)

# Tabs
tab_chat, tab_sources, tab_eval, tab_stats = tf.tabs([
    "💬 Chat Interface", 
    "📂 Retrieved Sources", 
    "📊 RAGAS Evaluation", 
    "📈 Document Statistics"
])

# ---- TAB 1: Chat Interface ----
with tab_chat:
    if tf.session_state.messages:
        col_space, col_dl_txt, col_dl_pdf = tf.columns([6, 2, 2])
        with col_dl_txt:
            txt_data = export_chat_to_txt(tf.session_state.messages)
            tf.download_button(
                label="📥 Download Chat as TXT",
                data=txt_data,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_dl_pdf:
            try:
                pdf_buffer = export_chat_to_pdf(tf.session_state.messages)
                tf.download_button(
                    label="📥 Download Chat as PDF",
                    data=pdf_buffer,
                    file_name="chat_history.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                tf.error(f"PDF creation failed: {str(e)}")

    chat_container = tf.container()
    with chat_container:
        for message in tf.session_state.messages:
            with tf.chat_message(message["role"]):
                tf.markdown(message["content"])

    user_query = tf.chat_input("Ask a question about the uploaded documents...")

    if user_query:
        with tf.chat_message("user"):
            tf.markdown(user_query)
        tf.session_state.messages.append({"role": "user", "content": user_query})

        with tf.chat_message("assistant"):
            response_placeholder = tf.empty()
            status_placeholder = tf.empty()
            status_placeholder.markdown("🔍 *Searching database and generating response...*")
            
            try:
                stream = pipeline.query_stream(
                    question=user_query,
                    db_type=db_choice,
                    embedding_model=embedding_choice,
                    search_type=search_choice,
                    top_k=top_k,
                    temperature=temperature,
                    use_history=True
                )
                
                full_response = ""
                retrieved_docs = []
                
                for chunk in stream:
                    if chunk["type"] == "sources":
                        retrieved_docs = chunk["data"]
                        tf.session_state.last_sources = retrieved_docs
                    elif chunk["type"] == "chunk":
                        full_response += chunk["text"]
                        response_placeholder.markdown(full_response + "▌")
                    elif chunk["type"] == "evaluation":
                        tf.session_state.last_eval = chunk["scores"]
                
                response_placeholder.markdown(full_response)
                status_placeholder.empty()
                tf.session_state.messages.append({"role": "assistant", "content": full_response})
                tf.rerun()

            except Exception as e:
                status_placeholder.empty()
                error_msg = f"An error occurred: {str(e)}"
                tf.error(error_msg)
                logger.error(f"Error querying pipeline: {str(e)}")

# ---- TAB 2: Retrieved Sources ----
with tab_sources:
    tf.subheader("Retrieved Document Chunks")
    tf.write("Below are the most relevant sections retrieved from your knowledge base to answer the last question.")
    
    if not tf.session_state.last_sources:
        tf.info("No sources retrieved yet. Submit a query in the Chat interface to retrieve documents.")
    else:
        for idx, src in enumerate(tf.session_state.last_sources):
            doc = src["document"]
            title = src["source"]
            page = src["page"]
            score = src["score"]
            search_type = src.get("type", "Hybrid")
            
            tf.markdown(f"""
            <div class="source-block">
                <div class="source-header">
                    <span>📄 Chunk {idx + 1}: <b>{title}</b> (Page {page})</span>
                    <span style="color: #2563EB;">Search Method: {search_type} | Similarity: {score:.2f}</span>
                </div>
                <div class="source-body">{doc.page_content}</div>
            </div>
            """, unsafe_allow_html=True)

# ---- TAB 3: RAGAS Evaluation ----
with tab_eval:
    tf.subheader("RAG Evaluation Metrics (powered by RAGAS)")
    tf.write("RAGAS automatically evaluates the quality of the generated responses on the following dimensions:")
    
    tf.markdown("""
    - **Faithfulness**: Measures whether the generated answer is grounded *only* in the retrieved context. High scores prevent hallucinations.
    - **Answer Relevancy**: Evaluates if the generated answer directly addresses the user query.
    """)
    
    col_faith, col_relevancy = tf.columns(2)
    
    with col_faith:
        score_val = tf.session_state.last_eval.get("faithfulness", 0.0)
        tf.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Faithfulness Score</div>
            <div class="metric-value">{score_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        tf.progress(int(score_val * 100))
        
    with col_relevancy:
        score_val = tf.session_state.last_eval.get("answer_relevancy", 0.0)
        tf.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Answer Relevancy Score</div>
            <div class="metric-value">{score_val:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        tf.progress(int(score_val * 100))

    tf.markdown("""
    > [!NOTE]
    > RAGAS evaluations are computed dynamically using Google Gemini as the LLM-evaluator. 
    > Since there is no human-annotated ground truth for arbitrary user queries, the scores are computed on the groundedness of the retrieved text.
    """)

# ---- TAB 4: Document Statistics ----
with tab_stats:
    tf.subheader("Current Knowledge Base Statistics")
    
    if tf.session_state.doc_stats is None:
        tf.info("No documents indexed. Upload files or URLs in the sidebar and build the Knowledge Base to view statistics.")
    else:
        stats = tf.session_state.doc_stats
        col_sources, col_pages, col_chunks, col_tokens = tf.columns(4)
        
        with col_sources:
            tf.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Sources</div>
                <div class="metric-value">{stats['total_sources']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_pages:
            tf.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Pages</div>
                <div class="metric-value">{stats['total_pages']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_chunks:
            tf.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Chunks</div>
                <div class="metric-value">{stats['total_chunks']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_tokens:
            tf.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Est. Token Count</div>
                <div class="metric-value">{stats['total_tokens']:,}</div>
            </div>
            """, unsafe_allow_html=True)

        if stats.get("errors"):
            tf.warning("Some sources encountered errors during ingestion:")
            for err in stats["errors"]:
                tf.write(f"- {err}")
