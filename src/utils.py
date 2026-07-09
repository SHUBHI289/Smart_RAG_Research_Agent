import os
import shutil
import logging
from logging.handlers import RotatingFileHandler
import tiktoken
from typing import List, Dict
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================
# 2. LOGGING CONFIGURATION
# ==========================================
LOG_FILE = os.path.join(LOG_DIR, "app.log")

def setup_logger(name: str = "SmartResearchAssistant") -> logging.Logger:
    """
    Configures and returns a rotating file logger and standard console logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # Rotating File Output (5MB size, max 3 files)
        try:
            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to initialize rotating file logger: {str(e)}")

    return logger

logger = setup_logger()

# ==========================================
# 3. TOKEN COUNTING UTILITY
# ==========================================
def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Estimates the number of tokens in a string using tiktoken.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# ==========================================
# 4. UPLOADS MANAGEMENT
# ==========================================
def save_uploaded_file(uploaded_file) -> str:
    """
    Saves an uploaded file to the uploads directory.
    Returns the absolute path of the saved file.
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Successfully saved uploaded file to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving uploaded file {uploaded_file.name}: {str(e)}")
        raise e

def clear_uploads_directory():
    """
    Clears all files in the uploads directory.
    """
    try:
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        logger.info("Cleared uploads directory.")
    except Exception as e:
        logger.error(f"Error clearing uploads directory: {str(e)}")

# ==========================================
# 5. CHAT EXPORTER UTILITIES
# ==========================================
def sanitize_text_for_pdf(text: str) -> str:
    """
    Cleans Unicode characters to avoid rendering issues with default Helvetica.
    """
    return text.encode('ascii', errors='replace').decode('ascii')

def export_chat_to_txt(chat_history: List[Dict[str, str]]) -> str:
    """
    Formats the conversation history as a text file.
    """
    lines = [
        "=" * 60,
        "Smart Research Assistant - Chat History Export",
        "=" * 60 + "\n"
    ]
    for msg in chat_history:
        role = msg.get("role", "User").capitalize()
        content = msg.get("content", "")
        lines.append(f"[{role}]:")
        lines.append(content)
        lines.append("\n" + "-" * 40 + "\n")
    return "\n".join(lines)

def export_chat_to_pdf(chat_history: List[Dict[str, str]]) -> BytesIO:
    """
    Generates a PDF document buffer from the chat history.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=15
    )
    
    user_header_style = ParagraphStyle(
        name='UserHeaderStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#059669'),
        spaceBefore=10,
        spaceAfter=3
    )
    
    assistant_header_style = ParagraphStyle(
        name='AssistantHeaderStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#4B5563'),
        spaceBefore=10,
        spaceAfter=3
    )
    
    msg_style = ParagraphStyle(
        name='MessageStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=10
    )

    story = [
        Paragraph(sanitize_text_for_pdf("Smart Research Assistant - Chat History"), title_style),
        Spacer(1, 10),
        Paragraph(sanitize_text_for_pdf("---------------------------------------------------------------------------------------"), styles['Normal']),
        Spacer(1, 10)
    ]
    
    for msg in chat_history:
        role = msg.get("role", "User").capitalize()
        content = sanitize_text_for_pdf(msg.get("content", ""))
        
        if role == "User":
            story.append(Paragraph("User:", user_header_style))
        else:
            story.append(Paragraph("Assistant:", assistant_header_style))
            
        story.append(Paragraph(content.replace("\n", "<br/>"), msg_style))
        story.append(Spacer(1, 5))
        
    doc.build(story)
    buffer.seek(0)
    return buffer
