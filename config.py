import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_store"
LOG_DIR = ROOT_DIR / "logs"

FAQ_PATH = DATA_DIR / "faq.csv"
TICKETS_DB_PATH = DATA_DIR / "tickets.db"
GUIDE_PDF_PATH = DATA_DIR / "telecom_guide.pdf"
FEEDBACK_LOG_PATH = LOG_DIR / "interactions.jsonl"

PLACEHOLDER_API_KEY = "your_groq_api_key_here"


def _streamlit_secret(name: str) -> str:
    try:
        import streamlit as st

        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def _setting(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    return _streamlit_secret(name) or default


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = _setting("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_REASONING_EFFORT = _setting("GROQ_REASONING_EFFORT", "none")
LLM_TEMPERATURE = 0.0

TOP_K = 3
CONTEXT_DOCS_PER_QUERY = TOP_K * 3
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

FAQ_COLLECTION = "faq"
TICKETS_COLLECTION = "tickets"
GUIDES_COLLECTION = "guides"

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "How do I activate 4G/LTE on my phone?",
    "I was charged twice on my latest bill. What should I do?",
    "How do I set up roaming before my trip abroad?",
    "My phone says the SIM is not detected. How can I fix it?",
    "How do I enable VoLTE on my device?",
]


def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == PLACEHOLDER_API_KEY:
        key = _streamlit_secret("GROQ_API_KEY")
    return "" if key == PLACEHOLDER_API_KEY else key


def has_groq_api_key() -> bool:
    return bool(get_groq_api_key())
