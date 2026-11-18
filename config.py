import os
from collections.abc import Mapping
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


def _normalize(name: str) -> str:
    return str(name).strip().replace("-", "_").replace(".", "_").lower()


def _secret_value(*names: str) -> str:
    targets = {_normalize(name) for name in names}
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:
        return ""

    def search(mapping, prefix: str = "") -> str:
        try:
            items = mapping.items()
        except Exception:
            return ""
        for key, value in items:
            key_norm = _normalize(key)
            full = f"{prefix}{key_norm}"
            if isinstance(value, Mapping):
                if full in targets or key_norm in targets:
                    for inner in ("value", "api_key", "key", "token"):
                        if inner in value:
                            text = str(value[inner]).strip()
                            if text:
                                return text
                found = search(value, prefix=f"{full}_")
                if found:
                    return found
                continue
            if key_norm in targets or full in targets:
                text = str(value).strip()
                if text:
                    return text
        return ""

    return search(secrets)


def _setting(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    return _secret_value(name) or default


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
    key = os.getenv("GROQ_API_KEY", "").strip() or os.getenv(
        "GROQ_KEY", ""
    ).strip()
    if not key or key == PLACEHOLDER_API_KEY:
        key = _secret_value(
            "GROQ_API_KEY", "groq.api_key", "GROQ_KEY", "GROQ_TOKEN"
        )
    return "" if key == PLACEHOLDER_API_KEY else key


def has_groq_api_key() -> bool:
    return bool(get_groq_api_key())


def secrets_diagnostics() -> dict:
    info = {
        "env_key_present": bool(os.getenv("GROQ_API_KEY", "").strip()),
        "streamlit_secrets_available": False,
        "secret_names": [],
        "groq_key_found": False,
    }
    try:
        import streamlit as st

        secrets = st.secrets
        info["streamlit_secrets_available"] = True
        names: list[str] = []
        for key, value in secrets.items():
            if isinstance(value, Mapping):
                names.extend(f"{key}.{inner}" for inner in value.keys())
            else:
                names.append(str(key))
        info["secret_names"] = sorted(names)
    except Exception as exc:
        info["error"] = type(exc).__name__
    info["groq_key_found"] = has_groq_api_key()
    return info
