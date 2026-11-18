import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from config import FEEDBACK_LOG_PATH, LOG_DIR


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_feedback(
    *,
    question: str,
    answer: str,
    sources: list[dict],
    rating: str,
    session_id: str,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "sources": sources,
        "rating": rating,
    }
    try:
        _append(FEEDBACK_LOG_PATH, record)
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "novacell_logs" / "interactions.jsonl"
        try:
            _append(fallback, record)
        except OSError:
            pass
