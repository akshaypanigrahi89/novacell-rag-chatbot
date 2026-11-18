from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from langchain_core.documents import Document

from config import (
    FAQ_COLLECTION,
    GUIDES_COLLECTION,
    TICKETS_COLLECTION,
    TOP_K,
)
from vector_store import get_client, get_vectorstore

COLLECTIONS = [
    {"name": FAQ_COLLECTION, "label": "FAQ"},
    {"name": TICKETS_COLLECTION, "label": "TICKET"},
    {"name": GUIDES_COLLECTION, "label": "GUIDE"},
]


@dataclass
class RetrievedDocument:
    label: str
    identifier: str
    content: str
    category: str = ""
    metadata: dict = field(default_factory=dict)

    def as_source(self) -> dict:
        source = {
            "label": self.label,
            "identifier": self.identifier,
            "content": self.content,
        }
        if self.category:
            source["category"] = self.category
        return source


def _to_retrieved(label: str, document: Document) -> RetrievedDocument:
    metadata = document.metadata or {}
    identifier = str(
        metadata.get("identifier")
        or metadata.get("faq_id")
        or metadata.get("ticket_id")
        or "unknown"
    )
    return RetrievedDocument(
        label=label,
        identifier=identifier,
        content=document.page_content.strip(),
        category=str(metadata.get("category") or ""),
        metadata=metadata,
    )


def _search_collection(collection: dict, question: str, top_k: int) -> list[RetrievedDocument]:
    store = get_vectorstore(collection["name"])
    documents = store.similarity_search(question, k=top_k)
    return [_to_retrieved(collection["label"], document) for document in documents]


def retrieve(question: str, top_k: int = TOP_K) -> list[RetrievedDocument]:
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    with ThreadPoolExecutor(max_workers=len(COLLECTIONS)) as executor:
        futures = [
            executor.submit(_search_collection, collection, question, top_k)
            for collection in COLLECTIONS
        ]
        grouped = [future.result() for future in futures]

    documents = [document for group in grouped for document in group]
    if not documents:
        raise RuntimeError(
            "No knowledge documents were retrieved. Run 'python ingest_all.py' "
            "to build the Chroma store."
        )
    return documents


def collection_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    client = get_client()
    for collection in COLLECTIONS:
        try:
            counts[collection["label"]] = client.get_collection(
                collection["name"]
            ).count()
        except Exception:
            counts[collection["label"]] = 0
    return counts


def knowledge_base_ready() -> bool:
    return sum(collection_counts().values()) > 0
