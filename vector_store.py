import threading

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR
from embedding import get_embeddings

_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                CHROMA_DIR.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(
                    path=str(CHROMA_DIR),
                    settings=Settings(
                        anonymized_telemetry=False, allow_reset=True
                    ),
                )
    return _client


def get_vectorstore(collection_name: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        client=get_client(),
        embedding_function=get_embeddings(),
    )


def reset_collection(collection_name: str) -> None:
    client = get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    client.create_collection(collection_name)


def ingest_documents(
    collection_name: str, documents: list[Document], ids: list[str] | None = None
) -> int:
    if not documents:
        raise ValueError(f"No documents to ingest into '{collection_name}'")

    reset_collection(collection_name)
    store = get_vectorstore(collection_name)
    store.add_documents(documents, ids=ids)
    return len(documents)
