from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GUIDE_PDF_PATH,
    GUIDES_COLLECTION,
)
from vector_store import ingest_documents


def build_documents() -> tuple[list[Document], list[str]]:
    reader = PdfReader(str(GUIDE_PDF_PATH))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    ids: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        for chunk_index, chunk in enumerate(splitter.split_text(text), start=1):
            chunk = chunk.strip()
            if not chunk:
                continue

            chunk_id = f"guide-p{page_number}-c{chunk_index}"
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": "GUIDE",
                        "identifier": f"p.{page_number} c.{chunk_index}",
                        "page": page_number,
                        "chunk": chunk_index,
                        "file": GUIDE_PDF_PATH.name,
                    },
                )
            )
            ids.append(chunk_id)

    return documents, ids


def ingest() -> int:
    documents, ids = build_documents()
    return ingest_documents(GUIDES_COLLECTION, documents, ids)


if __name__ == "__main__":
    count = ingest()
    print(f"[guides] Ingested {count} guide chunks into '{GUIDES_COLLECTION}'.")
