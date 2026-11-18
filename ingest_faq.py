import csv

from langchain_core.documents import Document

from config import FAQ_COLLECTION, FAQ_PATH
from vector_store import ingest_documents


def build_documents() -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []

    with FAQ_PATH.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue

            faq_id = (row.get("id") or str(len(documents) + 1)).strip()
            category = (row.get("category") or "").strip()
            documents.append(
                Document(
                    page_content=f"Question: {question}\nAnswer: {answer}",
                    metadata={
                        "source": "FAQ",
                        "identifier": faq_id,
                        "faq_id": faq_id,
                        "category": category,
                    },
                )
            )
            ids.append(f"faq-{faq_id}")

    return documents, ids


def ingest() -> int:
    documents, ids = build_documents()
    return ingest_documents(FAQ_COLLECTION, documents, ids)


if __name__ == "__main__":
    count = ingest()
    print(f"[faq] Ingested {count} FAQ entries into '{FAQ_COLLECTION}'.")
