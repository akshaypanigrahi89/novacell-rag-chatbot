import sqlite3

from langchain_core.documents import Document

from config import TICKETS_COLLECTION, TICKETS_DB_PATH
from vector_store import ingest_documents


def build_documents() -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []

    connection = sqlite3.connect(TICKETS_DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT ticket_id, category, issue_type, description, resolution, status
            FROM tickets
            WHERE status = 'resolved'
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()

    for row in rows:
        ticket_id = row["ticket_id"]
        documents.append(
            Document(
                page_content=(
                    f"Ticket {ticket_id} - {row['category']} / {row['issue_type']}\n"
                    f"Customer issue: {row['description']}\n"
                    f"Resolution: {row['resolution']}"
                ),
                metadata={
                    "source": "TICKET",
                    "identifier": ticket_id,
                    "category": row["category"],
                    "issue_type": row["issue_type"],
                },
            )
        )
        ids.append(f"ticket-{ticket_id}")

    return documents, ids


def ingest() -> int:
    documents, ids = build_documents()
    return ingest_documents(TICKETS_COLLECTION, documents, ids)


if __name__ == "__main__":
    count = ingest()
    print(f"[tickets] Ingested {count} resolved tickets into '{TICKETS_COLLECTION}'.")
