import ingest_faq
import ingest_guides
import ingest_tickets


def main() -> None:
    faq_count = ingest_faq.ingest()
    ticket_count = ingest_tickets.ingest()
    guide_count = ingest_guides.ingest()

    print(f"[faq]     {faq_count} documents")
    print(f"[tickets] {ticket_count} documents")
    print(f"[guides]  {guide_count} documents")
    print(f"[total]   {faq_count + ticket_count + guide_count} documents")
    print("Chroma store persisted to 'chroma_store/'.")


if __name__ == "__main__":
    main()
