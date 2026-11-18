import sys

from config import has_groq_api_key
from embedding import get_embeddings
from rag import stream_answer
from retriever import knowledge_base_ready, retrieve


def _print_sources(documents) -> None:
    print("\nSources:")
    for document in documents:
        label = f"[{document.label} {document.identifier}]"
        category = f" ({document.category})" if document.category else ""
        print(f"  - {label}{category}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("NovaCell Customer Care assistant (CLI)")
    print("Answers are grounded in NovaCell's FAQ, resolved tickets, and guides.")
    print("Type 'quit' to exit.")

    if not knowledge_base_ready():
        print(
            "\nWARNING: The knowledge base is empty. Run 'python ingest_all.py' "
            "before asking questions."
        )
    if not has_groq_api_key():
        print(
            "\nWARNING: GROQ_API_KEY is not configured in .env. "
            "Generation will fail until a key is added."
        )

    if knowledge_base_ready():
        print("\nLoading local embedding model...")
        get_embeddings()

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        try:
            documents = retrieve(question)
        except Exception as exc:
            print(f"Retrieval error: {exc}")
            continue

        print("Assistant: ", end="", flush=True)
        try:
            for token in stream_answer(question, documents):
                print(token, end="", flush=True)
        except Exception as exc:
            print(f"\nGeneration error: {exc}")
            continue
        print()
        _print_sources(documents)


if __name__ == "__main__":
    main()
