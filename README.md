# NovaCell Customer Care — RAG Chatbot

A self-serve telecom support assistant that answers Tier-1 questions using
Retrieval-Augmented Generation. Every answer is grounded in NovaCell's verified
knowledge — FAQ entries, resolved support tickets, and an official PDF guide —
and the bot refuses to guess when the knowledge base does not cover a question.

Built to the requirements in [`prd.md`](prd.md).

## Features

- **Grounded answers only** — the LLM receives source-labelled context and is
  instructed never to use outside knowledge.
- **Three parallel knowledge collections** — top-3 FAQ entries, top-3 resolved
  tickets, and top-3 guide chunks (9 documents per query).
- **Expandable Sources panel** — every answer lists the FAQ / TICKET / GUIDE
  documents used, with identifiers.
- **Streaming responses** in both the browser UI and the CLI.
- **Graceful fallback** — when context is insufficient, the bot says so and
  directs the customer to 611 or the MyTelecom app.
- **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2` runs on the
  machine; only generation calls the Groq API.
- **Feedback capture** — 👍 / 👎 on every answer is appended to
  `logs/interactions.jsonl`.
- **Idempotent ingestion** — re-running an ingest script rebuilds its
  collection without duplicating documents.

## Architecture

```
User question
     |
     v
Merged retriever (parallel)
  |-- ChromaDB faq       top-3 questions/answers
  |-- ChromaDB tickets   top-3 resolved tickets
  |-- ChromaDB guides    top-3 PDF guide chunks
     |
     v  9 source-labelled documents
ChatPromptTemplate (grounding rules + context)
     |
     v
qwen/qwen3.8-27b on Groq (temperature 0, reasoning_effort "none")
     |
     v
Streamed answer + Sources panel
```

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local)
- Vector store: ChromaDB persisted to `chroma_store/`
- LLM: Groq API, model configurable in `.env`
- Framework: LangChain LCEL
- UI: Streamlit, plus a CLI REPL

## Quickstart

Requires Python 3.11+ and a free Groq API key from
[console.groq.com](https://console.groq.com).

```powershell
# 1. Project-local virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Configure the API key
Copy-Item .env.example .env
# Edit .env and set GROQ_API_KEY=...

# 4. Build the knowledge base (once)
python ingest_all.py

# 5. Run the browser app
streamlit run app.py

# or run the CLI
python main.py
```

macOS/Linux equivalents:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python ingest_all.py
streamlit run app.py
```

If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or call
`.\.venv\Scripts\python.exe` directly.

## Deploy to Streamlit Community Cloud

The public deployment repo is
[`akshaypanigrahi89/novacell-rag-chatbot`](https://github.com/akshaypanigrahi89/novacell-rag-chatbot).

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=akshaypanigrahi89/novacell-rag-chatbot&branch=main&mainModule=app.py)

1. Open the deploy link and sign in with GitHub.
2. Confirm the repository, branch `main`, and entrypoint `app.py`.
3. In **Advanced settings**, select **Python 3.12**.
4. Paste the following into **Secrets** and replace the key:

   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   GROQ_MODEL = "qwen/qwen3.8-27b"
   GROQ_REASONING_EFFORT = "none"
   ```

5. Choose an app subdomain and click **Deploy**.

Notes:

- `chroma_store/` is committed, so the deployed app does not ingest at startup.
- The first load downloads the ~90 MB embedding model; cold starts can take
  1–3 minutes.
- Feedback logs live on the container's ephemeral filesystem and reset on
  redeploy or restart.
- To update the knowledge base, re-run `python ingest_all.py`, then commit and
  push `data/` and `chroma_store/`.

## Configuration

`.env` (never commit this file):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Required for answer generation |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq model ID |
| `GROQ_REASONING_EFFORT` | `none` | Disables reasoning tokens |

If your Groq account does not expose the default model, set `GROQ_MODEL` to an
available model and restart the app. The PRD named `qwen/qwen3.6-27b`; the
default here is `qwen/qwen3.8-27b`, the Qwen 27B model currently available in
Groq's catalog.

## Knowledge ingestion

| Collection | Source | Documents |
|---|---|---|
| `faq` | `data/faq.csv` | 1 vector per row |
| `tickets` | `data/tickets.db` (resolved rows) | 1 vector per ticket |
| `guides` | `data/telecom_guide.pdf` | 600-character chunks, 100-character overlap |

Run all three:

```powershell
python ingest_all.py
```

Or rebuild one collection:

```powershell
python ingest_faq.py
python ingest_tickets.py
python ingest_guides.py
```

Support operations can update the source files and re-run the relevant script;
no application code change or redeploy is needed. The app never re-ingests at
startup.

## CLI

```text
You: How do I activate roaming before my trip?
Assistant: ...streamed answer...
Sources:
  - [FAQ 9] (roaming)
  - [TICKET TK-012] (roaming)
  ...
```

Type `quit` (or `exit`) to end the session.

## Adding a new knowledge source

1. Write `ingest_<source>.py` following the existing scripts and call
   `ingest_documents(...)`.
2. Add the collection name and source label to `COLLECTIONS` in
   `retriever.py`.
3. Run the new ingest script.

## Feedback log

Each rating appends one JSON line to `logs/interactions.jsonl` containing the
timestamp, session ID, question, answer, retrieved sources, and rating
(`up` / `down`). The `logs/` directory is gitignored.

## Project layout

```
app.py               Streamlit chat UI
main.py              CLI REPL
rag.py               Grounding prompt, Groq LLM, streaming chain
retriever.py         Parallel multi-collection retrieval
vector_store.py      Chroma client and ingestion helper
embedding.py         Local HuggingFace embeddings
feedback.py          Interaction log writer
ingest_all.py        Rebuild all collections
ingest_faq.py        FAQ CSV ingestion
ingest_tickets.py    SQLite ticket ingestion
ingest_guides.py     PDF chunking and ingestion
config.py            Paths, model settings, sample questions
data/                FAQ, tickets database, PDF guide
chroma_store/        Persisted vector store (generated)
logs/                Feedback log (generated)
```

## Troubleshooting

- **"The knowledge base is empty"** — run `python ingest_all.py`.
- **"GROQ_API_KEY is not configured"** — copy `.env.example` to `.env`, add the
  key, and restart Streamlit.
- **Model not found (404)** — set `GROQ_MODEL` in `.env` to a model available on
  your Groq account.
- **First run is slow** — the ~90 MB embedding model downloads once, is warmed
  up at app/CLI startup, and is then cached locally. Warm queries return the
  first token in under a second in local testing.
