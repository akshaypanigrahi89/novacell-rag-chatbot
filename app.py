import uuid

import streamlit as st

from config import (
    GROQ_MODEL,
    SAMPLE_QUESTIONS,
    has_groq_api_key,
    secrets_diagnostics,
)
from embedding import get_embeddings
from feedback import log_feedback
from rag import stream_answer
from retriever import collection_counts, retrieve

st.set_page_config(page_title="NovaCell Customer Care", layout="centered")

st.markdown(
    """
    <style>
        .block-container { max-width: 900px; }
        [data-testid="stSidebar"] .stButton button { text-align: left; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.pending_question = None


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for index, source in enumerate(sources):
            st.markdown(f"**{source['label']} · {source['identifier']}**")
            if source.get("category"):
                st.caption(source["category"].capitalize())
            st.text(source["content"])
            if index < len(sources) - 1:
                st.divider()


def render_feedback(message_index: int, message: dict) -> None:
    rating = message.get("rating")
    if rating == "up":
        st.caption("You marked this answer as helpful.")
        return
    if rating == "down":
        st.caption("You marked this answer as not helpful.")
        return

    helpful_col, unhelpful_col, _ = st.columns([1.2, 1.4, 5])
    with helpful_col:
        if st.button("👍 Helpful", key=f"fb-up-{message_index}"):
            record_feedback(message_index, "up")
    with unhelpful_col:
        if st.button("👎 Not helpful", key=f"fb-down-{message_index}"):
            record_feedback(message_index, "down")


def record_feedback(message_index: int, rating: str) -> None:
    message = st.session_state.messages[message_index]
    message["rating"] = rating
    log_feedback(
        question=message.get("question", ""),
        answer=message["content"],
        sources=message.get("sources", []),
        rating=rating,
        session_id=st.session_state.session_id,
    )
    st.toast("Thanks for the feedback.")
    st.rerun()


def render_message(message: dict, message_index: int) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            render_feedback(message_index, message)


def process_pending_question() -> None:
    question = st.session_state.pending_question
    if not question:
        return

    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching NovaCell's FAQ, tickets, and guides..."):
            try:
                documents = retrieve(question)
            except Exception as exc:
                st.error(f"Retrieval failed: {exc}")
                return

        answer = ""
        try:
            placeholder = st.empty()
            chunks: list[str] = []
            for token in stream_answer(question, documents):
                chunks.append(token)
                placeholder.markdown("".join(chunks) + "▌")
            answer = "".join(chunks).strip()
            placeholder.markdown(answer)
        except Exception as exc:
            st.error(
                f"Generation failed: {exc}\n\n"
                "Check that GROQ_API_KEY is valid and GROQ_MODEL is available "
                "on your Groq account."
            )
            return

    if answer:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "question": question,
                "sources": [document.as_source() for document in documents],
                "rating": None,
            }
        )
        st.rerun()


init_state()

with st.sidebar:
    st.title("NovaCell Care")
    st.caption("Tier-1 support, grounded in verified knowledge.")

    st.subheader("Try a question")
    ready = collection_counts()
    total_documents = sum(ready.values())
    can_ask = total_documents > 0 and has_groq_api_key()

    for index, sample in enumerate(SAMPLE_QUESTIONS):
        if st.button(
            sample,
            key=f"sample-{index}",
            use_container_width=True,
            disabled=not can_ask,
        ):
            st.session_state.pending_question = sample
            st.rerun()

    st.divider()
    st.caption(
        f"Knowledge: {ready['FAQ']} FAQ · {ready['TICKET']} tickets · "
        f"{ready['GUIDE']} guide chunks"
    )
    st.caption(f"Model: {GROQ_MODEL} via Groq")

    if st.button("Clear conversation", use_container_width=True):
        clear_conversation()
        st.rerun()

    st.divider()
    st.caption(
        "Answers come only from NovaCell's FAQ, resolved tickets, and official "
        "guides. When the bot cannot answer confidently, it directs you to 611 "
        "or the MyTelecom app."
    )

st.title("Customer Care Assistant")
st.caption(
    "Ask about data, billing, SIM, roaming, voice, or account issues. "
    "Every answer is grounded in NovaCell's verified knowledge."
)

with st.spinner("Loading local embedding model (first launch only)..."):
    get_embeddings()

if total_documents == 0:
    st.error(
        "The knowledge base is empty. Build it once with:\n\n"
        "```\npython ingest_all.py\n```"
    )
elif not has_groq_api_key():
    st.warning(
        "GROQ_API_KEY is not configured for this deployment. On Streamlit "
        "Community Cloud, click **Manage app** (lower-right) → **Settings** → "
        "**Secrets**, paste the TOML below, and save. Locally, add the key to "
        "`.env`."
    )
    st.code(
        'GROQ_API_KEY = "gsk_your_key_here"\n'
        'GROQ_MODEL = "qwen/qwen3.8-27b"\n'
        'GROQ_REASONING_EFFORT = "none"',
        language="toml",
    )
    with st.expander("Deployment diagnostics"):
        st.json(secrets_diagnostics())
    if st.button("I added the secret - recheck"):
        st.rerun()

for index, message in enumerate(st.session_state.messages):
    render_message(message, index)

if not st.session_state.messages and total_documents > 0:
    st.info(
        "Type a question below or pick one from the sidebar to get started."
    )

typed = st.chat_input(
    "Ask a support question...",
    disabled=not can_ask,
)
if typed:
    st.session_state.pending_question = typed
    st.rerun()

process_pending_question()
