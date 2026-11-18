from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import (
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    LLM_TEMPERATURE,
    get_groq_api_key,
    has_groq_api_key,
)
from retriever import RetrievedDocument

SYSTEM_PROMPT = """You are NovaCell's customer care assistant. You answer Tier-1 telecom support questions for NovaCell subscribers.

Use ONLY the information inside the CONTEXT block. The CONTEXT block contains excerpts from NovaCell's verified FAQ, resolved support tickets, and official user guides.

Rules:
1. Ground every factual statement in CONTEXT. Never use outside knowledge.
2. Never invent prices, fees, balances, plan details, coverage, policies, timeframes, or troubleshooting steps that are not in CONTEXT.
3. Cite the sources you use inline with the exact labels from CONTEXT, for example [FAQ 4], [TICKET TK-002], or [GUIDE p.2 c.1].
4. If CONTEXT does not contain enough information to answer confidently, do not guess. Reply: "I don't have enough verified information in NovaCell's knowledge base to answer that confidently. Please call 611 (free from your mobile) or use the MyTelecom app's live chat (8am-10pm) so a support agent can help." You may add one sentence pointing to a related topic only if that topic is in CONTEXT.
5. You cannot access customer accounts, live billing, or usage data. For personalized requests such as "what is my balance" or "why was I charged", explain that you cannot see their account and direct them to the MyTelecom app or 611. General guidance from CONTEXT may still be shared.
6. Be concise, use plain language, and use numbered steps for troubleshooting.

CONTEXT:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def format_context(documents: list[RetrievedDocument]) -> str:
    blocks = []
    for document in documents:
        blocks.append(
            f"[{document.label} {document.identifier}]\n{document.content}"
        )
    return "\n\n---\n\n".join(blocks)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not has_groq_api_key():
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Copy .env.example to .env and add "
            "your key from console.groq.com."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=get_groq_api_key(),
        streaming=True,
        reasoning_effort=GROQ_REASONING_EFFORT,
    )


def build_chain():
    return PROMPT | get_llm() | StrOutputParser()


def stream_answer(question: str, documents: list[RetrievedDocument]):
    chain = build_chain()
    inputs = {
        "context": format_context(documents),
        "question": question,
    }
    yield from chain.stream(inputs)


def answer_question(question: str, documents: list[RetrievedDocument]) -> str:
    chain = build_chain()
    return chain.invoke(
        {
            "context": format_context(documents),
            "question": question,
        }
    )
