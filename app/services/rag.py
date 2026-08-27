"""
RAG retrieval + chat generation.

Pipeline: embed the user's question with the same BGE-large model used
at ingestion time -> pull the top-k nearest chunks via pgvector cosine
distance -> stuff them into a prompt with page numbers -> stream a
completion back from Gemini (OpenAI-compatible endpoint).

Two retrieval scopes are supported:
  - Single document (retrieve_chunks / build_messages) — the original
    split-pane workspace flow, one doc open at a time.
  - Across all of a user's documents (retrieve_chunks_across_documents /
    build_multi_doc_messages) — Day 2's "ask across all my documents"
    feature. Citations here include the source document's title, not
    just its page number, since the answer may draw from several files.

Step 8 (highlight-to-ask) adds a third path: when the frontend sends
highlighted_text/highlighted_page, skip pgvector retrieval entirely and
answer from just that passage — see build_scoped_messages below.
"""

import uuid
from typing import AsyncGenerator

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document, DocumentChunk
from app.services.embeddings import embed_chunks

chat_client = AsyncOpenAI(
    api_key=settings.google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

CHAT_MODEL = "gemini-3.6-flash"  # same model used in summarization.py

SYSTEM_PROMPT = (
    "You are DocuSense AI, an assistant that answers questions about a "
    "single uploaded document using only the excerpts provided below. "
    "Every claim you make must be grounded in the excerpts. If the "
    "excerpts don't contain the answer, say so plainly instead of "
    "guessing. When you use an excerpt, cite its page number in the "
    "form (p. N)."
)

SCOPED_SYSTEM_PROMPT = (
    "You are DocuSense AI. The user has highlighted one specific passage "
    "in their document and is asking about that passage only. Answer "
    "using only the highlighted text below — do not draw on the rest of "
    "the document. If the passage doesn't contain the answer, say so "
    "plainly instead of guessing."
)

MULTI_DOC_SYSTEM_PROMPT = (
    "You are DocuSense AI, an assistant that answers questions across a "
    "user's entire document library, using only the excerpts provided "
    "below. Excerpts may come from different documents — every claim "
    "you make must be grounded in the excerpts, and you must cite which "
    "document each claim comes from. If the excerpts don't contain the "
    "answer, say so plainly instead of guessing. When you use an "
    "excerpt, cite it in the form (\"document title\", p. N)."
)


def embed_query(text: str) -> list[float]:
    """Embed a single query string with the same model used for chunks."""
    return embed_chunks([text])[0]


def retrieve_chunks(
    db: Session,
    document_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[DocumentChunk]:
    """Top-k chunks for one document, nearest first, by cosine distance."""
    return (
        db.query(DocumentChunk)
        .filter(DocumentChunk.doc_id == document_id)
        .order_by(DocumentChunk.embedding_vector.cosine_distance(query_embedding))
        .limit(top_k)
        .all()
    )


def retrieve_chunks_across_documents(
    db: Session,
    owner_id: uuid.UUID,
    query_embedding: list[float],
    top_k: int = 8,
) -> list[tuple[DocumentChunk, str]]:
    """
    Day 2: top-k chunks across ALL of a user's ready documents, nearest
    first, by cosine distance — no per-document limit, so one document
    could dominate the results if it's simply more relevant to the
    question. Deliberately not doing any per-document quota/reranking
    here, matching the plan's "no reranking" scope for this pass.

    Returns (chunk, document_title) tuples since the caller needs the
    title to cite which document each excerpt came from.
    """
    return (
        db.query(DocumentChunk, Document.title)
        .join(Document, DocumentChunk.doc_id == Document.id)
        .filter(Document.owner_id == owner_id, Document.status == "ready")
        .order_by(DocumentChunk.embedding_vector.cosine_distance(query_embedding))
        .limit(top_k)
        .all()
    )


def build_messages(
    chunks: list[DocumentChunk],
    question: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """Assemble the OpenAI-style message list the chat client expects."""
    if chunks:
        excerpts = "\n\n".join(
            f"[Excerpt from p. {c.page_num}]\n{c.content}" for c in chunks
        )
        context_block = f"Document excerpts:\n\n{excerpts}"
    else:
        context_block = (
            "No excerpts were retrieved for this document — "
            "it may not be indexed yet."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Bounded history: keep only the last few turns so the prompt doesn't
    # grow unbounded across a long chat session.
    for turn in (history or [])[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append(
        {"role": "user", "content": f"{context_block}\n\nQuestion: {question}"}
    )
    return messages


def build_scoped_messages(
    highlighted_text: str,
    highlighted_page: int | None,
    question: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Step 8: message list for a query scoped to one highlighted passage.
    No retrieval — the passage itself is the entire context.
    """
    page_note = f" (page {highlighted_page})" if highlighted_page else ""
    context_block = f"Highlighted passage{page_note}:\n\n{highlighted_text}"

    messages = [{"role": "system", "content": SCOPED_SYSTEM_PROMPT}]

    for turn in (history or [])[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append(
        {"role": "user", "content": f"{context_block}\n\nQuestion: {question}"}
    )
    return messages


def build_multi_doc_messages(
    chunks_with_titles: list[tuple[DocumentChunk, str]],
    question: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Day 2: message list for a query scoped across all of a user's
    documents. Each excerpt is labeled with its source document's
    title (not just its page number) so the model — and the citations
    the frontend renders — can tell them apart.
    """
    if chunks_with_titles:
        excerpts = "\n\n".join(
            f'[Excerpt from "{title}", p. {c.page_num}]\n{c.content}'
            for c, title in chunks_with_titles
        )
        context_block = (
            f"Document excerpts (from across your uploaded documents):\n\n{excerpts}"
        )
    else:
        context_block = (
            "No excerpts were retrieved from your documents — "
            "you may not have any ready documents yet."
        )

    messages = [{"role": "system", "content": MULTI_DOC_SYSTEM_PROMPT}]

    for turn in (history or [])[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append(
        {"role": "user", "content": f"{context_block}\n\nQuestion: {question}"}
    )
    return messages


async def stream_answer(messages: list[dict]) -> AsyncGenerator[str, None]:
    """Yield answer text incrementally from Gemini's streaming chat endpoint."""
    if not settings.google_api_key:
        yield (
            "[Setup needed] GOOGLE_API_KEY is empty in .env. Add your key "
            "and restart uvicorn."
        )
        return

    try:
        stream = await chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
            temperature=0.2,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except Exception as e:
        yield f"[Gemini error] {e}"