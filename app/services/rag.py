"""
RAG retrieval + chat generation.

Pipeline: embed the user's question with the same BGE-large model used
at ingestion time -> pull the top-k nearest chunks for one document via
pgvector cosine distance -> stuff them into a prompt with page numbers
-> stream a completion back from Groq (OpenAI-compatible endpoint).

Chat is scoped to a single document_id, matching the split-pane
workspace (one doc open at a time in the viewer).

Step 8 (highlight-to-ask) adds a second path: when the frontend sends
highlighted_text/highlighted_page, skip pgvector retrieval entirely and
answer from just that passage — see build_scoped_messages below.
"""

import json
import uuid
from typing import AsyncGenerator

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.models import DocumentChunk
from app.services.embeddings import embed_chunks

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

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


def build_messages(
    chunks: list[DocumentChunk],
    question: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """Assemble the OpenAI-style message list Groq expects."""
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


async def stream_answer(messages: list[dict]) -> AsyncGenerator[str, None]:
    """Yield answer text incrementally from Groq's streaming chat endpoint."""
    if not settings.groq_api_key:
        yield (
            "[Setup needed] GROQ_API_KEY is empty in .env. Get a free key "
            "at console.groq.com and add it, then restart uvicorn."
        )
        return

    payload = {
        "model": settings.groq_model,
        "messages": messages,
        "stream": True,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", GROQ_CHAT_URL, json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                yield f"[Groq error {response.status_code}] {body.decode(errors='replace')}"
                return

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = parsed.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content")
                if token:
                    yield token