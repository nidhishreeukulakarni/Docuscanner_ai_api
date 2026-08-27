import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.db import SessionLocal
from app.models import ChatMessage, User
from app.dependencies import get_current_user
from app.services import rag

router = APIRouter()


class ChatTurn(BaseModel):
    role: str
    content: str


class MultiDocChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("")
async def chat_across_documents(
    body: MultiDocChatRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Day 2: SSE stream, same shape as chat.chat_with_document, but
    retrieval isn't scoped to one document_id — it searches across
    all of the current user's "ready" documents via pgvector, and
    citations include each excerpt's source document title so the
    user can tell which file an answer came from.

    No highlight-to-ask path here (that only makes sense scoped to
    one open document, see chat.py) — just question + history.

    Persisted the same way as single-doc chat, except doc_id is left
    NULL on both ChatMessage rows, so history for this mode is scoped
    by user_id only (see get_multi_doc_history below).
    """
    if not body.question.strip():
        raise HTTPException(400, "question must not be empty")

    db = SessionLocal()

    # Persist the user's turn right away, before any retrieval/streaming
    # work, so it's saved even if the LLM call fails.
    db.add(
        ChatMessage(
            doc_id=None,
            user_id=current_user.id,
            role="user",
            content=body.question,
            citations_json=None,
        )
    )
    db.commit()

    history_dicts = [t.model_dump() for t in body.history]

    query_embedding = rag.embed_query(body.question)
    chunks_with_titles = rag.retrieve_chunks_across_documents(
        db, current_user.id, query_embedding, top_k=8
    )
    messages = rag.build_multi_doc_messages(chunks_with_titles, body.question, history_dicts)

    # bbox is None for any chunk ingested before bounding-box extraction
    # existed, or where a page had no words at all — the frontend falls
    # back to a page-only jump in that case, same as single-doc chat.
    citations = [
        {
            "chunk_id": str(c.chunk_id),
            "doc_id": str(c.doc_id),
            "doc_title": title,
            "page_num": c.page_num,
            "bbox": c.bbox_json,
        }
        for c, title in chunks_with_titles
    ]

    db.close()

    async def event_stream():
        yield _sse("citations", {"citations": citations})

        full_text_parts: list[str] = []
        async for token in rag.stream_answer(messages):
            full_text_parts.append(token)
            yield _sse("token", {"text": token})

        # Persist the assistant's turn now that the full text is known.
        # Uses a fresh short-lived session rather than holding the
        # request-scoped one open for the whole stream duration.
        save_db = SessionLocal()
        try:
            save_db.add(
                ChatMessage(
                    doc_id=None,
                    user_id=current_user.id,
                    role="assistant",
                    content="".join(full_text_parts),
                    citations_json=citations,
                )
            )
            save_db.commit()
        finally:
            save_db.close()

        yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if present
        },
    )


@router.get("/history")
def get_multi_doc_history(current_user: User = Depends(get_current_user)):
    """
    Replays the multi-document chat thread in order, scoped by
    user_id only (doc_id IS NULL identifies these rows), so the
    frontend can hydrate an "ask across all documents" panel instead
    of starting blank on reload — same idea as chat.get_chat_history.
    """
    db = SessionLocal()
    try:
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.doc_id.is_(None), ChatMessage.user_id == current_user.id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "citations": m.citations_json,
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ]
        }
    finally:
        db.close()