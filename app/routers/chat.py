import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.db import SessionLocal
from app.models import ChatMessage, Document, User
from app.dependencies import get_current_user
from app.services import rag

router = APIRouter()


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []
    highlighted_text: str | None = None
    highlighted_page: int | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/{document_id}")
async def chat_with_document(
    document_id: str,
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """
    SSE stream: embed the question, retrieve the document's nearest
    chunks via pgvector, then stream the Groq completion back token
    by token so the frontend can render it live.

    Step 9: also persists both turns to ChatMessage so a conversation
    survives a refresh — the user's turn is saved immediately (so it's
    not lost even if the stream fails partway), and the assistant's
    turn is saved once the full answer text is known, right before the
    "done" event.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(400, "document_id must be a valid UUID")

    if not body.question.strip():
        raise HTTPException(400, "question must not be empty")

    db = SessionLocal()
    doc = (
        db.query(Document)
        .filter(Document.id == doc_uuid, Document.owner_id == current_user.id)
        .first()
    )
    if doc is None:
        db.close()
        # 404 rather than 403 — don't reveal that a document with this id
        # exists but belongs to someone else.
        raise HTTPException(404, "document not found")
    if doc.status != "ready":
        db.close()
        raise HTTPException(409, f"document is not ready yet (status: {doc.status})")

    # Persist the user's turn right away, before any retrieval/streaming
    # work, so it's saved even if the LLM call fails.
    db.add(
        ChatMessage(
            doc_id=doc_uuid,
            user_id=current_user.id,
            role="user",
            content=body.question,
            citations_json=None,
        )
    )
    db.commit()

    history_dicts = [t.model_dump() for t in body.history]

    if body.highlighted_text and body.highlighted_text.strip():
        # Step 8: scoped-to-selection path — skip retrieval entirely and
        # answer from just the highlighted passage. There's no computed
        # bbox for an arbitrary user selection (bboxes only exist for
        # chunks from ingestion), so this citation carries page_num only
        # — the frontend already falls back to a page-only jump when
        # bbox is None.
        messages = rag.build_scoped_messages(
            body.highlighted_text, body.highlighted_page, body.question, history_dicts
        )
        citations = (
            [
                {
                    "chunk_id": f"highlight-p{body.highlighted_page}",
                    "page_num": body.highlighted_page,
                    "bbox": None,
                }
            ]
            if body.highlighted_page
            else []
        )
    else:
        query_embedding = rag.embed_query(body.question)
        chunks = rag.retrieve_chunks(db, doc_uuid, query_embedding, top_k=5)
        messages = rag.build_messages(chunks, body.question, history_dicts)
        # bbox is None for any chunk ingested before bounding-box
        # extraction existed, or where a page had no words at all - the
        # frontend falls back to a page-only jump in that case.
        citations = [
            {
                "chunk_id": str(c.chunk_id),
                "page_num": c.page_num,
                "bbox": c.bbox_json,
            }
            for c in chunks
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
                    doc_id=doc_uuid,
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


@router.get("/{document_id}/history")
def get_chat_history(
    document_id: str, current_user: User = Depends(get_current_user)
):
    """
    Step 9: replays a document's chat history in order, so the
    frontend can hydrate the Chat tab instead of starting blank on
    reload. Scoped per-user (not just per-document) since two users
    could theoretically share a document in the future.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(400, "document_id must be a valid UUID")

    db = SessionLocal()
    try:
        doc = (
            db.query(Document)
            .filter(Document.id == doc_uuid, Document.owner_id == current_user.id)
            .first()
        )
        if doc is None:
            raise HTTPException(404, "document not found")

        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.doc_id == doc_uuid, ChatMessage.user_id == current_user.id)
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