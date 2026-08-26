import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.db import SessionLocal
from app.models import Document, User
from app.dependencies import get_current_user
from app.services.summarization import stream_summary

router = APIRouter()


@router.post("/{document_id}")
def summarize_document(
    document_id: str, current_user: User = Depends(get_current_user)
):
    """
    Streams SSE progress events, ending with a `complete` event containing
    the structured summary: {"overview": "...", "key_points": [...], "entities": [...]}.
    Mounted at /summary in main.py, so full path is POST /summary/{document_id}.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(400, "document_id must be a valid UUID")

    db = SessionLocal()
    doc = (
        db.query(Document)
        .filter(Document.id == doc_uuid, Document.owner_id == current_user.id)
        .first()
    )
    db.close()
    if doc is None:
        raise HTTPException(404, "document not found")

    return StreamingResponse(
        stream_summary(document_id),
        media_type="text/event-stream",
    )