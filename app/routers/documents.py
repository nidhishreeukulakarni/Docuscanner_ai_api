import mimetypes
import tempfile
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, HTTPException, Response
from fastapi.responses import RedirectResponse

from app.services.ingestion import (
    validate_file,
    store_original,
    load_original,
    get_original_redirect_url,
)
from app.services.ocr import extract_text_per_page, extract_text_from_image
from app.services.text_extract import extract_text_from_docx, extract_text_from_txt
from app.services.chunking import chunk_text, chunk_plain_text
from app.services.embeddings import embed_chunks
from app.models import Document, DocumentChunk, User
from app.db import SessionLocal
from app.dependencies import get_current_user

router = APIRouter()

# Real extension per allowed MIME type, used for the temp file handed
# to the extraction pipeline below. Previously this was hardcoded to
# ".pdf" for every upload regardless of real type, which crashed
# fitz.open() on anything else (item 5 bug fix).
_EXT_BY_TYPE = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "text/plain": ".txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}

# PDFs and directly-uploaded images both carry real per-word bounding
# boxes (native or OCR'd); docx/txt don't (see text_extract.py) — this
# set decides which chunker + which DocumentChunk.bbox_json behavior a
# given upload gets.
_BBOX_CAPABLE_TYPES = {"application/pdf", "image/png", "image/jpeg"}


def _extract_pages(file_path: str, content_type: str) -> list[dict]:
    if content_type == "application/pdf":
        return extract_text_per_page(file_path)
    if content_type in ("image/png", "image/jpeg"):
        return extract_text_from_image(file_path)
    if content_type == "text/plain":
        return extract_text_from_txt(file_path)
    if content_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return extract_text_from_docx(file_path)
    # validate_file() should have already rejected anything else, so
    # this only fires if ALLOWED_TYPES and this dispatch table drift
    # out of sync with each other.
    raise HTTPException(400, f"No extraction pipeline for {content_type}")


def _chunk_page(page: dict, content_type: str) -> list[dict]:
    if content_type in _BBOX_CAPABLE_TYPES:
        return chunk_text(page["words"], page["page_num"])
    return chunk_plain_text(page["text"], page["page_num"])


@router.post("/upload")
async def upload_document(
    file: UploadFile, current_user: User = Depends(get_current_user)
):
    content = await file.read()
    try:
        validate_file(file.filename, file.content_type, len(content))
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Real extension for the temp file, not a hardcoded ".pdf" — this
    # is the fix for the crash on non-PDF uploads.
    ext = _EXT_BY_TYPE.get(file.content_type) or Path(file.filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    doc_id = uuid.uuid4()
    key = store_original(content, file.filename, doc_id)

    try:
        pages = _extract_pages(tmp_path, file.content_type)
    finally:
        os.unlink(tmp_path)

    db = SessionLocal()
    try:
        doc = Document(
            id=doc_id,
            owner_id=current_user.id,
            title=file.filename,
            file_url=key,
            mime_type=file.content_type,
            page_count=len(pages),
            status="processing",
        )
        db.add(doc)
        db.commit()

        all_chunks = []
        for page in pages:
            all_chunks.extend(_chunk_page(page, file.content_type))

        if all_chunks:
            vectors = embed_chunks([c["content"] for c in all_chunks])
            for chunk, vector in zip(all_chunks, vectors):
                db.add(DocumentChunk(
                    chunk_id=uuid.uuid4(),
                    doc_id=doc.id,
                    page_num=chunk["page_num"],
                    content=chunk["content"],
                    bbox_json=chunk["bbox"],
                    embedding_vector=vector,
                ))

        doc.status = "ready"
        db.commit()
    finally:
        db.close()

    return {
        "document_id": str(doc_id),
        "file_key": key,
        "page_count": len(pages),
        "chunk_count": len(all_chunks),
        "status": "ready",
    }


def _get_owned_document(db, document_id: str, current_user: User) -> Document:
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(400, "document_id must be a valid UUID")

    doc = (
        db.query(Document)
        .filter(Document.id == doc_uuid, Document.owner_id == current_user.id)
        .first()
    )
    if doc is None:
        # 404, not 403 — matches the pattern already used in chat.py
        # and annotations.py: don't reveal that a document with this
        # id exists but belongs to someone else.
        raise HTTPException(404, "document not found")
    return doc


@router.get("")
def list_documents(current_user: User = Depends(get_current_user)):
    """Item 4: lets the frontend show a document list/dashboard
    instead of only ever being able to work with whatever was just
    uploaded in the current browser session."""
    db = SessionLocal()
    try:
        docs = (
            db.query(Document)
            .filter(Document.owner_id == current_user.id)
            .order_by(Document.created_at.desc())
            .all()
        )
        return [
            {
                "document_id": str(d.id),
                "title": d.title,
                "mime_type": d.mime_type,
                "page_count": d.page_count,
                "status": d.status,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ]
    finally:
        db.close()


@router.get("/{document_id}/file")
def get_document_file(
    document_id: str, current_user: User = Depends(get_current_user)
):
    """Item 4: lets the viewer re-fetch the original file after a
    refresh, instead of only ever working from the in-memory bytes
    captured at upload time. Redirects to a presigned URL when S3 is
    the active storage backend; streams bytes directly for local disk.
    """
    db = SessionLocal()
    try:
        doc = _get_owned_document(db, document_id, current_user)
        file_url, mime_type = doc.file_url, doc.mime_type
    finally:
        db.close()

    redirect = get_original_redirect_url(file_url)
    if redirect:
        return RedirectResponse(redirect)

    try:
        file_bytes, guessed_type = load_original(file_url)
    except FileNotFoundError:
        raise HTTPException(404, "original file not found in storage")

    return Response(
        content=file_bytes,
        media_type=mime_type or guessed_type or "application/octet-stream",
    )