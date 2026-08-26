"""
Step 9 — Persistent annotations (SRS 5: Annotations table).

Saves the passages a user acted on via the 4-pill toolbar so they
survive a refresh, instead of living only in frontend React state.
Mounted at /annotations in main.py.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.db import SessionLocal
from app.models import Annotation, Document, User
from app.dependencies import get_current_user

router = APIRouter()


class AnnotationCreate(BaseModel):
    page_num: int
    selected_text: str
    # Normalized {x0,y0,x1,y1} fraction bbox, same convention as
    # DocumentChunk.bbox_json. Optional — a raw text selection doesn't
    # always map cleanly to a single box.
    rect_coords: dict | None = None
    # Reused to store which preset pill produced this highlight
    # ("explain" | "summarize" | "risks") so the frontend can show the
    # right label when it reloads a saved highlight.
    ai_notes: str | None = None


class AnnotationOut(BaseModel):
    anno_id: str
    doc_id: str
    page_num: int
    selected_text: str
    rect_coords: dict | None
    ai_notes: str | None
    created_at: str


def _to_out(a: Annotation) -> AnnotationOut:
    return AnnotationOut(
        anno_id=str(a.anno_id),
        doc_id=str(a.doc_id),
        page_num=a.page_num,
        selected_text=a.selected_text,
        rect_coords=a.rect_coords,
        ai_notes=a.ai_notes,
        created_at=a.created_at.isoformat(),
    )


def _get_owned_document(db, document_id: str, current_user: User) -> uuid.UUID:
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
        # 404, not 403 — don't reveal that a document with this id
        # exists but belongs to someone else (matches chat.py's pattern).
        raise HTTPException(404, "document not found")
    return doc_uuid


@router.post("/{document_id}", response_model=AnnotationOut)
def create_annotation(
    document_id: str,
    body: AnnotationCreate,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        doc_uuid = _get_owned_document(db, document_id, current_user)

        anno = Annotation(
            doc_id=doc_uuid,
            user_id=current_user.id,
            page_num=body.page_num,
            rect_coords=body.rect_coords,
            selected_text=body.selected_text,
            ai_notes=body.ai_notes,
        )
        db.add(anno)
        db.commit()
        db.refresh(anno)
        return _to_out(anno)
    finally:
        db.close()


@router.get("/{document_id}", response_model=list[AnnotationOut])
def list_annotations(
    document_id: str, current_user: User = Depends(get_current_user)
):
    db = SessionLocal()
    try:
        doc_uuid = _get_owned_document(db, document_id, current_user)

        annos = (
            db.query(Annotation)
            .filter(Annotation.doc_id == doc_uuid, Annotation.user_id == current_user.id)
            .order_by(Annotation.created_at.desc())
            .all()
        )
        return [_to_out(a) for a in annos]
    finally:
        db.close()


@router.delete("/{document_id}/{anno_id}")
def delete_annotation(
    document_id: str,
    anno_id: str,
    current_user: User = Depends(get_current_user),
):
    db = SessionLocal()
    try:
        doc_uuid = _get_owned_document(db, document_id, current_user)
        try:
            anno_uuid = uuid.UUID(anno_id)
        except ValueError:
            raise HTTPException(400, "anno_id must be a valid UUID")

        anno = (
            db.query(Annotation)
            .filter(
                Annotation.anno_id == anno_uuid,
                Annotation.doc_id == doc_uuid,
                Annotation.user_id == current_user.id,
            )
            .first()
        )
        if anno is None:
            raise HTTPException(404, "annotation not found")

        db.delete(anno)
        db.commit()
        return {"status": "deleted"}
    finally:
        db.close()