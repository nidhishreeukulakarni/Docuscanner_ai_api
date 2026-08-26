import uuid
from pathlib import Path
from app.services.storage import get_storage_backend

ALLOWED_TYPES = {"application/pdf", "image/png", "image/jpeg",
                  "text/plain",
                  "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
MAX_BYTES = 50 * 1024 * 1024  # 50MB

def validate_file(filename: str, content_type: str, size: int):
    if content_type not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported file type: {content_type}")
    if size > MAX_BYTES:
        raise ValueError("File exceeds 50MB limit")

def store_original(file_bytes: bytes, filename: str, doc_id: uuid.UUID) -> str:
    """Persists the original uploaded file via whichever StorageBackend
    is configured (app/services/storage.py — local disk by default,
    S3 once STORAGE_BACKEND=s3 and real credentials are set).

    Returns an opaque "key" string, stored in Document.file_url, whose
    meaning depends on the active backend. Callers should treat it as
    opaque and pass it back into load_original()/get_original_redirect_url()
    rather than parsing it themselves.
    """
    return get_storage_backend().save(file_bytes, filename, doc_id)


def load_original(key: str) -> tuple[bytes, str]:
    """Fetches the original file's bytes + content type. Used by
    GET /documents/{id}/file when the active backend has no
    redirect_url() (i.e. local storage)."""
    return get_storage_backend().load(key)


def get_original_redirect_url(key: str) -> str | None:
    """Returns a presigned/public URL to redirect the client to
    instead of proxying bytes, or None if the active backend wants to
    be streamed directly (local storage always returns None here)."""
    return get_storage_backend().redirect_url(key)