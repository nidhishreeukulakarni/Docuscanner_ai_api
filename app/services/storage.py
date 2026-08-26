"""
Pluggable storage for original uploaded files (item 5 / SRS FR-01.4).

Two backends implement the same three-method interface, so nothing
that calls into this module ever needs to know or care which one is
active:

  - LocalStorage: writes to disk under storage/originals/. Active by
    default, needs no setup — this is what a fresh clone of the repo
    runs on out of the box.
  - S3Storage: real S3, gated behind STORAGE_BACKEND=s3 in .env plus
    real AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/S3_BUCKET values.
    boto3 is already an installed dependency (per the project's own
    notes) but was unused until now.

Swapping backends is a one-line .env change (STORAGE_BACKEND=local|s3)
— no code in documents.py, ingestion.py, or anywhere else needs to
change, because both backends return the same "key" string shape from
save() and the same (bytes, content_type) shape from load().
"""

from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from typing import Optional

from app.config import settings, STORAGE_ROOT


class StorageBackend:
    def save(self, file_bytes: bytes, filename: str, doc_id: uuid.UUID) -> str:
        """Persists the file, returns an opaque "key" string stored in
        Document.file_url. Meaning of the key is backend-specific — a
        relative disk path for LocalStorage, an S3 object key for
        S3Storage — callers should never parse it themselves."""
        raise NotImplementedError

    def load(self, key: str) -> tuple[bytes, str]:
        """Returns (file_bytes, content_type). Only ever called when
        redirect_url() returned None — i.e. this backend wants the
        FastAPI app to proxy the bytes itself rather than redirect."""
        raise NotImplementedError

    def redirect_url(self, key: str) -> Optional[str]:
        """A presigned/public URL the client can be redirected to
        instead of the backend proxying bytes through itself. Returning
        None (the default) means "call load() and stream it yourself"
        — that's what local storage does, since a local disk path has
        no meaningful URL."""
        return None


class LocalStorage(StorageBackend):
    """Today's default. Behavior is unchanged from before this file
    existed — just moved out of ingestion.py so it sits behind the
    same interface as S3Storage."""

    def save(self, file_bytes: bytes, filename: str, doc_id: uuid.UUID) -> str:
        originals_dir = STORAGE_ROOT / "originals"
        originals_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(filename).suffix or ".bin"
        disk_filename = f"{doc_id}{ext}"
        dest_path = originals_dir / disk_filename

        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        return f"originals/{disk_filename}"

    def load(self, key: str) -> tuple[bytes, str]:
        path = STORAGE_ROOT / key
        if not path.exists():
            raise FileNotFoundError(f"No stored file at key: {key}")
        content_type, _ = mimetypes.guess_type(str(path))
        return path.read_bytes(), content_type or "application/octet-stream"


class S3Storage(StorageBackend):
    """Dormant until STORAGE_BACKEND=s3 and real credentials are set.
    boto3 is imported lazily inside __init__ (not at module load) so
    importing this file never requires boto3 to be configured — only
    actually selecting this backend does."""

    def __init__(self):
        import boto3  # local import — see class docstring

        self._client = boto3.client(
            "s3",
            region_name=settings.aws_region or None,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
        )
        self._bucket = settings.s3_bucket

    def save(self, file_bytes: bytes, filename: str, doc_id: uuid.UUID) -> str:
        ext = Path(filename).suffix or ".bin"
        key = f"originals/{doc_id}{ext}"
        content_type, _ = mimetypes.guess_type(filename)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=file_bytes,
            ContentType=content_type or "application/octet-stream",
        )
        return key

    def load(self, key: str) -> tuple[bytes, str]:
        obj = self._client.get_object(Bucket=self._bucket, Key=key)
        return obj["Body"].read(), obj.get("ContentType", "application/octet-stream")

    def redirect_url(self, key: str) -> Optional[str]:
        # Pre-signed, time-limited — matches the SRS's FR-01.4 wording
        # ("accessible only via pre-signed, time-limited tokens").
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=3600,
        )


_backend: Optional[StorageBackend] = None


def get_storage_backend() -> StorageBackend:
    """Lazily constructs and caches the configured backend. Lazy so
    that importing this module never touches boto3/AWS unless S3 is
    actually the selected backend."""
    global _backend
    if _backend is None:
        _backend = S3Storage() if settings.storage_backend == "s3" else LocalStorage()
    return _backend