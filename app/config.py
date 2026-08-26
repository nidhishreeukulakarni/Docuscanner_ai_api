from pydantic_settings import BaseSettings
from pathlib import Path

# Project root = docuscanner_ai_api/ (two levels up from this file:
# app/config.py -> app/ -> docuscanner_ai_api/). Used to resolve
# storage_dir to an absolute path regardless of the working directory
# uvicorn was launched from.
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/docusense"
    redis_url: str = "redis://localhost:6379/0"
    s3_bucket: str = "docusense-dev"
    # Which StorageBackend (app/services/storage.py) is active. "local"
    # (default) writes to disk under storage_dir below and needs no
    # setup. Flip to "s3" once real AWS credentials exist below —
    # nothing else in the codebase has to change, since both backends
    # implement the same save/load/redirect_url interface.
    storage_backend: str = "local"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    # Local disk storage for original uploaded files, used until real
    # S3 credentials are wired up. Relative paths are resolved against
    # BASE_DIR so it works the same whether uvicorn is launched from
    # the project root or elsewhere.
    storage_dir: str = "storage"
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_api_key: str = ""
    jwt_secret: str = "change-me-in-prod"
    xai_api_key: str = ""
    google_api_key: str = ""

    class Config:
        env_file = ".env"

settings = Settings()

# Absolute path to the storage root, e.g. C:\dev\docuscanner_ai_api\storage
STORAGE_ROOT = (BASE_DIR / settings.storage_dir).resolve()
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)