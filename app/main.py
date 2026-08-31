import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, documents, chat, chat_all, summary, annotations

app = FastAPI(title="DocuSense AI API")

# Always allow local dev. Additionally allow a deployed frontend origin
# if FRONTEND_ORIGIN is set (e.g. in the backend's .env file for a real
# deploy). Comma-separate multiple origins if needed:
#   FRONTEND_ORIGIN=https://yourdomain.com,https://www.yourdomain.com
allow_origins = ["http://localhost:3000"]
extra_origins = os.getenv("FRONTEND_ORIGIN")
if extra_origins:
    allow_origins += [o.strip() for o in extra_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(summary.router, prefix="/summary", tags=["summary"])
app.include_router(annotations.router, prefix="/annotations", tags=["annotations"])
app.include_router(chat_all.router, prefix="/chat-all", tags=["chat-all"])


@app.get("/health")
def health():
    return {"status": "ok"}