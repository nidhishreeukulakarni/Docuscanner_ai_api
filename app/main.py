from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, documents, chat, summary, annotations
from app.routers import auth, documents, chat, chat_all, summary, annotations

app = FastAPI(title="DocuSense AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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