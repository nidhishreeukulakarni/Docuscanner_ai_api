from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector
import uuid, datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0, nullable=False)

class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String)
    file_url = Column(String)
    mime_type = Column(String)
    page_count = Column(Integer)
    status = Column(String, default="processing")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    page_num = Column(Integer)
    bbox_json = Column(JSONB)
    content = Column(Text)
    embedding_vector = Column(Vector(1024))

class ChatMessage(Base):
    """One turn of a chat thread (SRS 5: Chat_Messages). Two rows per
    exchange — one role="user", one role="assistant" — so a thread can
    be replayed in order via ORDER BY created_at. citations_json mirrors
    the same {chunk_id, page_num, bbox} shape chat.py already streams
    to the frontend, so history + live chat render identically.

    doc_id is nullable: a normal single-document chat turn has doc_id
    set to that document's id, but a multi-document ("ask across all my
    documents") turn has doc_id = NULL and is scoped by user_id only —
    see routers/chat_all.py."""
    __tablename__ = "chat_messages"
    msg_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    citations_json = Column(JSONB)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

class Annotation(Base):
    """A saved highlight (SRS 5: Annotations). rect_coords reuses the
    same normalized {x0,y0,x1,y1} fraction-of-page-size bbox shape as
    DocumentChunk.bbox_json — null when the user's raw text selection
    couldn't be mapped to a box. ai_notes optionally holds an AI answer
    the user chose to attach to this highlight."""
    __tablename__ = "annotations"
    anno_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    page_num = Column(Integer, nullable=False)
    rect_coords = Column(JSONB)
    selected_text = Column(Text, nullable=False)
    ai_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)