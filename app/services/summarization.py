"""
Step 7 — Document summarization.

Map-reduce over the same DocumentChunk rows created during upload (Step 4):
  1. MAP:    summarize the document in small batches of chunks
  2. REDUCE: combine those batch-summaries into one structured JSON summary

Uses Grok (xAI) via the OpenAI-compatible endpoint.
Streams progress + the final result as Server-Sent Events.
"""

from typing import Generator
import json

from openai import OpenAI

from app.config import settings
from app.db import SessionLocal
from app.models import DocumentChunk

client = OpenAI(
    api_key=settings.google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

GROK_MODEL = "gemini-3.6-flash" # keep the variable name, or rename to MODEL_NAME if you prefer

CHUNK_BATCH_SIZE = 8

MAP_PROMPT = """You are summarizing a portion of a longer document.
Summarize ONLY the key points, named entities, and section titles present
in this excerpt. Do not add commentary. Be terse — this summary will be
combined with summaries of other excerpts later.

Excerpt:
{text}
"""

REDUCE_PROMPT = """You are given several partial summaries of one document,
in order. Combine them into a single structured summary with exactly
these three sections:

1. Overview - 2-3 sentence high-level summary
2. Key Points - the most important points
3. Entities - people, organizations, dates, and key terms mentioned

Return ONLY valid JSON with this exact shape, nothing else, no markdown fences:
{{"overview": "...", "key_points": ["...", "..."], "entities": ["...", "..."]}}

Partial summaries:
{summaries}
"""


def get_document_chunks(doc_id: str) -> list[str]:
    db = SessionLocal()
    try:
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.doc_id == doc_id)
            .order_by(DocumentChunk.page_num.asc())
            .all()
        )
        return [c.content for c in chunks]
    finally:
        db.close()


def batch_chunks(chunks: list[str], batch_size: int) -> list[list[str]]:
    return [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]


def map_batch(batch: list[str]) -> str:
    text = "\n\n---\n\n".join(batch)
    response = client.chat.completions.create(
        model=GROK_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": MAP_PROMPT.format(text=text)}],
    )
    return response.choices[0].message.content


def stream_summary(doc_id: str) -> Generator[str, None, None]:
    chunks = get_document_chunks(doc_id)
    if not chunks:
        yield f"data: {json.dumps({'type': 'error', 'message': 'No chunks found for this document'})}\n\n"
        return

    batches = batch_chunks(chunks, CHUNK_BATCH_SIZE)
    yield f"data: {json.dumps({'type': 'status', 'message': f'Summarizing {len(batches)} section(s)...'})}\n\n"

    partial_summaries = []
    for i, batch in enumerate(batches):
        summary = map_batch(batch)
        partial_summaries.append(summary)
        yield f"data: {json.dumps({'type': 'progress', 'completed': i + 1, 'total': len(batches)})}\n\n"

    yield f"data: {json.dumps({'type': 'status', 'message': 'Combining into final summary...'})}\n\n"
    combined_text = "\n\n---\n\n".join(partial_summaries)
    response = client.chat.completions.create(
        model=GROK_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": REDUCE_PROMPT.format(summaries=combined_text)}],
    )

    try:
        final_summary = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        final_summary = {"overview": response.choices[0].message.content, "key_points": [], "entities": []}

    yield f"data: {json.dumps({'type': 'complete', 'summary': final_summary})}\n\n"