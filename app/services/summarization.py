"""
Step 7 — Document summarization.

Map-reduce over the same DocumentChunk rows created during upload (Step 4):
  1. MAP:    summarize the document in small batches of chunks
  2. REDUCE: combine those batch-summaries into one structured JSON summary

Uses Gemini via the OpenAI-compatible endpoint.
Streams progress + the final result as Server-Sent Events.
"""

from typing import Generator
import json
import re

from openai import OpenAI, RateLimitError

from app.config import settings
from app.db import SessionLocal
from app.models import DocumentChunk


class SummaryGenerationError(Exception):
    """Raised when the LLM call itself fails (rate limit, API error, etc.)
    so stream_summary can yield one clean SSE error event instead of the
    whole generator dying with an unhandled exception mid-stream."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def _call_model(**kwargs):
    """
    Fix: map_batch()/the reduce call previously called
    client.chat.completions.create() with no error handling. A 429
    RateLimitError (or any other API error) would propagate straight up
    through the generator and crash the SSE stream — the frontend's
    "Couldn't generate a summary" screen with no useful detail, and a
    raw traceback in the backend logs instead of a clean signal.

    Wrapping every call here means both map_batch() and stream_summary()
    raise SummaryGenerationError with a readable message on failure,
    which stream_summary can turn into a proper `error` SSE event.
    """
    try:
        return client.chat.completions.create(**kwargs)
    except RateLimitError as e:
        raise SummaryGenerationError(
            "Gemini API quota exceeded for today (free-tier limit reached). "
            "Try again later, or check your plan/billing at "
            "https://ai.google.dev/gemini-api/docs/rate-limits."
        ) from e
    except Exception as e:
        raise SummaryGenerationError(f"Summary generation failed: {e}") from e

client = OpenAI(
    api_key=settings.google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL_NAME = "gemini-3.6-flash"

# Fix: was 8. Most test/demo documents only produce 3-5 chunks (roughly
# one per page for typical page lengths — see chunking.py), which was
# always LESS than the old batch size of 8. That meant batch_chunks()
# put the entire document into a single batch, so map_batch() ran
# exactly once on the whole document's text, capped at the map step's
# max_tokens — reduce then had nothing to actually combine, just one
# oversized summary to reformat. Lowering this to 2 means even a short
# 3-4 chunk document produces 2+ real batches, so reduce is actually
# combining multiple partial summaries instead of reformatting one.
CHUNK_BATCH_SIZE = 2

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
    response = _call_model(
        model=MODEL_NAME,
        max_tokens=800,
        messages=[{"role": "user", "content": MAP_PROMPT.format(text=text)}],
    )
    return response.choices[0].message.content


def _parse_summary_json(raw: str) -> dict:
    """
    Fix: the model can (a) wrap its JSON in ```json ... ``` fences despite
    being told not to, or (b) run out of max_tokens mid-output and get
    cut off before the closing brace. Previously either case fell through
    to json.JSONDecodeError and the raw, possibly-truncated text —
    braces, quotes and all — was dumped straight into "overview" and
    shown to the user looking like broken JSON.

    This now: strips markdown fences if present, then tries to parse.
    If it still fails (e.g. genuinely truncated), falls back to a plain
    readable message instead of leaking the raw JSON fragment.
    """
    cleaned = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "overview": (
                "Summary generation didn't finish cleanly — try "
                "regenerating. (The document was processed, but the "
                "final formatting step was cut off.)"
            ),
            "key_points": [],
            "entities": [],
        }


def stream_summary(doc_id: str) -> Generator[str, None, None]:
    chunks = get_document_chunks(doc_id)
    if not chunks:
        yield f"data: {json.dumps({'type': 'error', 'message': 'No chunks found for this document'})}\n\n"
        return

    batches = batch_chunks(chunks, CHUNK_BATCH_SIZE)
    yield f"data: {json.dumps({'type': 'status', 'message': f'Summarizing {len(batches)} section(s)...'})}\n\n"

    try:
        partial_summaries = []
        for i, batch in enumerate(batches):
            summary = map_batch(batch)
            partial_summaries.append(summary)
            yield f"data: {json.dumps({'type': 'progress', 'completed': i + 1, 'total': len(batches)})}\n\n"

        yield f"data: {json.dumps({'type': 'status', 'message': 'Combining into final summary...'})}\n\n"
        combined_text = "\n\n---\n\n".join(partial_summaries)
        response = _call_model(
            model=MODEL_NAME,
            # Fix: was 1000. If gemini-3.6-flash spends internal reasoning
            # tokens out of the same budget before producing visible output,
            # 1000 could be exhausted before the JSON closes — which is what
            # produced the truncated '{"overview": "...cut off' output seen
            # in testing. Raised substantially for headroom.
            max_tokens=3000,
            messages=[{"role": "user", "content": REDUCE_PROMPT.format(summaries=combined_text)}],
        )
    except SummaryGenerationError as e:
        # Fix: previously this exception propagated all the way up
        # through the SSE generator and crashed the stream with an
        # unhandled traceback (the 429 case seen in testing) — the
        # frontend just showed a generic "Couldn't generate a summary"
        # with no detail. Now it's a clean, readable error event.
        yield f"data: {json.dumps({'type': 'error', 'message': e.message})}\n\n"
        return

    final_summary = _parse_summary_json(response.choices[0].message.content)
    yield f"data: {json.dumps({'type': 'complete', 'summary': final_summary})}\n\n"