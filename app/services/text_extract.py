"""
Non-PDF, non-image text extraction (item 5: docx/txt ingestion).

Unlike ocr.py's paths, these formats carry no per-word bounding boxes —
python-docx has no concept of a laid-out "page" (that only exists once
a real rendering/pagination engine processes the file, which this
project doesn't do), and a .txt file was never paginated at all. So
every chunk built from these sources gets bbox_json = None in
documents.py, and citation-click-to-highlight (which needs a bbox to
draw a rectangle) simply won't draw one for them — there's no page
image to draw on top of in the first place for these formats. Chat,
summary, and semantic search all still work fully; only the "jump to
the exact spot on the page" visual doesn't apply here.
"""

from docx import Document as DocxDocument


def extract_text_from_docx(file_path: str) -> list[dict]:
    """The whole document is treated as a single logical page (page_num
    = 1) since python-docx exposes paragraphs, not rendered pages."""
    doc = DocxDocument(file_path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"page_num": 1, "text": text, "words": []}]


def extract_text_from_txt(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [{"page_num": 1, "text": text, "words": []}]