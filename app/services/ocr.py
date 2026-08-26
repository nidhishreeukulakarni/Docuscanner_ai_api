import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image
import io

# Bounding boxes are stored as FRACTIONS of the page's own width/height
# (0.0-1.0), not raw points or pixels. That way a box means the same
# thing regardless of whether it came from a native-text page (measured
# in PDF points) or an OCR'd page (measured in rasterized image
# pixels), and the frontend can place it correctly at any zoom level
# just by multiplying against whatever size it's currently rendering
# the page at. Origin is top-left, y grows downward (matches both
# PyMuPDF's word coordinates and normal screen/image conventions).


def extract_text_per_page(file_path: str) -> list[dict]:
    """Returns, for each page: page_num, the full text (used by chat/
    summary as before), and a flat list of words with normalized
    bounding boxes (used to compute a bbox per chunk in chunking.py).
    """
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        density = len(text.strip()) / max(page.rect.width * page.rect.height, 1)

        if density < 0.0002:
            # Scanned/image page - fall back to OCR for both the text
            # and the word boxes (pytesseract gives us both in one pass).
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text, words = _ocr_words(img)
        else:
            words = _native_words(page)

        pages.append({"page_num": i + 1, "text": text, "words": words})
    return pages


def extract_text_from_image(file_path: str) -> list[dict]:
    """Single-page OCR path for a directly-uploaded image (png/jpg) —
    as opposed to a scanned page found inside a PDF. Reuses the same
    _ocr_words() Tesseract pass and the same 0-1 normalized bbox
    convention as the PDF path, so citations/highlighting work
    identically once this goes through chunk_text() in chunking.py.
    """
    img = Image.open(file_path)
    text, words = _ocr_words(img)
    return [{"page_num": 1, "text": text, "words": words}]


def _native_words(page: "fitz.Page") -> list[dict]:
    """Word boxes for a page that already has a text layer (not scanned).
    page.get_text("words") returns (x0, y0, x1, y1, word, block, line, no)
    in PDF points, top-left origin - we just normalize by page size.
    """
    page_w, page_h = page.rect.width, page.rect.height
    words = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        if not word.strip():
            continue
        words.append({
            "text": word,
            "bbox": [x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h],
        })
    return words


def _ocr_words(img: "Image.Image") -> tuple[str, list[dict]]:
    """Word boxes for a scanned page via Tesseract. image_to_data gives
    per-word pixel boxes (left, top, width, height) on the rasterized
    image - normalize by that image's own dimensions, which corresponds
    1:1 to the full page since we rasterized the whole page.
    """
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    img_w, img_h = img.size
    words = []
    texts = []
    for j, word in enumerate(data["text"]):
        if not word.strip():
            continue
        texts.append(word)
        left, top = data["left"][j], data["top"][j]
        w, h = data["width"][j], data["height"][j]
        words.append({
            "text": word,
            "bbox": [
                left / img_w,
                top / img_h,
                (left + w) / img_w,
                (top + h) / img_h,
            ],
        })
    return " ".join(texts), words