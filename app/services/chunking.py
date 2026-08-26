def chunk_text(words: list[dict], page_num: int, chunk_size=800, overlap=80):
    """Groups a page's words into overlapping chunks (same windowing as
    before: chunk_size words per chunk, overlap words shared between
    consecutive chunks), but now each word carries its own normalized
    bounding box, so we can also compute a bounding box for the chunk
    as a whole: the union (min/max) of every word box inside it.

    NOTE on the union-box approach: this draws one rectangle spanning
    from the chunk's first word to its last. For a normal single-column
    page that rectangle hugs the actual paragraph. For a multi-column
    layout it can end up wider than intended (e.g. spanning across a
    column gap) since "first word" and "last word" may not be visually
    adjacent. Fine for a portfolio demo on typical single-column PDFs;
    a more robust version would keep one bbox per line and highlight
    each line separately instead of one rectangle for the whole chunk.
    """
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            xs0 = [w["bbox"][0] for w in chunk_words]
            ys0 = [w["bbox"][1] for w in chunk_words]
            xs1 = [w["bbox"][2] for w in chunk_words]
            ys1 = [w["bbox"][3] for w in chunk_words]
            bbox = {
                "x0": min(xs0), "y0": min(ys0),
                "x1": max(xs1), "y1": max(ys1),
            }
        else:
            bbox = None

        chunks.append({
            "page_num": page_num,
            "content": " ".join(w["text"] for w in chunk_words),
            "bbox": bbox,
        })
        i += chunk_size - overlap
    return chunks


def chunk_plain_text(text: str, page_num: int, chunk_size=800, overlap=80):
    """Same windowing as chunk_text() above, for sources with no
    per-word bounding boxes (docx/txt — see text_extract.py). Every
    chunk's bbox is always None; citation-click-to-highlight has
    nothing to draw for these, but chat/summary/search work the same."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunks.append({
            "page_num": page_num,
            "content": " ".join(chunk_words),
            "bbox": None,
        })
        i += chunk_size - overlap
    return chunks