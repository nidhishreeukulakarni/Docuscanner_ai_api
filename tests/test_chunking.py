from app.services.chunking import chunk_text


def _word(text, x0, y0, x1, y1):
    return {"text": text, "bbox": [x0, y0, x1, y1]}


def test_chunk_text_empty_words_returns_no_chunks():
    assert chunk_text([], page_num=1) == []


def test_chunk_text_short_input_produces_single_chunk():
    words = [_word(w, 0.1, 0.1, 0.2, 0.2) for w in ["hello", "world"]]
    chunks = chunk_text(words, page_num=3, chunk_size=800, overlap=80)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "hello world"
    assert chunks[0]["page_num"] == 3


def test_chunk_text_bbox_is_union_of_word_boxes():
    words = [
        _word("a", 0.10, 0.20, 0.15, 0.25),
        _word("b", 0.30, 0.05, 0.35, 0.10),
        _word("c", 0.05, 0.40, 0.09, 0.45),
    ]
    chunks = chunk_text(words, page_num=1)
    bbox = chunks[0]["bbox"]
    assert bbox["x0"] == 0.05
    assert bbox["y0"] == 0.05
    assert bbox["x1"] == 0.35
    assert bbox["y1"] == 0.45


def test_chunk_text_splits_long_input_with_overlap():
    words = [_word(f"w{i}", 0, 0, 0.01, 0.01) for i in range(1000)]
    chunks = chunk_text(words, page_num=1, chunk_size=800, overlap=80)
    # step = 800 - 80 = 720, so the second chunk starts at word 720
    assert len(chunks) == 2
    assert chunks[0]["content"].startswith("w0 w1 w2")
    assert chunks[1]["content"].startswith("w720 w721")