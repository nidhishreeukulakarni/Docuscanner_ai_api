import uuid

from app.models import Document, DocumentChunk, User
from app.services.auth import hash_password
from app.services.rag import build_messages, build_scoped_messages, retrieve_chunks


class _FakeChunk:
    def __init__(self, page_num, content):
        self.page_num = page_num
        self.content = content


def test_build_messages_includes_excerpts_and_question():
    chunks = [_FakeChunk(2, "The termination clause requires 30 days notice.")]
    messages = build_messages(chunks, "What is the notice period?")
    user_msg = messages[-1]["content"]
    assert "p. 2" in user_msg
    assert "30 days notice" in user_msg
    assert "What is the notice period?" in user_msg


def test_build_messages_with_no_chunks_uses_fallback_text():
    messages = build_messages([], "Anything in here?")
    assert "No excerpts were retrieved" in messages[-1]["content"]


def test_build_messages_only_keeps_last_six_history_turns():
    history = [{"role": "user", "content": f"turn {i}"} for i in range(10)]
    messages = build_messages([], "final question", history=history)
    # 1 system + last 6 history turns + 1 final user message = 8
    assert len(messages) == 8
    assert messages[1]["content"] == "turn 4"


def test_build_scoped_messages_includes_page_number():
    messages = build_scoped_messages(
        highlighted_text="Employee shall not disclose confidential information.",
        highlighted_page=5,
        question="What does this mean?",
    )
    assert "page 5" in messages[-1]["content"]
    assert "confidential information" in messages[-1]["content"]


def test_retrieve_chunks_orders_by_cosine_distance(db_session):
    user = User(
        id=uuid.uuid4(),
        email="ragtest@example.com",
        hashed_password=hash_password("whatever123"),
    )
    db_session.add(user)
    db_session.flush()

    doc = Document(id=uuid.uuid4(), owner_id=user.id, title="Test doc", status="ready")
    db_session.add(doc)
    db_session.flush()

    query_vec = [1.0] + [0.0] * 1023

    close = DocumentChunk(
        doc_id=doc.id, page_num=1, content="close match",
        embedding_vector=[0.9] + [0.0] * 1023,
    )
    far = DocumentChunk(
        doc_id=doc.id, page_num=2, content="far match",
        embedding_vector=[-1.0] + [0.0] * 1023,
    )
    db_session.add_all([close, far])
    db_session.flush()

    results = retrieve_chunks(db_session, doc.id, query_vec, top_k=2)

    assert [r.content for r in results] == ["close match", "far match"]
    