from sqlalchemy import text
from app.db import engine

with engine.connect() as conn:
    conn.execute(text(
        "DROP TABLE IF EXISTS annotations, chat_messages, document_chunks, documents, users CASCADE;"
    ))
    conn.commit()

print("Old tables dropped.")