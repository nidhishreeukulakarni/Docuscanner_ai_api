from sqlalchemy import text
from app.db import engine
from app.models import Base

# Enable the pgvector extension first
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

# Create all tables defined in models.py
Base.metadata.create_all(engine)

print("Database initialized: pgvector extension enabled, tables created.")