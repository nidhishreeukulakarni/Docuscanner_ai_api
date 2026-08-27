"""
Shared pytest fixtures.

Tests run against the same Postgres database the app uses (see
DATABASE_URL / docker-compose.yml) rather than a separate test
database, to keep the Day 1 setup lightweight. Isolation from real
data is handled by wrapping each test in a transaction that's rolled
back afterward, so nothing written during a test run is ever actually
committed -- existing accounts/documents are untouched.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.dependencies import get_db
from app.main import app
from app.models import Base

engine = create_engine(settings.database_url)
TestingSessionLocal = sessionmaker(bind=engine)


@pytest.fixture(scope="session", autouse=True)
def _ensure_schema():
    """Make sure every table exists before the suite runs. Does not
    drop or alter anything -- safe to run against a DB that already
    has real data in it."""
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def db_session():
    """A session bound to a single connection + transaction that gets
    rolled back at the end of the test, so nothing persists."""
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db_session):
    """A FastAPI TestClient whose get_db dependency is overridden to
    use the rolled-back-per-test session above, so hitting real
    endpoints (register/login) never touches persistent data."""

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()