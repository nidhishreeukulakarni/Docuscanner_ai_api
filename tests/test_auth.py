def test_register_creates_new_user(client):
    resp = client.post("/auth/register", json={
        "email": "newuser@example.com",
        "password": "supersecret123",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "newuser@example.com"
    assert body["access_token"]


def test_register_duplicate_email_returns_409(client):
    payload = {"email": "dupe@example.com", "password": "supersecret123"}
    first = client.post("/auth/register", json=payload)
    assert first.status_code == 200
    second = client.post("/auth/register", json=payload)
    assert second.status_code == 409


def test_register_short_password_returns_400(client):
    resp = client.post("/auth/register", json={
        "email": "shortpw@example.com",
        "password": "short",
    })
    assert resp.status_code == 400


def test_login_with_correct_password_succeeds(client):
    client.post("/auth/register", json={
        "email": "logintest@example.com",
        "password": "correcthorse123",
    })
    resp = client.post("/auth/login", json={
        "email": "logintest@example.com",
        "password": "correcthorse123",
    })
    assert resp.status_code == 200
    assert resp.json()["access_token"]


def test_login_with_wrong_password_returns_401(client):
    client.post("/auth/register", json={
        "email": "wrongpw@example.com",
        "password": "correcthorse123",
    })
    resp = client.post("/auth/login", json={
        "email": "wrongpw@example.com",
        "password": "not-the-password",
    })
    assert resp.status_code == 401


def test_login_updates_last_login_and_login_count(client, db_session):
    from app.models import User

    client.post("/auth/register", json={
        "email": "counter@example.com",
        "password": "correcthorse123",
    })

    user = db_session.query(User).filter(User.email == "counter@example.com").first()
    assert user.login_count == 0
    assert user.last_login is None

    client.post("/auth/login", json={
        "email": "counter@example.com",
        "password": "correcthorse123",
    })
    db_session.refresh(user)
    assert user.login_count == 1
    assert user.last_login is not None

    client.post("/auth/login", json={
        "email": "counter@example.com",
        "password": "correcthorse123",
    })
    db_session.refresh(user)
    assert user.login_count == 2