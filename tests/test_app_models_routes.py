"""
Unit tests for the Flask application (models, routes) in offline mode.
No Ganache, no IPFS, no database server required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app import create_app, db
from app.models import User


@pytest.fixture()
def app():
    """Create an in-memory Flask app for testing."""
    _app = create_app()
    _app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY": "test-secret",
        "LOGIN_DISABLED": False,
    })
    with _app.app_context():
        db.create_all()
        yield _app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestUserModel:
    def test_password_hashing(self, app):
        """Password should be hashed and verify correctly."""
        with app.app_context():
            u = User(username="alice", email="alice@example.com", role="patient")
            u.set_password("secret123")
            assert u.check_password("secret123"), "Correct password must verify"
            assert not u.check_password("wrong"), "Wrong password must not verify"

    def test_password_not_stored_plaintext(self, app):
        """The raw password must not appear in password_hash."""
        with app.app_context():
            u = User(username="bob", email="bob@example.com", role="doctor")
            u.set_password("mypassword")
            assert "mypassword" not in u.password_hash

    def test_did_generated_on_register(self, app, client):
        """A freshly registered patient user must have a DID in did:ethr: format."""
        with app.app_context():
            resp = client.post("/auth/register", data={
                "username": "carol",
                "email": "carol@example.com",
                "password": "Test1234!",
                "role": "patient",
                "first_name": "Carol",
                "last_name": "Smith",
                "date_of_birth": "1990-01-01",
                "gender": "female",
                "phone": "0000000000",
                "address": "123 Main St",
                "emergency_contact": "0000000001",
            }, follow_redirects=True)
            assert resp.status_code == 200
            user = User.query.filter_by(email="carol@example.com").first()
            assert user is not None, "User must be created in DB"
            assert user.did is not None, "DID must be set at registration"
            assert user.did.startswith("did:ethr:0x"), \
                f"DID must start with did:ethr:0x, got: {user.did}"

    def test_did_uniqueness(self, app):
        """Two users must not share the same DID."""
        import secrets
        with app.app_context():
            u1 = User(username="u1", email="u1@example.com", role="patient",
                      did="did:ethr:0x" + secrets.token_hex(20))
            u1.set_password("pass1")
            u2 = User(username="u2", email="u2@example.com", role="patient",
                      did="did:ethr:0x" + secrets.token_hex(20))
            u2.set_password("pass2")
            db.session.add_all([u1, u2])
            db.session.commit()
            assert u1.did != u2.did, "DIDs must be unique"


# ---------------------------------------------------------------------------
# Auth route tests
# ---------------------------------------------------------------------------

class TestAuthRoutes:
    def test_login_page_loads(self, client):
        resp = client.get("/auth/login")
        assert resp.status_code == 200

    def test_register_page_loads(self, client):
        resp = client.get("/auth/register")
        assert resp.status_code == 200

    def test_invalid_login_rejected(self, client):
        resp = client.post("/auth/login", data={
            "email": "nobody@example.com",
            "password": "wrong",
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"Invalid" in resp.data or b"error" in resp.data.lower()

    def test_duplicate_email_rejected(self, client, app):
        """Registering with an existing email must fail gracefully."""
        with app.app_context():
            u = User(username="existing", email="dup@example.com", role="patient",
                     did="did:ethr:0xAABBCCDDEEFF00112233445566778899AABBCCDD")
            u.set_password("pass")
            db.session.add(u)
            db.session.commit()

        resp = client.post("/auth/register", data={
            "username": "newuser",
            "email": "dup@example.com",
            "password": "Test1234!",
            "role": "patient",
            "first_name": "X", "last_name": "Y",
            "date_of_birth": "2000-01-01",
            "gender": "other", "phone": "0", "address": ".", "emergency_contact": "0",
        }, follow_redirects=True)
        assert resp.status_code == 200
        # Should flash an error message
        assert b"already" in resp.data.lower() or b"error" in resp.data.lower()


# ---------------------------------------------------------------------------
# GDPR consent route (offline — blockchain best-effort, will gracefully fail)
# ---------------------------------------------------------------------------

class TestConsentRoutes:
    def _login_as_patient(self, client, app):
        """Helper: create a patient user and log in."""
        with app.app_context():
            u = User(username="patient1", email="p1@example.com", role="patient",
                     did="did:ethr:0x" + "ab" * 20)
            u.set_password("pass")
            db.session.add(u)
            db.session.commit()
        client.post("/auth/login", data={"email": "p1@example.com", "password": "pass"})

    def test_consent_grant_requires_login(self, client):
        resp = client.post("/patient/consent/grant",
                           json={"recipient": "0x1234", "data_type": "lab", "purpose": "x"},
                           follow_redirects=False)
        # Unauthenticated → redirect to login
        assert resp.status_code in (302, 401)

    def test_consent_revoke_requires_login(self, client):
        resp = client.post("/patient/consent/revoke",
                           json={"consent_id": 1},
                           follow_redirects=False)
        assert resp.status_code in (302, 401)

    def test_consent_grant_missing_fields(self, client, app):
        """Missing fields must return 400."""
        self._login_as_patient(client, app)
        resp = client.post("/patient/consent/grant",
                           json={"recipient": "0x1234"})
        assert resp.status_code == 400

    def test_consent_grant_returns_json(self, client, app):
        """A valid consent grant request must return JSON with status ok."""
        self._login_as_patient(client, app)
        resp = client.post("/patient/consent/grant", json={
            "recipient": "0xABCDEF1234567890ABCDEF1234567890ABCDEF12",
            "data_type": "lab_report",
            "purpose": "treatment",
        })
        assert resp.content_type.startswith("application/json")
        payload = resp.get_json()
        assert payload["status"] == "ok"
        assert "purpose_hash" in payload
