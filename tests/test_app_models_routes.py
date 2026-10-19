"""
Flask application tests (models, auth, consent) in offline mode.
No Ganache, no IPFS, no database server required: the `app` / `client` /
`make` fixtures live in conftest.py (in-memory SQLite, ephemeral keys).
"""
import secrets

from app import db
from app.models import Consent, Doctor, User
from conftest import PASSWORD


def _patient_form(**over):
    data = {
        "username": "carol", "email": "carol@example.com", "password": PASSWORD,
        "role": "patient", "first_name": "Carol", "last_name": "Smith",
        "date_of_birth": "1990-01-01", "gender": "female", "phone": "0000000000",
        "address": "123 Main St", "emergency_contact": "0000000001",
    }
    data.update(over)
    return data


class TestUserModel:
    def test_password_hashing(self, app):
        u = User(username="alice", email="alice@example.com", role="patient")
        u.set_password("secret-123456")
        assert u.check_password("secret-123456")
        assert not u.check_password("wrong")

    def test_password_not_stored_plaintext(self, app):
        u = User(username="bob", email="bob@example.com", role="doctor")
        u.set_password("mypassword-123")
        assert "mypassword-123" not in u.password_hash

    def test_password_hash_column_fits_modern_hashes(self, app):
        """A-10: scrypt hashes are ~162 chars; the column used to be 120."""
        assert User.__table__.c.password_hash.type.length >= 255

    def test_sentinel_hash_never_verifies(self, app):
        u = User(username="x", email="x@example.com", role="patient", password_hash="[ERASED]")
        assert not u.check_password("[ERASED]")
        assert not u.check_password("")

    def test_did_generated_on_register(self, app, client):
        resp = client.post("/auth/register", data=_patient_form(), follow_redirects=True)
        assert resp.status_code == 200
        user = User.query.filter_by(email="carol@example.com").first()
        assert user is not None
        assert user.did.startswith("did:ethr:0x")

    def test_did_uniqueness(self, app):
        u1 = User(username="u1", email="u1@example.com", role="patient",
                  did="did:ethr:0x" + secrets.token_hex(20))
        u2 = User(username="u2", email="u2@example.com", role="patient",
                  did="did:ethr:0x" + secrets.token_hex(20))
        u1.set_password("pass-123456")
        u2.set_password("pass-123456")
        db.session.add_all([u1, u2])
        db.session.commit()
        assert u1.did != u2.did


class TestAuthRoutes:
    def test_pages_load(self, client):
        assert client.get("/auth/login").status_code == 200
        assert client.get("/auth/register").status_code == 200
        assert client.get("/").status_code == 200

    def test_invalid_login_rejected(self, client):
        resp = client.post("/auth/login", data={"email": "nobody@example.com", "password": "wrong"},
                           follow_redirects=True)
        assert resp.status_code == 200
        assert b"Invalid email or password" in resp.data

    def test_duplicate_email_rejected(self, client, make):
        make.user("patient", "existing")
        resp = client.post("/auth/register",
                           data=_patient_form(username="newuser", email="existing@example.com"),
                           follow_redirects=True)
        # F-04: unified message — no longer leaks whether email or username clashed
        assert b"registration failed" in resp.data.lower()
        assert User.query.filter_by(username="newuser").first() is None

    def test_short_password_rejected(self, client):
        resp = client.post("/auth/register", data=_patient_form(password="short"),
                           follow_redirects=True)
        assert b"at least 10 characters" in resp.data
        assert User.query.count() == 0

    def test_registration_is_atomic_on_bad_profile_input(self, client):
        """A-07: bad profile data must not leave an orphan User row."""
        client.post("/auth/register", data=_patient_form(date_of_birth="not-a-date"))
        assert User.query.count() == 0
        client.post("/auth/register", data=_patient_form(
            role="doctor", specialization="gp", license_number="L1", experience_years="abc"))
        assert User.query.count() == 0

    def test_doctor_registration_creates_profile(self, client):
        client.post("/auth/register", data=_patient_form(
            role="doctor", username="drx", email="drx@example.com",
            specialization="gp", license_number="L-1", experience_years="4"))
        user = User.query.filter_by(username="drx").one()
        assert Doctor.query.filter_by(user_id=user.id).count() == 1

    def test_login_success_and_next_open_redirect_blocked(self, client, make):
        u = make.user("patient", "dave")
        resp = client.post("/auth/login?next=//evil.example/x",
                           data={"email": u.email, "password": PASSWORD})
        assert resp.status_code == 302
        assert "evil.example" not in resp.headers["Location"]


class TestConsentRoutes:
    ADDR = "0xABCDEF1234567890ABCDEF1234567890ABCDEF12"

    def _patient_client(self, make, login_as, name="p1"):
        u, p = make.patient(name)
        return login_as(u), p

    def test_requires_login(self, client):
        assert client.post("/patient/consent/grant", json={}).status_code in (302, 401)
        assert client.post("/patient/consent/revoke", json={"consent_id": 1}).status_code in (302, 401)

    def test_missing_fields(self, make, login_as):
        c, _ = self._patient_client(make, login_as)
        assert c.post("/patient/consent/grant", json={"recipient": self.ADDR}).status_code == 400

    def test_bad_address_and_data_type_rejected(self, make, login_as):
        c, _ = self._patient_client(make, login_as)
        for body in ({"recipient": "0x1234", "data_type": "file", "purpose": "x"},
                     {"recipient": self.ADDR, "data_type": "everything", "purpose": "x"}):
            assert c.post("/patient/consent/grant", json=body).status_code == 400

    def test_grant_is_db_authoritative_and_chain_status_is_honest(self, make, login_as):
        c, p = self._patient_client(make, login_as)
        resp = c.post("/patient/consent/grant", json={
            "recipient": self.ADDR, "data_type": "lab_report", "purpose": "treatment"})
        body = resp.get_json()
        assert resp.status_code == 200 and body["status"] == "ok"
        assert body["consent_status"] == "active"
        assert body["on_chain"] is False                # no chain configured in tests
        assert body["chain_status"] == "unavailable"    # never claimed as anchored
        assert Consent.query.filter_by(patient_id=p.id, status="active").count() == 1

    def test_cannot_revoke_another_patients_consent(self, make, login_as):
        c1, _ = self._patient_client(make, login_as, "p1")
        cid = c1.post("/patient/consent/grant", json={
            "recipient": self.ADDR, "data_type": "file", "purpose": "x"}).get_json()["consent_id"]
        c2, _ = self._patient_client(make, login_as, "p2")
        assert c2.post("/patient/consent/revoke", json={"consent_id": cid}).status_code == 404
        assert db.session.get(Consent, cid).status == "active"

    def test_revoke_own_consent(self, make, login_as):
        c, _ = self._patient_client(make, login_as)
        cid = c.post("/patient/consent/grant", json={
            "recipient": self.ADDR, "data_type": "file", "purpose": "x"}).get_json()["consent_id"]
        body = c.post("/patient/consent/revoke", json={"consent_id": cid}).get_json()
        assert body["status"] == "ok" and body["consent_status"] == "revoked"
        assert db.session.get(Consent, cid).status == "revoked"

    def test_purpose_at_exactly_200_chars_accepted(self, make, login_as):
        c, p = self._patient_client(make, login_as)
        purpose = "a" * 200
        resp = c.post("/patient/consent/grant", json={
            "recipient": self.ADDR, "data_type": "lab_report", "purpose": purpose})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"

    def test_purpose_over_200_chars_rejected(self, make, login_as):
        c, _ = self._patient_client(make, login_as)
        purpose = "a" * 201
        resp = c.post("/patient/consent/grant", json={
            "recipient": self.ADDR, "data_type": "lab_report", "purpose": purpose})
        assert resp.status_code == 400

    def test_purpose_empty_string_rejected(self, make, login_as):
        c, _ = self._patient_client(make, login_as)
        resp = c.post("/patient/consent/grant", json={
            "recipient": self.ADDR, "data_type": "lab_report", "purpose": ""})
        assert resp.status_code == 400
