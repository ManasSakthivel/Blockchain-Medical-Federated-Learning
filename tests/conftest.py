"""Shared pytest fixtures for the blockchain-medical-fl test suite.

Everything here is test-only: credentials are fixed placeholders, the Flask app
runs with TESTING=True, an in-memory SQLite database and ephemeral keys. None of
these values are read by production code.
"""
import datetime as _dt
import hashlib
import os
import sys

import numpy as np
import pytest

# Make the project root importable without a package install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Test-only placeholders. Never valid outside the test suite.
PASSWORD = "Correct-Horse-9"
TEST_PASSWORD = PASSWORD
TEST_SECRET_KEY = "test-only-secret-key-not-for-production"


@pytest.fixture()
def small_dataset():
    """Small synthetic dataset (5 features, 60 samples, binary labels).

    Function-scoped so a test that mutates the arrays cannot leak into another test.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _base_app_cfg(**overrides):
    """Minimal TESTING=True Flask config for the test app."""
    cfg = {
        "TESTING": True,
        "SECRET_KEY": TEST_SECRET_KEY,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False,
    }
    cfg.update(overrides)
    return cfg


def _stable_did(name):
    """Deterministic DID per name. Python's hash() is salted per process, so it is not stable."""
    return "did:ethr:0x" + hashlib.sha256(name.encode()).hexdigest()[:40]


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    # Developer shell settings must not change the test configuration.
    for name in ("EHR_DEV_MODE", "SESSION_COOKIE_SECURE", "BLOCKCHAIN_OPERATOR_PRIVATE_KEY"):
        monkeypatch.delenv(name, raising=False)

    from app import create_app, db
    _app = create_app(_base_app_cfg())

    @_app.before_request
    def _fresh_login_state():
        # This fixture keeps ONE app context open for the whole test, so Flask-Login's
        # per-context user cache would leak between test clients. Production creates
        # a new app context per request; emulate that here.
        from flask import g
        g.pop("_login_user", None)

    with _app.app_context():
        db.create_all()
        yield _app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def make(app):
    """Factory for users, profiles and relationships, committed to the test DB."""
    from app import db
    from app.models import Consultation, Doctor, Lab, LabRequest, Patient, User

    class Factory:
        def user(self, role, name):
            u = User(username=name, email=f"{name.lower()}@example.com", role=role,
                     did=_stable_did(name))
            u.set_password(TEST_PASSWORD)
            db.session.add(u)
            db.session.commit()
            return u

        def patient(self, name):
            u = self.user("patient", name)
            p = Patient(user_id=u.id, first_name=name, last_name="Test",
                        date_of_birth=_dt.date(1990, 1, 1), gender="f", phone="1",
                        address="addr", emergency_contact="2")
            db.session.add(p)
            db.session.commit()
            return u, p

        def doctor(self, name):
            u = self.user("doctor", name)
            d = Doctor(user_id=u.id, first_name=name, last_name="Doc",
                       specialization="gp", license_number=f"L-{name}", phone="1",
                       address="addr", experience_years=3, education="med")
            db.session.add(d)
            db.session.commit()
            return u, d

        def lab(self, name):
            u = self.user("lab", name)
            lab = Lab(user_id=u.id, lab_name=name, license_number=f"LAB-{name}",
                      phone="1", address="addr", specialization="retinal")
            db.session.add(lab)
            db.session.commit()
            return u, lab

        def consultation(self, patient, doctor, status="scheduled"):
            c = Consultation(patient_id=patient.id, doctor_id=doctor.id,
                             date=_dt.date.today(), time=_dt.time(9, 0),
                             reason="checkup", status=status)
            db.session.add(c)
            db.session.commit()
            return c

        def lab_request(self, patient, doctor, lab, status="pending"):
            r = LabRequest(patient_id=patient.id, doctor_id=doctor.id, lab_id=lab.id,
                           request_type="retinal", reason="screen", status=status)
            db.session.add(r)
            db.session.commit()
            return r

    return Factory()


def login(client, user, password=TEST_PASSWORD):
    return client.post("/auth/login", data={"email": user.email, "password": password})


@pytest.fixture()
def login_as(app):
    """Return a factory that yields a fresh, logged-in test client for a user."""
    def _login(user):
        c = app.test_client()
        resp = login(c, user)
        assert resp.status_code == 302, "login should redirect on success"
        return c

    return _login
