"""Central authorisation helpers.

Every route that returns or mutates patient data must decide access through
these functions, never through ad-hoc ``Consultation.query...first()`` checks.
"""
import re
from functools import wraps

from flask import abort, flash, jsonify, redirect, request, url_for
from flask_login import current_user

from app.models import Consultation, Doctor, Lab, Patient

# A "treatment relationship" requires a consultation the doctor has accepted.
# Pending requests, cancelled and unknown statuses do NOT grant access to a
# patient's record. ('in-progress' is accepted as a legacy spelling.)
TREATING_STATUSES = ("scheduled", "in_progress", "in-progress", "completed")

MIN_PASSWORD_LENGTH = 10
_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")


def valid_email(value: str) -> bool:
    return bool(value) and len(value) <= 120 and bool(_EMAIL_RE.match(value))


def password_problem(password: str | None) -> str | None:
    """Return a human readable problem with ``password`` or None if acceptable."""
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
    if len(password) > 128:
        return "Password must be at most 128 characters long."
    return None


def current_doctor() -> Doctor | None:
    if not current_user.is_authenticated or current_user.role != "doctor":
        return None
    return Doctor.query.filter_by(user_id=current_user.id).first()


def current_patient() -> Patient | None:
    if not current_user.is_authenticated or current_user.role != "patient":
        return None
    return Patient.query.filter_by(user_id=current_user.id).first()


def current_lab() -> Lab | None:
    if not current_user.is_authenticated or current_user.role != "lab":
        return None
    return Lab.query.filter_by(user_id=current_user.id).first()


def doctor_has_access(doctor_id: int, patient_id: int) -> bool:
    """True if ``doctor_id`` has an accepted treatment relationship with the patient."""
    patient = db_get(Patient, patient_id)
    if patient is None or patient.erased_at is not None:
        return False
    return (
        Consultation.query.filter(
            Consultation.doctor_id == doctor_id,
            Consultation.patient_id == patient_id,
            Consultation.status.in_(TREATING_STATUSES),
        ).first()
        is not None
    )


def patient_ids_for_doctor(doctor_id: int) -> list[int]:
    rows = (
        Consultation.query.with_entities(Consultation.patient_id)
        .filter(
            Consultation.doctor_id == doctor_id,
            Consultation.status.in_(TREATING_STATUSES),
        )
        .distinct()
        .all()
    )
    ids = [r[0] for r in rows]
    if not ids:
        return []
    live = Patient.query.with_entities(Patient.id).filter(
        Patient.id.in_(ids), Patient.erased_at.is_(None)
    )
    return [r[0] for r in live.all()]


def db_get(model, pk):
    from app import db

    try:
        return db.session.get(model, int(pk))
    except (TypeError, ValueError):
        return None


def wants_json() -> bool:
    return (
        request.is_json
        or request.path.startswith(("/lab/api", "/file-verification/api"))
        or "application/json" in request.headers.get("Accept", "")
    )


def deny(message: str = "Access denied.", redirect_to: str = "main.home"):
    """Uniform denial: JSON 403 for API callers, flash + redirect otherwise."""
    if wants_json():
        return jsonify({"status": "error", "error": message}), 403
    flash(message, "error")
    return redirect(url_for(redirect_to))


def roles_required(*roles):
    """Decorator: the logged in user must have one of ``roles``."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                if wants_json():
                    return jsonify({"status": "error", "error": "Authentication required."}), 401
                return redirect(url_for("auth.login"))
            if current_user.role not in roles:
                return deny("Access denied.")
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def json_body() -> dict:
    """Parsed JSON object body, or {} for a missing / malformed / non-object body."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def abort_not_found():
    abort(404)


# ---------------------------------------------------------------------------
# Stored-file access (encrypted uploads). Decided from the DB only, never from
# anything read back from a blockchain or IPFS.
# ---------------------------------------------------------------------------

def can_access_file(user, stored) -> bool:
    """Patient owner, treating doctor, or the uploading user. Admins get no PHI."""
    if user is None or not user.is_authenticated or stored is None or stored.is_erased:
        return False
    if stored.uploaded_by_user_id == user.id:
        return True
    if user.role == "patient":
        patient = current_patient()
        return patient is not None and patient.id == stored.patient_id
    if user.role == "doctor":
        doctor = current_doctor()
        return doctor is not None and doctor_has_access(doctor.id, stored.patient_id)
    return False


def can_upload_for_patient(user, patient) -> bool:
    """Who may attach a file to ``patient``: the patient, a treating doctor, or a
    lab holding an open request for that patient."""
    from app.models import LabRequest

    if patient is None or patient.erased_at is not None:
        return False
    if user.role == "patient":
        mine = current_patient()
        return mine is not None and mine.id == patient.id
    if user.role == "doctor":
        doctor = current_doctor()
        return doctor is not None and doctor_has_access(doctor.id, patient.id)
    if user.role == "lab":
        lab = current_lab()
        return lab is not None and LabRequest.query.filter(
            LabRequest.lab_id == lab.id, LabRequest.patient_id == patient.id,
            LabRequest.status.in_(("pending", "approved", "completed"))).first() is not None
    return False
