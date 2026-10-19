import secrets
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from app import db


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'patient', 'doctor', 'admin', 'lab'
    # Decentralized Identifier (W3C DID, did:ethr:<eth_address>) — set at registration
    did = db.Column(db.String(120), unique=True, nullable=True)
    # Rotated on logout / password change / erasure; embedded in the session id
    # so that a copied session cookie stops working (see User.get_id()).
    session_token = db.Column(db.String(32), nullable=False, default=lambda: secrets.token_hex(16))
    # False for deactivated or erased accounts; they can no longer log in.
    active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def is_active(self):  # Flask-Login: inactive users cannot log in or keep a session
        return bool(self.active)

    def get_id(self):
        return f"{self.id}:{self.session_token}"

    def rotate_session_token(self):
        self.session_token = secrets.token_hex(16)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        self.rotate_session_token()

    def check_password(self, password):
        # Explicit sentinel guard: erased accounts must never authenticate,
        # and passing a sentinel string to check_password_hash may raise or
        # log a noisy ValueError on some Werkzeug versions (F-07).
        if self.password_hash in (None, '', '[ERASED]'):
            return False
        if not password:
            return False
        try:
            return check_password_hash(self.password_hash, password)
        except ValueError:  # other malformed hashes
            return False

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=True)  # nullable so erasure can clear it
    gender = db.Column(db.String(10), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    address = db.Column(db.Text, nullable=False)
    emergency_contact = db.Column(db.String(15), nullable=False)
    blood_group = db.Column(db.String(5), nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    erased_at = db.Column(db.DateTime, nullable=True)  # set by the erasure workflow
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship('User', backref='patient_profile')
    consultations = db.relationship('Consultation', backref='patient', lazy=True)
    lab_reports = db.relationship('LabReport', backref='patient', lazy=True)
    prescriptions = db.relationship('Prescription', backref='patient', lazy=True)

class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    license_number = db.Column(db.String(50), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    address = db.Column(db.Text, nullable=False)
    experience_years = db.Column(db.Integer, nullable=False)
    education = db.Column(db.Text, nullable=False)
    consultation_fee = db.Column(db.Float, default=0.0)
    is_available = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship('User', backref='doctor_profile')
    consultations = db.relationship('Consultation', backref='doctor', lazy=True)
    lab_reports = db.relationship('LabReport', backref='doctor', lazy=True)
    prescriptions = db.relationship('Prescription', backref='doctor', lazy=True)

class Lab(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    lab_name = db.Column(db.String(100), nullable=False)
    license_number = db.Column(db.String(50), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    address = db.Column(db.Text, nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship('User', backref='lab_profile')
    lab_reports = db.relationship('LabReport', backref='lab', lazy=True)

class Consultation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    reason = db.Column(db.Text, nullable=False)
    consultation_type = db.Column(db.String(50), default='in-person')
    duration = db.Column(db.Integer, default=30)  # minutes
    status = db.Column(db.String(20), default='scheduled')  # scheduled, in-progress, completed, cancelled
    notes = db.Column(db.Text, nullable=True)
    diagnosis = db.Column(db.Text, nullable=True)
    treatment_plan = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LabReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    lab_id = db.Column(db.Integer, db.ForeignKey('lab.id'), nullable=False)
    consultation_id = db.Column(db.Integer, db.ForeignKey('consultation.id'), nullable=True)
    report_type = db.Column(db.String(50), nullable=False)  # 'retinal', 'blood', 'urine', etc.
    image_path = db.Column(db.String(255), nullable=True)  # legacy (pre-encryption) value; not served
    stored_file_id = db.Column(db.Integer, db.ForeignKey('stored_file.id'), nullable=True)
    diagnosis = db.Column(db.String(100), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    findings = db.Column(db.Text, nullable=True)
    recommendations = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='pending')  # pending, completed, delivered
    amount_charged = db.Column(db.Float, default=1000.0)
    is_paid = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    consultation = db.relationship('Consultation', backref='lab_reports')
    stored_file = db.relationship('StoredFile', foreign_keys=[stored_file_id])

class Prescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    consultation_id = db.Column(db.Integer, db.ForeignKey('consultation.id'), nullable=True)
    medication_name = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50), nullable=False)
    frequency = db.Column(db.String(50), nullable=False)
    duration = db.Column(db.String(50), nullable=False)
    instructions = db.Column(db.Text, nullable=True)
    prescribed_date = db.Column(db.Date, default=lambda: datetime.utcnow().date())
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    consultation = db.relationship('Consultation', backref='prescriptions')

class MedicalRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    record_type = db.Column(db.String(50), nullable=False)  # 'lab_report', 'prescription', 'consultation'
    record_id = db.Column(db.Integer, nullable=False)  # ID of the related record
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    date = db.Column(db.Date, default=lambda: datetime.utcnow().date())
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship('Patient', backref='medical_records')

class LabRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    lab_id = db.Column(db.Integer, db.ForeignKey('lab.id'), nullable=False)
    consultation_id = db.Column(db.Integer, db.ForeignKey('consultation.id'), nullable=True)
    request_type = db.Column(db.String(50), nullable=False)  # 'retinal', 'blood', 'urine', etc.
    reason = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, approved, completed, rejected
    priority = db.Column(db.String(20), default='normal')  # low, normal, high, urgent
    requested_date = db.Column(db.Date, default=lambda: datetime.utcnow().date())
    completed_date = db.Column(db.Date, nullable=True)
    lab_report_id = db.Column(db.Integer, db.ForeignKey('lab_report.id'), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = db.relationship('Patient', backref='lab_requests')
    doctor = db.relationship('Doctor', backref='lab_requests')
    lab = db.relationship('Lab', backref='lab_requests')
    consultation = db.relationship('Consultation', backref='lab_requests')
    lab_report = db.relationship('LabReport', backref='lab_request')

class StoredFile(db.Model):
    """Metadata for an encrypted, off-chain file.

    The plaintext never leaves the application server. Only ciphertext is
    written to disk / IPFS. The chain (when reachable) receives only an
    opaque id and a *keyed* digest (HMAC-SHA256) of the plaintext.
    Authorisation is always decided from this table (patient_id,
    uploaded_by_user_id) and never from chain data.
    """
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(32), unique=True, nullable=False)  # opaque uuid4 hex
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    uploaded_by_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    category = db.Column(db.String(30), nullable=False, default='medical')
    description = db.Column(db.String(500), nullable=True)
    original_filename = db.Column(db.String(255), nullable=False)
    extension = db.Column(db.String(10), nullable=False)
    size_bytes = db.Column(db.Integer, nullable=False)
    anchor_digest = db.Column(db.String(64), nullable=False)   # HMAC-SHA256(anchor key, plaintext), hex
    nonce = db.Column(db.String(32), nullable=False)           # base64, AES-256-GCM file nonce
    wrapped_dek = db.Column(db.Text, nullable=True)            # base64; NULL after crypto-shredding
    kek_id = db.Column(db.String(32), nullable=False, default='v1')
    local_path = db.Column(db.String(255), nullable=True)      # relative to UPLOAD_DIR; NULL once removed
    ipfs_cid = db.Column(db.String(100), nullable=True)
    ipfs_status = db.Column(db.String(20), nullable=False, default='not_attempted')  # not_attempted|pinned|failed|unpinned
    chain_file_id = db.Column(db.Integer, nullable=True)
    chain_tx = db.Column(db.String(80), nullable=True)
    anchor_status = db.Column(db.String(20), nullable=False, default='not_anchored')  # not_anchored|anchored|unavailable|failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    erased_at = db.Column(db.DateTime, nullable=True)

    patient = db.relationship('Patient', backref='stored_files')
    uploaded_by = db.relationship('User', foreign_keys=[uploaded_by_user_id])

    @property
    def is_erased(self):
        return self.erased_at is not None


class Consent(db.Model):
    """Application-level consent record -- the authoritative source for the API.

    ``chain_status`` states whether an anchoring transaction was actually mined
    (``anchored``), could not be attempted (``unavailable``) or was attempted and
    failed (``failed``). It is never assumed. The anchoring transaction is
    signed by the server's operator account, so the chain proves "the operator
    recorded this", not "the patient signed this".
    """
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    recipient = db.Column(db.String(100), nullable=False)       # 0x checksum address
    data_type = db.Column(db.String(50), nullable=False)
    purpose = db.Column(db.String(200), nullable=False)
    purpose_hash = db.Column(db.String(70), nullable=False)
    status = db.Column(db.String(10), nullable=False, default='active')  # active | revoked
    chain_status = db.Column(db.String(20), nullable=False, default='not_anchored')
    chain_consent_id = db.Column(db.Integer, nullable=True)
    chain_tx = db.Column(db.String(80), nullable=True)
    revoke_chain_status = db.Column(db.String(20), nullable=True)
    revoke_chain_tx = db.Column(db.String(80), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    revoked_at = db.Column(db.DateTime, nullable=True)

    patient = db.relationship('Patient', backref='consents')


class ErasureLog(db.Model):
    """Accountability record of an erasure. Deliberately holds NO personal data
    (no user / patient reference) -- only an opaque reference and counters."""
    id = db.Column(db.Integer, primary_key=True)
    erasure_ref = db.Column(db.String(64), unique=True, nullable=False)   # random, opaque
    requested_at = db.Column(db.DateTime, default=datetime.utcnow)
    summary = db.Column(db.Text, nullable=True)                           # JSON counters/status only
    chain_status = db.Column(db.String(20), nullable=False, default='not_anchored')
    chain_tx = db.Column(db.String(80), nullable=True)


class VerificationLog(db.Model):
    """Result of an integrity check (HMAC of the supplied bytes vs. the stored digest)."""
    id = db.Column(db.Integer, primary_key=True)
    stored_file_id = db.Column(db.Integer, db.ForeignKey('stored_file.id'), nullable=False)
    verified_by_user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    is_match = db.Column(db.Boolean, nullable=False)
    notes = db.Column(db.String(500), nullable=True)
    chain_status = db.Column(db.String(20), nullable=False, default='not_anchored')
    chain_tx = db.Column(db.String(80), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    stored_file = db.relationship('StoredFile', backref='verifications')
    verified_by = db.relationship('User', foreign_keys=[verified_by_user_id])
