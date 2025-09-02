from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from app import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'patient', 'doctor', 'admin', 'lab'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    address = db.Column(db.Text, nullable=False)
    emergency_contact = db.Column(db.String(15), nullable=False)
    blood_group = db.Column(db.String(5), nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
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
    image_path = db.Column(db.String(255), nullable=True)
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
    prescribed_date = db.Column(db.Date, default=datetime.utcnow().date)
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
    date = db.Column(db.Date, default=datetime.utcnow().date)
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
    requested_date = db.Column(db.Date, default=datetime.utcnow().date)
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