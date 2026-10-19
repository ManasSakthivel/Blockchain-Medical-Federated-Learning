import hashlib
import io
import logging
import os
import pickle
from datetime import date, datetime

from flask import (Blueprint, current_app, flash, jsonify, redirect, render_template,
                   request, url_for)
from flask_login import current_user, login_user
from PIL import Image, UnidentifiedImageError

from app.access import current_lab, json_body, password_problem, roles_required
from app.models import (Consultation, Doctor, LabReport, LabRequest, MedicalRecord,
                        Patient, User, db)
from app.services import file_store

log = logging.getLogger(__name__)
lab_bp = Blueprint('lab', __name__)

# Retinal disease classes (matching model.pkl)
RETINAL_DISEASES = ['NORMAL', 'DR', 'AMD', 'CNV', 'DME', 'CSR', 'DRUSEN', 'MH']
REPORT_TYPES = ('retinal', 'blood', 'urine', 'xray', 'mri', 'other')
LAB_IMAGE_EXTENSIONS = file_store.IMAGE_EXTENSIONS
# A lab may only work on requests that a doctor/patient addressed to it.
OPEN_REQUEST_STATUSES = ('pending', 'approved')


# ---------------------------------------------------------------------------
# Prototype classifier. NOTE: model.pkl is an exact-image-hash lookup table,
# not a trained network; unknown images are reported as inconclusive.
# ---------------------------------------------------------------------------

def load_retinal_model():
    """Load a Keras model if present, else the pickled prototype lookup table.

    The pickle is a repository asset (never user supplied). Do not load
    pickles from any other source.
    """
    try:
        keras_path = os.path.join(current_app.root_path, 'models', 'retinal_cnn.h5')
        if os.path.exists(keras_path):
            import tensorflow as tf
            return tf.keras.models.load_model(keras_path)
        model_path = os.path.join(current_app.root_path, 'models', 'model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as fh:
                return pickle.load(fh)  # nosec B301
        log.warning('No retinal model found; AI classification disabled.')
    except Exception:
        log.exception('Error loading retinal model')
    return None


def classify_retinal_disease(image_bytes: bytes):
    """Classify in memory. Returns (label, confidence)."""
    model = load_retinal_model()
    if model is None:
        return 'AI Module Offline', 0.0
    try:
        if hasattr(model, 'predict'):
            import numpy as np
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
            arr = np.expand_dims(np.array(img) / 255.0, axis=0)
            preds = model.predict(arr)
            idx = int(np.argmax(preds[0]))
            label = RETINAL_DISEASES[idx] if idx < len(RETINAL_DISEASES) else 'Unknown'
            return label, float(np.max(preds[0]))
        # Prototype lookup keyed by MD5 of the file (lookup key only, not security)
        key = hashlib.md5(image_bytes, usedforsecurity=False).hexdigest()
        if key in model:
            result = model[key]
            if isinstance(result, tuple):
                return result[0], float(result[1])
            return result, 1.0
        return 'Inconclusive (Requires Specialist Review)', 0.0
    except Exception:
        log.exception('Error in classification')
        return 'Internal Error', 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lab_or_redirect():
    lab = current_lab()
    if lab is None:
        flash('Lab profile not found.', 'error')
    return lab


def _lab_requests_for(lab):
    return LabRequest.query.filter(
        LabRequest.lab_id == lab.id, LabRequest.status.in_(OPEN_REQUEST_STATUSES))


def _verify_image(data: bytes):
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise file_store.UploadRejected('The uploaded file is not a valid image.')


def _read_upload():
    file = request.files.get('image')
    if not file or not file.filename:
        raise file_store.UploadRejected('No image file selected.')
    data = file.read(file_store.MAX_BYTES + 1)
    file_store.validate_upload(file.filename, data, allowed=LAB_IMAGE_EXTENSIONS)
    _verify_image(data)
    return file.filename, data


def _create_report(*, lab, patient_id, doctor_id, consultation_id, report_type,
                   filename, data, lab_request=None):
    """Store the encrypted image, create the report + medical record atomically."""
    stored = file_store.store_file(
        patient_id=patient_id, uploader_user_id=current_user.id, filename=filename,
        data=data, category='lab_report', allowed=LAB_IMAGE_EXTENSIONS)
    diagnosis = confidence = None
    if report_type == 'retinal':
        diagnosis, confidence = classify_retinal_disease(data)

    db.session.flush()  # assigns stored.id
    report = LabReport(
        patient_id=patient_id, doctor_id=doctor_id, lab_id=lab.id,
        consultation_id=consultation_id, report_type=report_type,
        stored_file_id=stored.id, diagnosis=diagnosis, confidence_score=confidence,
        findings=request.form.get('findings'),
        recommendations=request.form.get('recommendations'),
        status='completed', amount_charged=1000.0, is_paid=True)
    db.session.add(report)
    db.session.flush()  # report.id is needed by the medical record (A-06)

    if lab_request is not None:
        lab_request.status = 'completed'
        lab_request.completed_date = date.today()
        lab_request.lab_report_id = report.id
        lab_request.notes = request.form.get('notes')
        lab_request.updated_at = datetime.utcnow()

    db.session.add(MedicalRecord(
        patient_id=patient_id, record_type='lab_report', record_id=report.id,
        title=f'Lab Report - {report_type.title()}',
        description=f'Report completed by {lab.lab_name}', date=date.today()))
    db.session.commit()
    return report


def _resolve_target():
    """Resolve patient / doctor / consultation from the upload form.

    The lab may only upload for a patient+doctor pair that has an OPEN request
    addressed to this lab, and a consultation id must belong to that pair.
    """
    lab = current_lab()
    patient_email = (request.form.get('patient_email') or '').strip().lower()
    doctor_email = (request.form.get('doctor_email') or '').strip().lower()
    report_type = (request.form.get('report_type') or '').strip()
    if not patient_email or not doctor_email or report_type not in REPORT_TYPES:
        raise file_store.UploadRejected('Patient e-mail, doctor e-mail and a valid report type are required.')

    p_user = User.query.filter_by(email=patient_email, role='patient').first()
    d_user = User.query.filter_by(email=doctor_email, role='doctor').first()
    patient = Patient.query.filter_by(user_id=p_user.id).first() if p_user else None
    doctor = Doctor.query.filter_by(user_id=d_user.id).first() if d_user else None
    # Same message for every failure: do not reveal which e-mails exist (A-13).
    not_authorised = file_store.UploadRejected(
        'No open request from this doctor and patient is addressed to your lab.')
    if not patient or not doctor or patient.erased_at is not None:
        raise not_authorised
    if not _lab_requests_for(lab).filter_by(patient_id=patient.id, doctor_id=doctor.id).first():
        raise not_authorised

    consultation_id = None
    raw = (request.form.get('consultation_id') or '').strip()
    if raw:
        try:
            consultation_id = int(raw)
        except ValueError:
            raise file_store.UploadRejected('Invalid consultation.')
        c = db.session.get(Consultation, consultation_id)
        if not c or c.patient_id != patient.id or c.doctor_id != doctor.id:
            raise file_store.UploadRejected('Invalid consultation.')
    return patient, doctor, consultation_id, report_type


def _upload_view(template, success_message, **extra):
    lab = _lab_or_redirect()
    if lab is None:
        return redirect(url_for('main.home'))
    endpoint = 'lab.' + request.endpoint.split('.')[-1]
    if request.method == 'POST':
        try:
            patient, doctor, consultation_id, report_type = _resolve_target()
            filename, data = _read_upload()
            _create_report(lab=lab, patient_id=patient.id, doctor_id=doctor.id,
                           consultation_id=consultation_id, report_type=report_type,
                           filename=filename, data=data)
        except file_store.UploadRejected as exc:
            db.session.rollback()
            flash(str(exc), 'error')
            return redirect(url_for(endpoint))
        flash(success_message, 'success')
        return redirect(url_for('lab.reports'))

    consultations = Consultation.query.join(
        LabRequest, LabRequest.consultation_id == Consultation.id).filter(
        LabRequest.lab_id == lab.id).all()
    return render_template(template, lab=lab, consultations=consultations, **extra)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@lab_bp.route('/dashboard')
@roles_required('lab')
def dashboard():
    lab = _lab_or_redirect()
    if lab is None:
        return redirect(url_for('main.home'))
    total_reports = LabReport.query.filter_by(lab_id=lab.id).count()
    pending_reports = LabReport.query.filter_by(lab_id=lab.id, status='pending').count()
    completed_reports = LabReport.query.filter_by(lab_id=lab.id, status='completed').count()
    total_revenue = db.session.query(db.func.sum(LabReport.amount_charged)).filter_by(
        lab_id=lab.id, is_paid=True).scalar() or 0
    recent_reports = LabReport.query.filter_by(lab_id=lab.id).order_by(
        LabReport.created_at.desc()).limit(5).all()
    return render_template('lab/dashboard.html', lab=lab, total_reports=total_reports,
                           pending_reports=pending_reports, completed_reports=completed_reports,
                           total_revenue=total_revenue, recent_reports=recent_reports)


@lab_bp.route('/upload-report', methods=['GET', 'POST'])
@roles_required('lab')
def upload_report():
    return _upload_view('lab/upload_report.html', 'Lab report uploaded successfully!')


@lab_bp.route('/upload-scan', methods=['GET', 'POST'])
@roles_required('lab')
def upload_scan():
    return _upload_view('lab/upload_scan.html', 'Lab report uploaded successfully!',
                        retinal_diseases=RETINAL_DISEASES)


@lab_bp.route('/reports')
@roles_required('lab')
def reports():
    lab = _lab_or_redirect()
    if lab is None:
        return redirect(url_for('main.home'))
    items = LabReport.query.filter_by(lab_id=lab.id).order_by(LabReport.created_at.desc()).all()
    return render_template('lab/reports.html', lab=lab, reports=items)


@lab_bp.route('/report/<int:report_id>')
@roles_required('lab')
def view_report(report_id):
    lab = _lab_or_redirect()
    if lab is None:
        return redirect(url_for('main.home'))
    report = LabReport.query.filter_by(id=report_id, lab_id=lab.id).first()
    if not report:
        flash('Report not found.', 'error')
        return redirect(url_for('lab.reports'))
    return render_template('lab/view_report.html', lab=lab, report=report)


@lab_bp.route('/profile', methods=['GET', 'POST'])
@roles_required('lab')
def profile():
    lab = _lab_or_redirect()
    if lab is None:
        return redirect(url_for('main.home'))
    if request.method == 'POST':
        name = (request.form.get('lab_name') or '').strip()
        if not name:
            flash('Lab name is required.', 'error')
            return redirect(url_for('lab.profile'))
        lab.lab_name = name[:100]
        lab.phone = (request.form.get('phone') or lab.phone)[:15]
        lab.address = request.form.get('address') or lab.address
        lab.specialization = (request.form.get('specialization') or lab.specialization)[:100]
        lab.updated_at = datetime.utcnow()
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('lab.profile'))
    return render_template('lab/profile.html', lab=lab)


@lab_bp.route('/requests')
@roles_required('lab')
def requests():
    lab = _lab_or_redirect()
    if lab is None:
        return redirect(url_for('main.home'))
    items = LabRequest.query.filter_by(lab_id=lab.id).order_by(LabRequest.created_at.desc()).all()
    return render_template('lab/requests.html', lab=lab, requests=items)


@lab_bp.route('/request/<int:request_id>')
@roles_required('lab')
def view_request(request_id):
    lab = _lab_or_redirect()
    if lab is None:
        return redirect(url_for('main.home'))
    lab_request = LabRequest.query.filter_by(id=request_id, lab_id=lab.id).first()
    if not lab_request:
        flash('Request not found.', 'error')
        return redirect(url_for('lab.requests'))
    return render_template('lab/view_request.html', lab=lab, lab_request=lab_request)


@lab_bp.route('/request/<int:request_id>/process', methods=['GET', 'POST'])
@roles_required('lab')
def process_request(request_id):
    lab = _lab_or_redirect()
    if lab is None:
        return redirect(url_for('main.home'))
    lab_request = LabRequest.query.filter_by(id=request_id, lab_id=lab.id).first()
    if not lab_request:
        flash('Request not found.', 'error')
        return redirect(url_for('lab.requests'))
    if lab_request.status not in OPEN_REQUEST_STATUSES:
        flash(f"This request is already '{lab_request.status}'.", 'error')
        return redirect(url_for('lab.requests'))

    if request.method == 'POST':
        try:
            filename, data = _read_upload()
            _create_report(lab=lab, patient_id=lab_request.patient_id,
                           doctor_id=lab_request.doctor_id,
                           consultation_id=lab_request.consultation_id,
                           report_type=lab_request.request_type, filename=filename,
                           data=data, lab_request=lab_request)
        except file_store.UploadRejected as exc:
            db.session.rollback()
            flash(str(exc), 'error')
            return redirect(url_for('lab.process_request', request_id=request_id))
        flash('Lab report completed and sent to patient!', 'success')
        return redirect(url_for('lab.requests'))

    return render_template('lab/process_request.html', lab=lab, lab_request=lab_request)


@lab_bp.route('/detect', methods=['POST'])
@roles_required('lab')
def detect():
    if (request.form.get('report_type') or '') != 'retinal':
        return jsonify({'success': False,
                        'message': 'AI detection is only available for retinal scans.'}), 400
    try:
        _, data = _read_upload()
    except file_store.UploadRejected as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400
    diagnosis, confidence = classify_retinal_disease(data)  # in memory: no temp file
    return jsonify({'success': True, 'diagnosis': diagnosis, 'confidence': confidence})


@lab_bp.route('/profile/change-password', methods=['POST'])
@roles_required('lab')
def change_password():
    """F-02: authenticated lab users can rotate their own password.

    Accepts JSON ``{current_password, new_password, confirm_password}``.
    Mirrors doctor.change_password: verifies current password, rotates the
    session token so all other sessions are invalidated, then re-establishes
    the current session via ``login_user``.
    """
    data = json_body()
    current_password = data.get('current_password') or ''
    new_password = data.get('new_password') or ''
    confirm = data.get('confirm_password')

    if not current_user.check_password(current_password):
        return jsonify({'success': False, 'message': 'Current password is incorrect'}), 403
    if confirm is not None and new_password != confirm:
        return jsonify({'success': False, 'message': 'New passwords do not match'}), 400
    problem = password_problem(new_password)
    if problem:
        return jsonify({'success': False, 'message': problem}), 400

    current_user.set_password(new_password)  # also rotates session token
    db.session.commit()
    login_user(current_user)  # keep THIS session alive; all others are now invalid
    return jsonify({'success': True, 'message': 'Password changed successfully'})
