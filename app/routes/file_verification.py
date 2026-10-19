import hmac
import io
import logging

from flask import (Blueprint, flash, jsonify, redirect, render_template, request,
                   send_file, url_for)
from flask_login import current_user

from app.access import (TREATING_STATUSES, can_access_file, can_upload_for_patient,
                        roles_required)
from app.models import (Consultation, Doctor, Lab, LabRequest,
                        Patient, StoredFile, User, VerificationLog, db)
from app.services import anchoring, file_store
from app.services.crypto_service import DecryptionError

log = logging.getLogger(__name__)
file_verification_bp = Blueprint('file_verification', __name__)

CATEGORIES = ('medical', 'lab_report', 'prescription', 'imaging', 'other')
UPLOADER_ROLES = ('patient', 'doctor', 'lab')
ALL_ROLES = ('patient', 'doctor', 'lab')


def _record(stored: StoredFile) -> dict:
    """Template-friendly view of a StoredFile. Contains no storage secrets."""
    return {
        'file_id': stored.id,
        'file_name': stored.original_filename,
        'file_type': stored.extension,
        'file_size': stored.size_bytes,
        'timestamp': int(stored.created_at.timestamp()) if stored.created_at else None,
        'is_valid': not stored.is_erased,
        'uploaded_by': stored.uploaded_by.username if stored.uploaded_by else '-',
        'patient_id': stored.patient_id,
        'ipfs_hash': stored.ipfs_cid or '(not stored on IPFS)',
        'file_hash': stored.anchor_digest,
        'anchor_status': stored.anchor_status,
        'ipfs_status': stored.ipfs_status,
        'metadata': {'description': stored.description, 'category': stored.category},
    }


def _get_authorised(file_id):
    """Load a file the current user may access, else None (uniform 'not found')."""
    stored = db.session.get(StoredFile, file_id)
    return stored if can_access_file(current_user, stored) else None


def _accessible_patients():
    """Return the patients the current user may upload files for.

    F-10: For lab and doctor roles the access filter is pushed into SQL so the
    function never loads all patients into Python memory.
    The patient role path is already a single indexed lookup (unchanged).
    Authorization is enforced independently in _do_upload() → can_upload_for_patient();
    this function only populates the upload-form dropdown.
    """
    if current_user.role == 'patient':
        mine = Patient.query.filter_by(user_id=current_user.id).first()
        return [mine] if mine else []

    if current_user.role == 'lab':
        lab = Lab.query.filter_by(user_id=current_user.id).first()
        if lab is None:
            return []
        return (
            Patient.query
            .join(LabRequest, LabRequest.patient_id == Patient.id)
            .filter(
                LabRequest.lab_id == lab.id,
                LabRequest.status.in_(('pending', 'approved', 'completed')),
                Patient.erased_at.is_(None),
            )
            .distinct()
            .all()
        )

    if current_user.role == 'doctor':
        doctor = Doctor.query.filter_by(user_id=current_user.id).first()
        if doctor is None:
            return []
        return (
            Patient.query
            .join(Consultation, Consultation.patient_id == Patient.id)
            .filter(
                Consultation.doctor_id == doctor.id,
                Consultation.status.in_(TREATING_STATUSES),
                Patient.erased_at.is_(None),
            )
            .distinct()
            .all()
        )

    return []


def _do_upload(patient: Patient, file, description: str, category: str):
    if not can_upload_for_patient(current_user, patient):
        raise file_store.UploadRejected('You are not allowed to upload files for this patient.')
    if category not in CATEGORIES:
        raise file_store.UploadRejected('Invalid category.')
    if not file or not file.filename:
        raise file_store.UploadRejected('No file selected.')
    data = file.read(file_store.MAX_BYTES + 1)
    stored = file_store.store_file(
        patient_id=patient.id, uploader_user_id=current_user.id, filename=file.filename,
        data=data, category=category)
    stored.description = (description or '')[:500] or None
    db.session.flush()
    outcome = anchoring.anchor_file(stored.public_id, stored.anchor_digest)
    stored.anchor_status = outcome.status if outcome.status != 'unavailable' else 'unavailable'
    stored.chain_file_id, stored.chain_tx = outcome.object_id, outcome.tx_hash
    db.session.commit()
    return stored


@file_verification_bp.route('/upload', methods=['GET', 'POST'])
@roles_required(*UPLOADER_ROLES)
def upload_file():
    if request.method == 'POST':
        email = (request.form.get('patient_email') or '').strip().lower()
        user = User.query.filter_by(email=email, role='patient').first() if email else None
        patient = Patient.query.filter_by(user_id=user.id).first() if user else None
        try:
            if patient is None:  # same answer as "not permitted": no e-mail enumeration
                raise file_store.UploadRejected('You are not allowed to upload files for this patient.')
            stored = _do_upload(patient, request.files.get('file'),
                                request.form.get('description'),
                                request.form.get('category', 'medical'))
        except file_store.UploadRejected as exc:
            db.session.rollback()
            flash(str(exc), 'error')
            return redirect(url_for('file_verification.upload_file'))
        flash(f'File uploaded (encrypted). File ID: {stored.id}. '
              f'Blockchain anchor: {stored.anchor_status}.', 'success')
        return redirect(url_for('file_verification.view_file', file_id=stored.id))
    return render_template('file_verification/upload.html', patients=_accessible_patients())


@file_verification_bp.route('/verify', methods=['GET', 'POST'])
@roles_required(*ALL_ROLES)
def verify_file():
    if request.method == 'POST':
        try:
            file_id = int(request.form.get('file_id', ''))
        except ValueError:
            flash('A numeric file ID is required.', 'error')
            return redirect(url_for('file_verification.verify_file'))
        upload = request.files.get('file')
        if not upload or not upload.filename:
            flash('No file selected for verification.', 'error')
            return redirect(url_for('file_verification.verify_file'))
        result = _verify(file_id, upload.read(file_store.MAX_BYTES + 1),
                         (request.form.get('notes') or '')[:500])
        if result is None:
            flash('File not found.', 'error')
            return redirect(url_for('file_verification.verify_file'))
        flash('File integrity verified: the file is identical to the stored original.'
              if result['is_match'] else
              'WARNING: this file does NOT match the stored original.',
              'success' if result['is_match'] else 'error')
        return render_template('file_verification/verification_result.html', result=result)
    return render_template('file_verification/verify.html')


def _verify(file_id: int, data: bytes, notes: str):
    stored = _get_authorised(file_id)
    if stored is None:
        return None
    crypto = file_store.get_crypto()
    current = crypto.anchor_digest(data)
    is_match = hmac.compare_digest(current, stored.anchor_digest)
    outcome = anchoring.record_verification(stored.chain_file_id, is_match)
    log_row = VerificationLog(stored_file_id=stored.id, verified_by_user_id=current_user.id,
                              is_match=is_match, notes=notes or None,
                              chain_status=outcome.status, chain_tx=outcome.tx_hash)
    db.session.add(log_row)
    db.session.commit()
    return {'success': True, 'is_match': is_match, 'file_record': _record(stored),
            'original_hash': stored.anchor_digest, 'current_hash': current,
            'verification_log_id': log_row.id, 'chain_status': outcome.status}


@file_verification_bp.route('/file/<int:file_id>')
@roles_required(*ALL_ROLES)
def view_file(file_id):
    stored = _get_authorised(file_id)
    if stored is None:
        flash('File not found.', 'error')
        return redirect(url_for('file_verification.list_files'))
    logs = [{'log_id': v.id, 'timestamp': int(v.created_at.timestamp()),
             'verified_by': v.verified_by.username if v.verified_by else '-',
             'is_match': v.is_match, 'notes': v.notes}
            for v in sorted(stored.verifications, key=lambda v: v.created_at, reverse=True)]
    return render_template('file_verification/view_file.html',
                           file_record=_record(stored), verification_logs=logs)


@file_verification_bp.route('/files')
@roles_required(*ALL_ROLES)
def list_files():
    rows = StoredFile.query.filter(StoredFile.erased_at.is_(None)).order_by(
        StoredFile.created_at.desc()).all()
    files = [_record(s) for s in rows if can_access_file(current_user, s)]
    return render_template('file_verification/list_files.html', files=files)


@file_verification_bp.route('/demo/tamper')
@roles_required(*ALL_ROLES)
def demo_tamper():
    """In-memory demo: nothing is written to disk or sent anywhere."""
    crypto = file_store.get_crypto()
    original = b"This is a demo medical record file for testing integrity verification."
    tampered = original + b"\x00"
    o_hash, t_hash = crypto.anchor_digest(original), crypto.anchor_digest(tampered)
    return render_template('file_verification/demo_tamper.html', demo_data={
        'original_file': 'demo_medical_record.txt (in memory)',
        'tampered_file': 'demo_medical_record.txt + 1 byte (in memory)',
        'original_hash': o_hash, 'tampered_hash': t_hash,
        'hashes_match': o_hash == t_hash})


@file_verification_bp.route('/download/<int:file_id>')
@roles_required(*ALL_ROLES)
def download_file(file_id):
    stored = _get_authorised(file_id)
    if stored is None:
        flash('File not found.', 'error')
        return redirect(url_for('file_verification.list_files'))
    try:
        data = file_store.read_plaintext(stored)
    except (FileNotFoundError, DecryptionError):
        flash('This file is no longer available.', 'error')
        return redirect(url_for('file_verification.list_files'))
    return send_file(io.BytesIO(data), mimetype='application/octet-stream',
                     as_attachment=True, download_name=stored.original_filename)


@file_verification_bp.route('/view-inline/<int:file_id>')
@roles_required(*ALL_ROLES)
def inline_image(file_id):
    """Authenticated image display for report pages (never cached, never public)."""
    stored = _get_authorised(file_id)
    if stored is None or stored.extension not in file_store.IMAGE_EXTENSIONS:
        return ('Not found', 404)
    try:
        data = file_store.read_plaintext(stored)
    except (FileNotFoundError, DecryptionError):
        return ('Not found', 404)
    return send_file(io.BytesIO(data), mimetype=file_store.mime_for(stored))


@file_verification_bp.route('/api/upload', methods=['POST'])
@roles_required(*UPLOADER_ROLES)
def api_upload_file():
    try:
        patient = db.session.get(Patient, int(request.form.get('patient_id', '')))
    except ValueError:
        return jsonify({'success': False, 'error': 'patient_id must be an integer'}), 400
    try:
        if patient is None:
            raise file_store.UploadRejected('You are not allowed to upload files for this patient.')
        stored = _do_upload(patient, request.files.get('file'),
                            request.form.get('description'),
                            request.form.get('category', 'medical'))
    except file_store.UploadRejected as exc:
        db.session.rollback()
        status = 403 if 'allowed' in str(exc) else 400
        return jsonify({'success': False, 'error': str(exc)}), status
    return jsonify({'success': True, 'file_id': stored.id,
                    'anchor_status': stored.anchor_status})


@file_verification_bp.route('/api/verify', methods=['POST'])
@roles_required(*ALL_ROLES)
def api_verify_file():
    try:
        file_id = int(request.form.get('file_id', ''))
    except ValueError:
        return jsonify({'success': False, 'error': 'file_id must be an integer'}), 400
    upload = request.files.get('file')
    if not upload:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    result = _verify(file_id, upload.read(file_store.MAX_BYTES + 1),
                     (request.form.get('notes') or '')[:500])
    if result is None:
        return jsonify({'success': False, 'error': 'File not found'}), 404
    return jsonify({'success': True, 'is_match': result['is_match'],
                    'file_id': file_id, 'verification_log_id': result['verification_log_id'],
                    'chain_status': result['chain_status']})
