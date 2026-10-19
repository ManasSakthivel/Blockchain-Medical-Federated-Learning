import logging
import re
import secrets as sec
from datetime import date, datetime

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required, logout_user

from app.access import current_patient, json_body
from app.models import (Consent, Consultation, Doctor, ErasureLog, Lab, LabReport,
                        LabRequest, MedicalRecord, Prescription, StoredFile, db)
from app.services import anchoring, file_store
from app.services.crypto_service import CryptoConfigError, purpose_hmac

log = logging.getLogger(__name__)
patient_bp = Blueprint('patient', __name__)

_ADDRESS_RE = re.compile(r'^0x[0-9a-fA-F]{40}$')
DATA_TYPES = ('lab_report', 'prescription', 'consultation', 'medical_record', 'file')
REQUEST_TYPES = ('retinal', 'blood', 'urine', 'xray', 'mri', 'other')
PRIORITIES = ('low', 'normal', 'high', 'urgent')


@patient_bp.before_request
def require_patient():
    if not current_user.is_authenticated or current_user.role != 'patient':
        flash('Access denied. Patient privileges required.', 'error')
        return redirect(url_for('main.home'))


def _patient_or_redirect():
    patient = current_patient()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
    return patient


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@patient_bp.route('/dashboard')
@login_required
def dashboard():
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))
    records = MedicalRecord.query.filter_by(patient_id=patient.id).all()
    consultations = Consultation.query.filter_by(patient_id=patient.id).all()
    return render_template('patient/dashboard.html', patient=patient, records=records,
                           consultations=consultations)


@patient_bp.route('/records')
@login_required
def records():
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))
    records = MedicalRecord.query.filter_by(patient_id=patient.id).all()
    return render_template('patient/records.html', records=records)


@patient_bp.route('/record/<int:record_id>')
@login_required
def view_record(record_id):
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))
    record = MedicalRecord.query.filter_by(id=record_id, patient_id=patient.id).first_or_404()

    # The related row is only shown if it belongs to THIS patient (record_id is
    # a bare integer, so it must never be trusted on its own).
    related_record = None
    model = {'consultation': Consultation, 'lab_report': LabReport,
             'prescription': Prescription}.get(record.record_type)
    if model is not None:
        candidate = db.session.get(model, record.record_id)
        if candidate is not None and candidate.patient_id == patient.id:
            related_record = candidate
    return render_template('patient/view_record.html', record=record,
                           related_record=related_record)


@patient_bp.route('/lab-reports')
@login_required
def lab_reports():
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))
    items = LabReport.query.filter_by(patient_id=patient.id).order_by(
        LabReport.created_at.desc()).all()
    return render_template('patient/lab_reports.html', patient=patient, lab_reports=items)


@patient_bp.route('/prescriptions')
@login_required
def prescriptions():
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))
    items = Prescription.query.filter_by(patient_id=patient.id).order_by(
        Prescription.prescribed_date.desc()).all()
    now = datetime.now()
    this_month = sum(1 for p in items if p.prescribed_date
                     and p.prescribed_date.month == now.month
                     and p.prescribed_date.year == now.year)
    return render_template('patient/prescriptions.html', patient=patient,
                           prescriptions=items, now=now, this_month_count=this_month)


@patient_bp.route('/lab-report/<int:report_id>')
@login_required
def view_lab_report(report_id):
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))
    lab_report = LabReport.query.filter_by(id=report_id, patient_id=patient.id).first()
    if not lab_report:
        flash('Lab report not found.', 'error')
        return redirect(url_for('patient.lab_reports'))
    return render_template('patient/view_lab_report.html', patient=patient,
                           lab_report=lab_report)


@patient_bp.route('/consultations')
@login_required
def consultations():
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))
    items = Consultation.query.filter_by(patient_id=patient.id).all()
    return render_template('patient/consultations.html', consultations=items)


@patient_bp.route('/consultation/<int:consultation_id>')
@login_required
def view_consultation(consultation_id):
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))
    consultation = Consultation.query.filter_by(
        id=consultation_id, patient_id=patient.id).first_or_404()
    return render_template('patient/view_consultation.html', consultation=consultation)


@patient_bp.route('/book-consultation', methods=['GET', 'POST'])
@login_required
def book_consultation():
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))

    if request.method == 'POST':
        doctor = db.session.get(Doctor, _to_int(request.form.get('doctor_id')) or 0)
        date_str = request.form.get('date')
        time_str = request.form.get('time')
        reason = (request.form.get('reason') or '').strip()
        if doctor is None or not date_str or not time_str or not reason:
            flash('Please select a doctor and fill in all required fields.', 'error')
            return redirect(url_for('patient.book_consultation'))
        try:
            when_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            when_time = datetime.strptime(time_str, '%H:%M').time()
        except ValueError:
            flash('Invalid date or time format.', 'error')
            return redirect(url_for('patient.book_consultation'))
        if when_date < date.today():
            flash('The consultation date cannot be in the past.', 'error')
            return redirect(url_for('patient.book_consultation'))

        # 'pending' until the doctor approves; only accepted consultations grant
        # the doctor access to this patient's record (app.access.TREATING_STATUSES).
        db.session.add(Consultation(patient_id=patient.id, doctor_id=doctor.id,
                                    date=when_date, time=when_time, reason=reason[:2000],
                                    status='pending'))
        db.session.commit()
        flash('Consultation requested. The doctor must accept it.', 'success')
        return redirect(url_for('patient.consultations'))

    doctors = Doctor.query.all()
    return render_template('patient/book_consultation.html', doctors=doctors)


@patient_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))

    if request.method == 'POST':
        dob_raw = (request.form.get('date_of_birth') or '').strip()
        dob = None
        if dob_raw:
            try:
                dob = datetime.strptime(dob_raw, '%Y-%m-%d').date()
            except ValueError:
                flash('Invalid date of birth. Use YYYY-MM-DD.', 'error')
                return redirect(url_for('patient.profile'))
            if dob > date.today():
                flash('Date of birth cannot be in the future.', 'error')
                return redirect(url_for('patient.profile'))
        patient.date_of_birth = dob
        patient.gender = (request.form.get('gender') or patient.gender)[:10]
        patient.blood_group = (request.form.get('blood_group') or '')[:5] or None
        patient.phone = (request.form.get('phone') or patient.phone)[:15]
        patient.address = request.form.get('address') or patient.address
        patient.emergency_contact = (request.form.get('emergency_contact')
                                     or patient.emergency_contact)[:15]
        patient.medical_history = request.form.get('medical_history')
        patient.allergies = request.form.get('allergies')
        patient.updated_at = datetime.utcnow()
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('patient.profile'))

    return render_template('patient/profile.html', patient=patient)


@patient_bp.route('/request-lab-report', methods=['GET', 'POST'])
@login_required
def request_lab_report():
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))

    if request.method == 'POST':
        doctor = db.session.get(Doctor, _to_int(request.form.get('doctor_id')) or 0)
        lab = db.session.get(Lab, _to_int(request.form.get('lab_id')) or 0)
        request_type = (request.form.get('request_type') or '').strip()
        reason = (request.form.get('reason') or '').strip()
        priority = request.form.get('priority', 'normal')
        consultation_id = None
        raw = (request.form.get('consultation_id') or '').strip()

        if doctor is None or lab is None or not lab.is_active or not reason \
                or request_type not in REQUEST_TYPES or priority not in PRIORITIES:
            flash('Please fill in all required fields with valid values.', 'error')
            return redirect(url_for('patient.request_lab_report'))
        if raw:  # A-01: the consultation must belong to THIS patient and doctor
            consultation_id = _to_int(raw)
            c = db.session.get(Consultation, consultation_id or 0)
            if c is None or c.patient_id != patient.id or c.doctor_id != doctor.id:
                flash('Invalid consultation selected.', 'error')
                return redirect(url_for('patient.request_lab_report'))

        db.session.add(LabRequest(patient_id=patient.id, doctor_id=doctor.id, lab_id=lab.id,
                                  consultation_id=consultation_id, request_type=request_type,
                                  reason=reason[:2000], priority=priority, status='pending'))
        db.session.commit()
        flash('Lab report request submitted successfully!', 'success')
        return redirect(url_for('patient.lab_requests'))

    doctors = Doctor.query.all()
    labs = Lab.query.filter_by(is_active=True).all()
    consultations = Consultation.query.filter_by(patient_id=patient.id, status='completed').all()
    return render_template('patient/request_lab_report.html', patient=patient,
                           doctors=doctors, labs=labs, consultations=consultations)


@patient_bp.route('/lab-requests')
@login_required
def lab_requests():
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))
    items = LabRequest.query.filter_by(patient_id=patient.id).order_by(
        LabRequest.created_at.desc()).all()
    return render_template('patient/lab_requests.html', patient=patient, lab_requests=items)


@patient_bp.route('/lab-request/<int:request_id>')
@login_required
def view_lab_request(request_id):
    patient = _patient_or_redirect()
    if not patient:
        return redirect(url_for('main.home'))
    lab_request = LabRequest.query.filter_by(id=request_id, patient_id=patient.id).first()
    if not lab_request:
        flash('Lab request not found.', 'error')
        return redirect(url_for('patient.lab_requests'))
    return render_template('patient/view_lab_request.html', patient=patient,
                           lab_request=lab_request)


# ---------------------------------------------------------------------------
# Consent (the database is authoritative; the chain is a best-effort anchor)
# ---------------------------------------------------------------------------

def _consent_json(c: Consent) -> dict:
    return {'consent_id': c.id, 'recipient': c.recipient, 'data_type': c.data_type,
            'purpose': c.purpose, 'purpose_hash': c.purpose_hash,
            'consent_status': c.status,
            'chain_status': c.chain_status, 'revoke_chain_status': c.revoke_chain_status}


@patient_bp.route('/consent', methods=['GET'])
@login_required
def consent_list():
    patient = current_patient()
    items = Consent.query.filter_by(patient_id=patient.id).order_by(Consent.id.desc()).all() \
        if patient else []
    return jsonify({'status': 'ok', 'consents': [_consent_json(c) for c in items]})


@patient_bp.route('/consent/grant', methods=['POST'])
@login_required
def consent_grant():
    """Record consent for ``recipient`` (an 0x address) to ``data_type``.

    Body: {"recipient": "0x..", "data_type": "lab_report", "purpose": "treatment"}
    The DB row is the source of truth. ``chain_status`` reports what the anchoring
    attempt really did: anchored | unavailable | failed | not_anchored.
    """
    patient = current_patient()
    if not patient:
        return jsonify({'status': 'error', 'message': 'Patient profile not found'}), 404
    data = json_body()
    recipient = str(data.get('recipient') or '').strip()
    data_type = str(data.get('data_type') or '').strip()
    purpose = str(data.get('purpose') or '').strip()

    if not recipient or not data_type or not purpose:
        return jsonify({'status': 'error',
                        'message': 'recipient, data_type, and purpose are required'}), 400
    if not _ADDRESS_RE.match(recipient):
        return jsonify({'status': 'error',
                        'message': 'recipient must be a 0x-prefixed 20-byte address'}), 400
    if data_type not in DATA_TYPES:
        return jsonify({'status': 'error',
                        'message': f'data_type must be one of {list(DATA_TYPES)}'}), 400
    if len(purpose) > 200:
        return jsonify({'status': 'error', 'message': 'purpose is too long'}), 400

    # F-03: use HMAC-SHA256(ANCHOR_HMAC_KEY, purpose) to prevent dictionary
    # attacks on low-entropy purpose strings (e.g. "treatment") on a public chain.
    # allow_ephemeral mirrors the file-encryption key behaviour: ephemeral in
    # test / dev mode, required (raises CryptoConfigError → 503) in production.
    _ephemeral = bool(current_app.config.get("TESTING") or current_app.config.get("EHR_DEV_MODE"))
    try:
        purpose_hash = purpose_hmac(purpose, allow_ephemeral=_ephemeral)
    except CryptoConfigError as exc:
        log.error("purpose_hmac: ANCHOR_HMAC_KEY not configured: %s", exc)
        return jsonify({'status': 'error',
                        'message': 'Consent anchoring key not configured. '
                                   'Set ANCHOR_HMAC_KEY and restart.'}), 503
    consent = Consent(patient_id=patient.id, recipient=recipient, data_type=data_type,
                      purpose=purpose, purpose_hash=purpose_hash, status='active')
    db.session.add(consent)
    db.session.commit()

    outcome = anchoring.anchor_consent(patient.id, current_user.did, recipient, data_type,
                                       purpose_hash)
    consent.chain_status = outcome.status
    consent.chain_consent_id, consent.chain_tx = outcome.object_id, outcome.tx_hash
    db.session.commit()
    return jsonify({'status': 'ok', 'did': current_user.did, **_consent_json(consent),
                    'on_chain': outcome.status == 'anchored'})


@patient_bp.route('/consent/revoke', methods=['POST'])
@login_required
def consent_revoke():
    """Revoke one of THIS patient's consents. Body: {"consent_id": <int>}"""
    patient = current_patient()
    if not patient:
        return jsonify({'status': 'error', 'message': 'Patient profile not found'}), 404
    consent_id = _to_int(json_body().get('consent_id'))
    if consent_id is None:
        return jsonify({'status': 'error', 'message': 'consent_id is required'}), 400

    consent = Consent.query.filter_by(id=consent_id, patient_id=patient.id).first()
    if consent is None:
        return jsonify({'status': 'error', 'message': 'Consent not found'}), 404
    if consent.status == 'revoked':
        return jsonify({'status': 'ok', **_consent_json(consent), 'on_chain': False})

    consent.status = 'revoked'
    consent.revoked_at = datetime.utcnow()
    db.session.commit()  # revocation is effective immediately, whatever the chain does

    outcome = anchoring.revoke_consent_anchor(consent.chain_consent_id)
    consent.revoke_chain_status, consent.revoke_chain_tx = outcome.status, outcome.tx_hash
    db.session.commit()
    return jsonify({'status': 'ok', **_consent_json(consent),
                    'on_chain': outcome.status == 'anchored'})


# ---------------------------------------------------------------------------
# Account erasure
# ---------------------------------------------------------------------------

@patient_bp.route('/erase-account', methods=['POST'])
@login_required
def erase_account():
    """Erase the patient's personal data held by this application.

    What this does:
      * deletes clinical rows (records, prescriptions, lab requests/reports,
        consultations, consents, verification logs);
      * crypto-shreds every stored file (wrapped key destroyed, local ciphertext
        deleted) so retained ciphertext copies can no longer be decrypted;
      * overwrites personal fields on the profile and user rows and disables login;
      * keeps an accountability row (ErasureLog) that references no person.

    What this does NOT do: it cannot remove data that was already written to a
    blockchain, and it cannot recall copies pinned on IPFS nodes outside this
    server. Only keyed digests / opaque ids were ever sent there (no PII, no
    plaintext). Back-ups are outside the scope of this endpoint.
    """
    patient = current_patient()
    if not patient:
        return jsonify({'status': 'error', 'message': 'Patient profile not found'}), 404
    if not current_user.check_password((json_body().get('password')
                                        or request.form.get('password') or '')):
        return jsonify({'status': 'error',
                        'message': 'Password confirmation is required to erase the account'}), 403

    summary = {'files_shredded': 0, 'ipfs_unpinned': 0, 'ipfs_unpin_failed': 0}
    for stored in StoredFile.query.filter_by(patient_id=patient.id).all():
        if stored.ipfs_cid and stored.ipfs_status == 'pinned':
            if anchoring.unpin(stored.ipfs_cid):
                summary['ipfs_unpinned'] += 1
                stored.ipfs_status = 'unpinned'
            else:
                summary['ipfs_unpin_failed'] += 1
        file_store.shred(stored)
        stored.original_filename = 'erased'
        stored.description = None
        stored.uploaded_by_user_id = None
        summary['files_shredded'] += 1

    from app.models import VerificationLog
    for stored in StoredFile.query.filter_by(patient_id=patient.id).all():
        VerificationLog.query.filter_by(stored_file_id=stored.id).delete()
    LabRequest.query.filter_by(patient_id=patient.id).update({'lab_report_id': None})
    MedicalRecord.query.filter_by(patient_id=patient.id).delete()
    Prescription.query.filter_by(patient_id=patient.id).delete()
    LabReport.query.filter_by(patient_id=patient.id).delete()
    LabRequest.query.filter_by(patient_id=patient.id).delete()
    Consultation.query.filter_by(patient_id=patient.id).delete()
    Consent.query.filter_by(patient_id=patient.id).delete()

    patient.first_name = patient.last_name = '[ERASED]'
    patient.date_of_birth = None
    patient.gender = '-'
    patient.phone = patient.emergency_contact = '-'
    patient.address = '[ERASED]'
    patient.blood_group = patient.allergies = patient.medical_history = None
    patient.erased_at = datetime.utcnow()

    token = sec.token_hex(8)
    current_user.email = f'erased_{token}@erased.invalid'
    current_user.username = f'erased_{token}'
    current_user.password_hash = '[ERASED]'  # nosec B105
    current_user.did = None  # the pseudonymous identifier is part of the personal data
    current_user.active = False
    current_user.rotate_session_token()

    erasure_ref = sec.token_hex(16)
    log_row = ErasureLog(erasure_ref=erasure_ref, summary=str(summary))
    db.session.add(log_row)
    db.session.commit()

    outcome = anchoring.record_erasure(erasure_ref)
    log_row.chain_status, log_row.chain_tx = outcome.status, outcome.tx_hash
    db.session.commit()

    logout_user()
    return jsonify({
        'status': 'ok',
        'erasure_ref': erasure_ref,
        'summary': summary,
        'chain_status': outcome.status,
        'message': ('Your account data held by this application was erased and your files '
                    'were crypto-shredded. Data already written to a blockchain cannot be '
                    'removed; only opaque identifiers and keyed digests were ever written.'),
    })
