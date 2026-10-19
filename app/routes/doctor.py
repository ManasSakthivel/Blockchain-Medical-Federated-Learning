from datetime import date, datetime, timedelta

from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user

from app.access import (current_doctor, doctor_has_access, json_body,
                        password_problem, patient_ids_for_doctor)
from app.models import (Consultation, LabReport, MedicalRecord, Patient,
                        Prescription, db)

doctor_bp = Blueprint('doctor', __name__)

RECORD_TYPES = ('lab_report', 'prescription', 'consultation', 'note')


@doctor_bp.before_request
def require_doctor():
    if not current_user.is_authenticated or current_user.role != 'doctor':
        flash('Access denied. Doctor privileges required.', 'error')
        return redirect(url_for('main.home'))


def _no_profile_redirect():
    flash('Doctor profile not found. Please contact support.', 'error')
    return redirect(url_for('main.home'))


def _api_error(message, status=400):
    return jsonify({'success': False, 'message': message}), status


def _denied_patient():
    flash('Access denied. You can only access patients you are treating.', 'error')
    return redirect(url_for('doctor.patients'))


@doctor_bp.route('/dashboard')
@login_required
def dashboard():
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()

    today = datetime.now().date()
    today_consultations = Consultation.query.filter_by(doctor_id=doctor.id, date=today).all()
    pending_consultations = Consultation.query.filter_by(
        doctor_id=doctor.id, status='scheduled'
    ).order_by(Consultation.date).limit(10).all()

    patient_ids = patient_ids_for_doctor(doctor.id)
    recent_records = MedicalRecord.query.filter(
        MedicalRecord.patient_id.in_(patient_ids)
    ).order_by(MedicalRecord.created_at.desc()).limit(5).all() if patient_ids else []

    return render_template('doctor/dashboard.html', doctor=doctor,
                           today_consultations=today_consultations,
                           pending_consultations=pending_consultations,
                           recent_records=recent_records)


@doctor_bp.route('/patients')
@login_required
def patients():
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    ids = patient_ids_for_doctor(doctor.id)
    patients = Patient.query.filter(Patient.id.in_(ids)).all() if ids else []
    return render_template('doctor/patients.html', patients=patients)


@doctor_bp.route('/patient/<int:patient_id>')
@login_required
def view_patient(patient_id):
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    if not doctor_has_access(doctor.id, patient_id):
        return _denied_patient()

    patient = db.get_or_404(Patient, patient_id)
    records = MedicalRecord.query.filter_by(patient_id=patient_id).order_by(
        MedicalRecord.created_at.desc()).all()
    consultations = Consultation.query.filter_by(
        patient_id=patient_id, doctor_id=doctor.id).order_by(Consultation.date.desc()).all()
    return render_template('doctor/view_patient.html', patient=patient, records=records,
                           consultations=consultations, current_date=date.today())


@doctor_bp.route('/patient/<int:patient_id>/medical-history')
@login_required
def patient_medical_history(patient_id):
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    if not doctor_has_access(doctor.id, patient_id):
        return _denied_patient()

    patient = db.get_or_404(Patient, patient_id)
    consultations = Consultation.query.filter_by(patient_id=patient.id).order_by(
        Consultation.date.desc()).all()
    lab_reports = LabReport.query.filter_by(patient_id=patient.id).order_by(
        LabReport.created_at.desc()).all()
    prescriptions = Prescription.query.filter_by(patient_id=patient.id).order_by(
        Prescription.prescribed_date.desc()).all()
    medical_records = MedicalRecord.query.filter_by(patient_id=patient.id).order_by(
        MedicalRecord.date.desc()).all()
    return render_template('doctor/patient_medical_history.html', doctor=doctor,
                           patient=patient, consultations=consultations,
                           lab_reports=lab_reports, prescriptions=prescriptions,
                           medical_records=medical_records)


@doctor_bp.route('/patient/<int:patient_id>/record/new', methods=['GET', 'POST'])
@login_required
def create_record(patient_id):
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    if not doctor_has_access(doctor.id, patient_id):
        return _denied_patient()
    patient = db.get_or_404(Patient, patient_id)

    if request.method == 'POST':
        record_type = (request.form.get('record_type') or '').strip()
        title = (request.form.get('title') or '').strip()
        if record_type not in RECORD_TYPES or not title or len(title) > 100:
            flash('A valid record type and a title (max 100 characters) are required.', 'error')
            return redirect(url_for('doctor.create_record', patient_id=patient_id))

        db.session.add(MedicalRecord(
            patient_id=patient_id, record_type=record_type,
            record_id=0,  # free-form note: not linked to another table row
            title=title, description=request.form.get('description'),
            date=datetime.now().date()))
        db.session.commit()
        flash('Medical record created successfully!', 'success')
        return redirect(url_for('doctor.view_patient', patient_id=patient_id))

    return render_template('doctor/create_record.html', patient=patient)


@doctor_bp.route('/record/<int:record_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_record(record_id):
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    record = db.get_or_404(MedicalRecord, record_id)
    if not doctor_has_access(doctor.id, record.patient_id):
        return _denied_patient()

    if request.method == 'POST':
        record_type = (request.form.get('record_type') or '').strip()
        title = (request.form.get('title') or '').strip()
        if record_type not in RECORD_TYPES or not title or len(title) > 100:
            flash('A valid record type and a title (max 100 characters) are required.', 'error')
            return redirect(url_for('doctor.edit_record', record_id=record_id))
        record.record_type = record_type
        record.title = title
        record.description = request.form.get('description')
        db.session.commit()
        flash('Medical record updated successfully!', 'success')
        return redirect(url_for('doctor.view_patient', patient_id=record.patient_id))

    return render_template('doctor/edit_record.html', record=record)


@doctor_bp.route('/consultations')
@login_required
def consultations():
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    consultations = Consultation.query.filter_by(doctor_id=doctor.id).order_by(
        Consultation.date.desc()).all()
    return render_template('doctor/consultations.html', consultations=consultations)


def _transition(consultation_id, allowed_from, new_status, ok_message):
    doctor = current_doctor()
    if not doctor:
        return _api_error('Doctor profile not found', 404)
    consultation = Consultation.query.filter_by(id=consultation_id, doctor_id=doctor.id).first()
    if not consultation:
        return _api_error('Consultation not found', 404)
    if consultation.status not in allowed_from:
        return _api_error(f"Cannot change a '{consultation.status}' consultation this way.", 409)
    consultation.status = new_status
    consultation.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True, 'message': ok_message})


@doctor_bp.route('/consultations/<int:consultation_id>/start', methods=['POST'])
@login_required
def start_consultation(consultation_id):
    return _transition(consultation_id, ('scheduled',), 'in_progress',
                       'Consultation started successfully')


@doctor_bp.route('/consultations/<int:consultation_id>/complete', methods=['POST'])
@login_required
def complete_consultation(consultation_id):
    return _transition(consultation_id, ('scheduled', 'in_progress', 'in-progress'),
                       'completed', 'Consultation completed successfully')


@doctor_bp.route('/consultation/<int:consultation_id>')
@login_required
def view_consultation(consultation_id):
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    consultation = Consultation.query.filter_by(
        id=consultation_id, doctor_id=doctor.id).first_or_404()
    return render_template('doctor/view_consultation.html', consultation=consultation)


@doctor_bp.route('/consultation/<int:consultation_id>/update', methods=['POST'])
@login_required
def update_consultation(consultation_id):
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    consultation = Consultation.query.filter_by(
        id=consultation_id, doctor_id=doctor.id).first_or_404()
    if consultation.status not in ('scheduled', 'in_progress', 'in-progress', 'completed'):
        flash('This consultation cannot be edited.', 'error')
        return redirect(url_for('doctor.consultations'))

    notes = request.form.get('notes')
    consultation.diagnosis = request.form.get('diagnosis')
    consultation.treatment_plan = request.form.get('treatment_plan')
    consultation.notes = notes
    consultation.updated_at = datetime.utcnow()

    medication_name = (request.form.get('medication_name') or '').strip()
    dosage = (request.form.get('dosage') or '').strip()
    frequency = (request.form.get('frequency') or '').strip()
    duration = (request.form.get('duration') or '').strip()
    created = False
    if medication_name and dosage and frequency and duration:
        db.session.add(Prescription(
            patient_id=consultation.patient_id, doctor_id=doctor.id,
            consultation_id=consultation.id, medication_name=medication_name[:100],
            dosage=dosage[:50], frequency=frequency[:50], duration=duration[:50],
            instructions=notes, prescribed_date=datetime.now().date(), is_active=True))
        created = True

    db.session.commit()
    flash('Consultation updated successfully!' + (' Prescription created.' if created else ''),
          'success')
    return redirect(url_for('doctor.view_consultation', consultation_id=consultation_id))


@doctor_bp.route('/schedule')
@login_required
def schedule():
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()

    today = datetime.now().date()
    today_consultations = Consultation.query.filter_by(
        doctor_id=doctor.id, date=today).order_by(Consultation.time).all()
    week_consultations = Consultation.query.filter(
        Consultation.doctor_id == doctor.id,
        Consultation.date >= today,
        Consultation.date <= today + timedelta(days=7),
    ).order_by(Consultation.date, Consultation.time).all()
    pending_requests = Consultation.query.filter_by(
        doctor_id=doctor.id, status='pending').order_by(Consultation.created_at).all()

    schedule_data = {
        'today_appointments': today_consultations,
        'week_appointments': week_consultations,
        'available_slots': 8 * 7,  # simplified: 8 h/day for 7 days
        'pending_requests': pending_requests,
    }
    return render_template('doctor/schedule.html', doctor=doctor, schedule_data=schedule_data)


@doctor_bp.route('/schedule/availability', methods=['POST'])
@login_required
def set_availability():
    doctor = current_doctor()
    if not doctor:
        return _api_error('Doctor profile not found', 404)
    doctor.is_available = bool(json_body().get('is_available', False))
    doctor.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'success': True, 'message': 'Availability updated successfully',
                    'is_available': doctor.is_available})


@doctor_bp.route('/schedule/requests/<int:request_id>/approve', methods=['POST'])
@login_required
def approve_request(request_id):
    return _transition(request_id, ('pending',), 'scheduled', 'Consultation request approved')


@doctor_bp.route('/schedule/requests/<int:request_id>/reject', methods=['POST'])
@login_required
def reject_request(request_id):
    return _transition(request_id, ('pending',), 'cancelled', 'Consultation request rejected')


def _apply_profile(doctor, data):
    """Validate and apply profile fields from a mapping. Returns an error string or None."""
    try:
        experience = int(data.get('experience_years') or 0)
        fee = float(data.get('consultation_fee') or 0)
    except (TypeError, ValueError):
        return 'Experience and fee must be numbers.'
    if not 0 <= experience <= 70 or not 0 <= fee <= 1_000_000:
        return 'Experience or fee is out of range.'
    first = (data.get('first_name') or '').strip()
    last = (data.get('last_name') or '').strip()
    if not first or not last:
        return 'First and last name are required.'
    doctor.first_name, doctor.last_name = first[:50], last[:50]
    doctor.specialization = (data.get('specialization') or doctor.specialization)[:100]
    doctor.phone = (data.get('phone') or doctor.phone)[:15]
    doctor.address = data.get('address') or doctor.address
    doctor.experience_years = experience
    doctor.education = data.get('education') or doctor.education
    doctor.consultation_fee = fee
    doctor.updated_at = datetime.utcnow()
    return None


@doctor_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    if request.method == 'POST':
        error = _apply_profile(doctor, request.form)
        if error:
            db.session.rollback()
            flash(error, 'error')
            return redirect(url_for('doctor.profile'))
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('doctor.profile'))
    return render_template('doctor/profile.html', doctor=doctor)


@doctor_bp.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    doctor = current_doctor()
    if not doctor:
        return _api_error('Doctor profile not found', 404)
    error = _apply_profile(doctor, json_body())
    if error:
        db.session.rollback()
        return _api_error(error)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Profile updated successfully', 'doctor': {
        'first_name': doctor.first_name, 'last_name': doctor.last_name,
        'specialization': doctor.specialization, 'phone': doctor.phone,
        'address': doctor.address, 'experience_years': doctor.experience_years,
        'education': doctor.education, 'consultation_fee': doctor.consultation_fee}})


@doctor_bp.route('/profile/change-password', methods=['POST'])
@login_required
def change_password():
    data = json_body()
    current_password = data.get('current_password') or ''
    new_password = data.get('new_password') or ''
    confirm = data.get('confirm_password')

    if not current_user.check_password(current_password):
        return _api_error('Current password is incorrect', 403)
    if confirm is not None and new_password != confirm:
        return _api_error('New passwords do not match')
    problem = password_problem(new_password)
    if problem:
        return _api_error(problem)

    current_user.set_password(new_password)  # also rotates the session token
    db.session.commit()
    login_user(current_user)  # keep THIS session alive; all other sessions are now invalid
    return jsonify({'success': True, 'message': 'Password changed successfully'})


@doctor_bp.route('/lab-reports')
@login_required
def lab_reports():
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    ids = patient_ids_for_doctor(doctor.id)
    reports = LabReport.query.filter(LabReport.patient_id.in_(ids)).order_by(
        LabReport.created_at.desc()).all() if ids else []
    return render_template('doctor/lab_reports.html', lab_reports=reports, doctor=doctor)


@doctor_bp.route('/lab-report/<int:report_id>')
@login_required
def view_lab_report(report_id):
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    lab_report = db.get_or_404(LabReport, report_id)
    if not doctor_has_access(doctor.id, lab_report.patient_id):
        flash('Access denied. You can only view lab reports for your patients.', 'error')
        return redirect(url_for('doctor.lab_reports'))
    return render_template('doctor/view_lab_report.html', lab_report=lab_report, doctor=doctor)


@doctor_bp.route('/prescriptions')
@login_required
def prescriptions():
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    ids = patient_ids_for_doctor(doctor.id)
    items = Prescription.query.filter(Prescription.patient_id.in_(ids)).order_by(
        Prescription.prescribed_date.desc()).all() if ids else []
    return render_template('doctor/prescriptions.html', prescriptions=items, doctor=doctor)


@doctor_bp.route('/prescription/<int:prescription_id>')
@login_required
def view_prescription(prescription_id):
    doctor = current_doctor()
    if not doctor:
        return _no_profile_redirect()
    prescription = db.get_or_404(Prescription, prescription_id)
    if not doctor_has_access(doctor.id, prescription.patient_id):
        flash('Access denied. You can only view prescriptions for your patients.', 'error')
        return redirect(url_for('doctor.prescriptions'))
    return render_template('doctor/view_prescription.html', prescription=prescription,
                           doctor=doctor)
