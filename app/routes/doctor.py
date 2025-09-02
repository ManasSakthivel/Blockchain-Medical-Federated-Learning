from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from app.models import db, Doctor, Patient, Consultation, LabReport, Prescription, MedicalRecord, LabRequest
from datetime import datetime, date, timedelta

doctor_bp = Blueprint('doctor', __name__)

@doctor_bp.before_request
def require_doctor():
    if not current_user.is_authenticated or current_user.role != 'doctor':
        flash('Access denied. Doctor privileges required.', 'error')
        return redirect(url_for('main.home'))

@doctor_bp.route('/dashboard')
@login_required
def dashboard():
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    # Get today's consultations
    today = datetime.now().date()
    today_consultations = Consultation.query.filter_by(
        doctor_id=doctor.id, 
        date=today
    ).all()
    
    # Get pending consultations
    pending_consultations = Consultation.query.filter_by(
        doctor_id=doctor.id, 
        status='scheduled'
    ).order_by(Consultation.date).limit(10).all()
    
    # Get recent medical records for patients this doctor has consulted with
    patient_ids = [c.patient_id for c in Consultation.query.filter_by(doctor_id=doctor.id).all()]
    recent_records = MedicalRecord.query.filter(
        MedicalRecord.patient_id.in_(patient_ids)
    ).order_by(MedicalRecord.created_at.desc()).limit(5).all()
    
    return render_template('doctor/dashboard.html', 
                         doctor=doctor,
                         today_consultations=today_consultations,
                         pending_consultations=pending_consultations,
                         recent_records=recent_records)

@doctor_bp.route('/patients')
@login_required
def patients():
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    # Get all patients who have consulted with this doctor
    consultations = Consultation.query.filter_by(doctor_id=doctor.id).all()
    patient_ids = list(set([c.patient_id for c in consultations]))
    patients = Patient.query.filter(Patient.id.in_(patient_ids)).all()
    
    return render_template('doctor/patients.html', patients=patients)

@doctor_bp.route('/patient/<int:patient_id>')
@login_required
def view_patient(patient_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    patient = Patient.query.get_or_404(patient_id)
    records = MedicalRecord.query.filter_by(patient_id=patient_id).order_by(MedicalRecord.created_at.desc()).all()
    consultations = Consultation.query.filter_by(patient_id=patient_id, doctor_id=doctor.id).order_by(Consultation.date.desc()).all()
    
    return render_template('doctor/view_patient.html', 
                         patient=patient, 
                         records=records, 
                         consultations=consultations,
                         current_date=date.today())

@doctor_bp.route('/patient/<int:patient_id>/medical-history')
@login_required
def patient_medical_history(patient_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    # Get patient
    patient = Patient.query.get_or_404(patient_id)
    
    # Verify doctor has consulted with this patient
    consultation_exists = Consultation.query.filter_by(
        doctor_id=doctor.id, 
        patient_id=patient.id
    ).first()
    
    if not consultation_exists:
        flash('Access denied. You can only view medical history of your patients.', 'error')
        return redirect(url_for('doctor.patients'))
    
    # Get patient's complete medical history
    consultations = Consultation.query.filter_by(patient_id=patient.id).order_by(Consultation.date.desc()).all()
    lab_reports = LabReport.query.filter_by(patient_id=patient.id).order_by(LabReport.created_at.desc()).all()
    prescriptions = Prescription.query.filter_by(patient_id=patient.id).order_by(Prescription.prescribed_date.desc()).all()
    medical_records = MedicalRecord.query.filter_by(patient_id=patient.id).order_by(MedicalRecord.date.desc()).all()
    
    return render_template('doctor/patient_medical_history.html', 
                         doctor=doctor, 
                         patient=patient,
                         consultations=consultations,
                         lab_reports=lab_reports,
                         prescriptions=prescriptions,
                         medical_records=medical_records)

@doctor_bp.route('/patient/<int:patient_id>/record/new', methods=['GET', 'POST'])
@login_required
def create_record(patient_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    patient = Patient.query.get_or_404(patient_id)
    
    if request.method == 'POST':
        record_type = request.form.get('record_type')
        title = request.form.get('title')
        description = request.form.get('description')
        
        record = MedicalRecord(
            patient_id=patient_id,
            record_type=record_type,
            record_id=0,  # This would need to be set based on the actual record being referenced
            title=title,
            description=description,
            date=datetime.now().date()
        )
        
        db.session.add(record)
        db.session.commit()
        
        flash('Medical record created successfully!', 'success')
        return redirect(url_for('doctor.view_patient', patient_id=patient_id))
    
    return render_template('doctor/create_record.html', patient=patient)

@doctor_bp.route('/record/<int:record_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_record(record_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    record = MedicalRecord.query.get_or_404(record_id)
    
    # Verify this record belongs to a patient the doctor has consulted with
    consultation_exists = Consultation.query.filter_by(
        doctor_id=doctor.id, 
        patient_id=record.patient_id
    ).first()
    
    if not consultation_exists:
        flash('Access denied. You can only edit records for your patients.', 'error')
        return redirect(url_for('doctor.patients'))
    
    if request.method == 'POST':
        record.record_type = request.form.get('record_type')
        record.title = request.form.get('title')
        record.description = request.form.get('description')
        
        db.session.commit()
        flash('Medical record updated successfully!', 'success')
        return redirect(url_for('doctor.view_patient', patient_id=record.patient_id))
    
    return render_template('doctor/edit_record.html', record=record)

@doctor_bp.route('/consultations')
@login_required
def consultations():
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    consultations = Consultation.query.filter_by(doctor_id=doctor.id).order_by(Consultation.date.desc()).all()
    return render_template('doctor/consultations.html', consultations=consultations)

@doctor_bp.route('/consultations/<int:consultation_id>/start', methods=['POST'])
@login_required
def start_consultation(consultation_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        return jsonify({'success': False, 'message': 'Doctor profile not found'})
    
    consultation = Consultation.query.filter_by(id=consultation_id, doctor_id=doctor.id).first()
    if not consultation:
        return jsonify({'success': False, 'message': 'Consultation not found'})
    
    consultation.status = 'in_progress'
    consultation.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Consultation started successfully'})

@doctor_bp.route('/consultations/<int:consultation_id>/complete', methods=['POST'])
@login_required
def complete_consultation(consultation_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        return jsonify({'success': False, 'message': 'Doctor profile not found'})
    
    consultation = Consultation.query.filter_by(id=consultation_id, doctor_id=doctor.id).first()
    if not consultation:
        return jsonify({'success': False, 'message': 'Consultation not found'})
    
    consultation.status = 'completed'
    consultation.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Consultation completed successfully'})

@doctor_bp.route('/consultation/<int:consultation_id>')
@login_required
def view_consultation(consultation_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    consultation = Consultation.query.filter_by(id=consultation_id, doctor_id=doctor.id).first_or_404()
    return render_template('doctor/view_consultation.html', consultation=consultation)

@doctor_bp.route('/consultation/<int:consultation_id>/update', methods=['POST'])
@login_required
def update_consultation(consultation_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    consultation = Consultation.query.filter_by(id=consultation_id, doctor_id=doctor.id).first_or_404()
    
    # Get form data
    diagnosis = request.form.get('diagnosis')
    treatment_plan = request.form.get('treatment_plan')
    notes = request.form.get('notes')
    medication_name = request.form.get('medication_name')
    dosage = request.form.get('dosage')
    frequency = request.form.get('frequency')
    duration = request.form.get('duration')
    
    # Update consultation
    consultation.diagnosis = diagnosis
    consultation.treatment_plan = treatment_plan
    consultation.notes = notes
    consultation.updated_at = datetime.utcnow()
    
    # Create prescription if medication details are provided
    if medication_name and dosage and frequency and duration:
        prescription = Prescription(
            patient_id=consultation.patient_id,
            doctor_id=doctor.id,
            consultation_id=consultation.id,
            medication_name=medication_name,
            dosage=dosage,
            frequency=frequency,
            duration=duration,
            instructions=notes,
            prescribed_date=datetime.now().date(),
            is_active=True
        )
        db.session.add(prescription)
    
    db.session.commit()
    flash('Consultation updated successfully!' + (' Prescription created.' if medication_name else ''), 'success')
    return redirect(url_for('doctor.view_consultation', consultation_id=consultation_id))

@doctor_bp.route('/schedule')
@login_required
def schedule():
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    # Get today's consultations
    today = datetime.now().date()
    today_consultations = Consultation.query.filter_by(
        doctor_id=doctor.id,
        date=today
    ).order_by(Consultation.time).all()
    
    # Get week's consultations (next 7 days)
    week_start = today
    week_end = today + timedelta(days=7)
    week_consultations = Consultation.query.filter(
        Consultation.doctor_id == doctor.id,
        Consultation.date >= week_start,
        Consultation.date <= week_end
    ).order_by(Consultation.date, Consultation.time).all()
    
    # Get pending consultation requests
    pending_requests = Consultation.query.filter_by(
        doctor_id=doctor.id,
        status='pending'
    ).order_by(Consultation.created_at).all()
    
    # Calculate available slots (simplified - assuming 8 hours per day)
    available_slots = 8 * 7  # 8 hours per day for 7 days
    
    # Create schedule_data structure
    schedule_data = {
        'today_appointments': today_consultations,
        'week_appointments': week_consultations,
        'available_slots': available_slots,
        'pending_requests': pending_requests
    }
    
    return render_template('doctor/schedule.html', 
                         doctor=doctor,
                         schedule_data=schedule_data)

@doctor_bp.route('/schedule/availability', methods=['POST'])
@login_required
def set_availability():
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        return jsonify({'success': False, 'message': 'Doctor profile not found'})
    
    is_available = request.json.get('is_available', False) if request.json else False
    doctor.is_available = is_available
    doctor.updated_at = datetime.utcnow()
    
    db.session.commit()
    
    return jsonify({
        'success': True, 
        'message': 'Availability updated successfully',
        'is_available': doctor.is_available
    })

@doctor_bp.route('/schedule/requests/<int:request_id>/approve', methods=['POST'])
@login_required
def approve_request(request_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        return jsonify({'success': False, 'message': 'Doctor profile not found'})
    
    consultation = Consultation.query.filter_by(id=request_id, doctor_id=doctor.id).first()
    if not consultation:
        return jsonify({'success': False, 'message': 'Consultation request not found'})
    
    consultation.status = 'scheduled'
    consultation.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Consultation request approved'})

@doctor_bp.route('/schedule/requests/<int:request_id>/reject', methods=['POST'])
@login_required
def reject_request(request_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        return jsonify({'success': False, 'message': 'Doctor profile not found'})
    
    consultation = Consultation.query.filter_by(id=request_id, doctor_id=doctor.id).first()
    if not consultation:
        return jsonify({'success': False, 'message': 'Consultation request not found'})
    
    consultation.status = 'cancelled'
    consultation.updated_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Consultation request rejected'})

@doctor_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    if request.method == 'POST':
        doctor.first_name = request.form.get('first_name')
        doctor.last_name = request.form.get('last_name')
        doctor.specialization = request.form.get('specialization')
        doctor.phone = request.form.get('phone')
        doctor.address = request.form.get('address')
        doctor.experience_years = int(request.form.get('experience_years', 0))
        doctor.education = request.form.get('education')
        doctor.consultation_fee = float(request.form.get('consultation_fee', 0))
        doctor.updated_at = datetime.utcnow()
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('doctor.profile'))
    
    return render_template('doctor/profile.html', doctor=doctor)

@doctor_bp.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        return jsonify({'success': False, 'message': 'Doctor profile not found'})
    
    try:
        doctor.first_name = request.json.get('first_name')
        doctor.last_name = request.json.get('last_name')
        doctor.specialization = request.json.get('specialization')
        doctor.phone = request.json.get('phone')
        doctor.address = request.json.get('address')
        doctor.experience_years = int(request.json.get('experience_years', 0))
        doctor.education = request.json.get('education')
        doctor.consultation_fee = float(request.json.get('consultation_fee', 0))
        doctor.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'doctor': {
                'first_name': doctor.first_name,
                'last_name': doctor.last_name,
                'specialization': doctor.specialization,
                'phone': doctor.phone,
                'address': doctor.address,
                'experience_years': doctor.experience_years,
                'education': doctor.education,
                'consultation_fee': doctor.consultation_fee
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error updating profile: {str(e)}'})

@doctor_bp.route('/profile/change-password', methods=['POST'])
@login_required
def change_password():
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        return jsonify({'success': False, 'message': 'Doctor profile not found'})
    
    current_password = request.json.get('current_password')
    new_password = request.json.get('new_password')
    confirm_password = request.json.get('confirm_password')
    
    if not current_user.check_password(current_password):
        return jsonify({'success': False, 'message': 'Current password is incorrect'})
    
    if new_password != confirm_password:
        return jsonify({'success': False, 'message': 'New passwords do not match'})
    
    if len(new_password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters long'})
    
    current_user.set_password(new_password)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Password changed successfully'})

@doctor_bp.route('/lab-reports')
@login_required
def lab_reports():
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    # Get all lab reports for patients this doctor has consulted with
    consultations = Consultation.query.filter_by(doctor_id=doctor.id).all()
    patient_ids = list(set([c.patient_id for c in consultations]))
    lab_reports = LabReport.query.filter(
        LabReport.patient_id.in_(patient_ids)
    ).order_by(LabReport.created_at.desc()).all()
    
    return render_template('doctor/lab_reports.html', lab_reports=lab_reports, doctor=doctor)

@doctor_bp.route('/lab-report/<int:report_id>')
@login_required
def view_lab_report(report_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    lab_report = LabReport.query.get_or_404(report_id)
    
    # Verify this report belongs to a patient the doctor has consulted with
    consultation_exists = Consultation.query.filter_by(
        doctor_id=doctor.id, 
        patient_id=lab_report.patient_id
    ).first()
    
    if not consultation_exists:
        flash('Access denied. You can only view lab reports for your patients.', 'error')
        return redirect(url_for('doctor.lab_reports'))
    
    return render_template('doctor/view_lab_report.html', lab_report=lab_report, doctor=doctor)

@doctor_bp.route('/prescriptions')
@login_required
def prescriptions():
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    # Get all prescriptions for patients this doctor has consulted with
    consultations = Consultation.query.filter_by(doctor_id=doctor.id).all()
    patient_ids = list(set([c.patient_id for c in consultations]))
    prescriptions = Prescription.query.filter(
        Prescription.patient_id.in_(patient_ids)
    ).order_by(Prescription.prescribed_date.desc()).all()
    
    return render_template('doctor/prescriptions.html', prescriptions=prescriptions, doctor=doctor)

@doctor_bp.route('/prescription/<int:prescription_id>')
@login_required
def view_prescription(prescription_id):
    doctor = Doctor.query.filter_by(user_id=current_user.id).first()
    if not doctor:
        flash('Doctor profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    prescription = Prescription.query.get_or_404(prescription_id)
    
    # Verify this prescription belongs to a patient the doctor has consulted with
    consultation_exists = Consultation.query.filter_by(
        doctor_id=doctor.id, 
        patient_id=prescription.patient_id
    ).first()
    
    if not consultation_exists:
        flash('Access denied. You can only view prescriptions for your patients.', 'error')
        return redirect(url_for('doctor.prescriptions'))
    
    return render_template('doctor/view_prescription.html', prescription=prescription, doctor=doctor) 