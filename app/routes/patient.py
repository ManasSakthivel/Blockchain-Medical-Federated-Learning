from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from app.models import db, Patient, MedicalRecord, Consultation, LabReport, Prescription, LabRequest, Lab
from datetime import datetime, date, time

patient_bp = Blueprint('patient', __name__)

@patient_bp.before_request
def require_patient():
    if not current_user.is_authenticated or current_user.role != 'patient':
        flash('Access denied. Patient privileges required.', 'error')
        return redirect(url_for('main.home'))

@patient_bp.route('/dashboard')
@login_required
def dashboard():
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    records = MedicalRecord.query.filter_by(patient_id=patient.id).all()
    consultations = Consultation.query.filter_by(patient_id=patient.id).all()
    return render_template('patient/dashboard.html', patient=patient, records=records, consultations=consultations)

@patient_bp.route('/records')
@login_required
def records():
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    records = MedicalRecord.query.filter_by(patient_id=patient.id).all()
    return render_template('patient/records.html', records=records)

@patient_bp.route('/record/<int:record_id>')
@login_required
def view_record(record_id):
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    record = MedicalRecord.query.filter_by(id=record_id, patient_id=patient.id).first_or_404()

    # Fetch the related record based on type
    related_record = None
    if record.record_type == 'consultation':
        from app.models import Consultation
        related_record = Consultation.query.get(record.record_id)
    elif record.record_type == 'lab_report':
        from app.models import LabReport
        related_record = LabReport.query.get(record.record_id)
    elif record.record_type == 'prescription':
        from app.models import Prescription
        related_record = Prescription.query.get(record.record_id)

    return render_template('patient/view_record.html', record=record, related_record=related_record)

@patient_bp.route('/lab-reports')
@login_required
def lab_reports():
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    # Get patient's lab reports
    lab_reports = LabReport.query.filter_by(patient_id=patient.id).order_by(LabReport.created_at.desc()).all()
    
    return render_template('patient/lab_reports.html', patient=patient, lab_reports=lab_reports)

@patient_bp.route('/prescriptions')
@login_required
def prescriptions():
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    prescriptions = Prescription.query.filter_by(patient_id=patient.id).order_by(Prescription.prescribed_date.desc()).all()
    from datetime import datetime
    now = datetime.now()
    this_month_count = sum(
        1 for p in prescriptions
        if p.prescribed_date.month == now.month and p.prescribed_date.year == now.year
    )
    return render_template('patient/prescriptions.html', patient=patient, prescriptions=prescriptions, now=now, this_month_count=this_month_count)

@patient_bp.route('/lab-report/<int:report_id>')
@login_required
def view_lab_report(report_id):
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    # Get lab report (only if it belongs to this patient)
    lab_report = LabReport.query.filter_by(id=report_id, patient_id=patient.id).first()
    if not lab_report:
        flash('Lab report not found.', 'error')
        return redirect(url_for('patient.lab_reports'))
    
    return render_template('patient/view_lab_report.html', patient=patient, lab_report=lab_report)

@patient_bp.route('/consultations')
@login_required
def consultations():
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    consultations = Consultation.query.filter_by(patient_id=patient.id).all()
    return render_template('patient/consultations.html', consultations=consultations)

@patient_bp.route('/consultation/<int:consultation_id>')
@login_required
def view_consultation(consultation_id):
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    consultation = Consultation.query.filter_by(id=consultation_id, patient_id=patient.id).first_or_404()
    return render_template('patient/view_consultation.html', consultation=consultation)

@patient_bp.route('/book-consultation', methods=['GET', 'POST'])
@login_required
def book_consultation():
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    if request.method == 'POST':
        doctor_id = request.form.get('doctor_id')
        date_str = request.form.get('date')
        time_str = request.form.get('time')
        reason = request.form.get('reason')
        
        # Validate required fields
        if not date_str or not time_str or not reason:
            flash('Please fill in all required fields.', 'error')
            return redirect(url_for('patient.book_consultation'))
        
        # Convert date string to Python date object
        try:
            consultation_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format. Please use YYYY-MM-DD format.', 'error')
            return redirect(url_for('patient.book_consultation'))
        
        # Convert time string to Python time object
        try:
            consultation_time = datetime.strptime(time_str, '%H:%M').time()
        except ValueError:
            flash('Invalid time format. Please use HH:MM format.', 'error')
            return redirect(url_for('patient.book_consultation'))
        
        consultation = Consultation(
            patient_id=patient.id,
            doctor_id=doctor_id,
            date=consultation_date,
            time=consultation_time,
            reason=reason,
            status='scheduled'
        )
        
        db.session.add(consultation)
        db.session.commit()
        
        flash('Consultation booked successfully!', 'success')
        return redirect(url_for('patient.consultations'))
    
    # Get available doctors
    from app.models import Doctor
    doctors = Doctor.query.all()
    return render_template('patient/book_consultation.html', doctors=doctors)

@patient_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    if request.method == 'POST':
        date_of_birth_str = request.form.get('date_of_birth')
        if date_of_birth_str:
            try:
                patient.date_of_birth = datetime.strptime(date_of_birth_str, '%Y-%m-%d').date()
            except ValueError:
                flash('Invalid date format for date of birth. Please use YYYY-MM-DD format.', 'error')
                return redirect(url_for('patient.profile'))
        else:
            patient.date_of_birth = None
            
        patient.gender = request.form.get('gender')
        patient.blood_group = request.form.get('blood_group')
        patient.phone = request.form.get('phone')
        patient.address = request.form.get('address')
        patient.emergency_contact = request.form.get('emergency_contact')
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
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    if request.method == 'POST':
        doctor_id = request.form.get('doctor_id')
        lab_id = request.form.get('lab_id')
        request_type = request.form.get('request_type')
        reason = request.form.get('reason')
        priority = request.form.get('priority', 'normal')
        consultation_id = request.form.get('consultation_id')
        
        # Validate required fields
        if not all([doctor_id, lab_id, request_type, reason]):
            flash('Please fill in all required fields.', 'error')
            return redirect(url_for('patient.request_lab_report'))
        
        # Create lab request
        lab_request = LabRequest(
            patient_id=patient.id,
            doctor_id=doctor_id,
            lab_id=lab_id,
            consultation_id=consultation_id if consultation_id else None,
            request_type=request_type,
            reason=reason,
            priority=priority,
            status='pending'
        )
        
        db.session.add(lab_request)
        db.session.commit()
        
        flash('Lab report request submitted successfully!', 'success')
        return redirect(url_for('patient.lab_requests'))
    
    # Get available doctors and labs
    from app.models import Doctor
    doctors = Doctor.query.all()
    labs = Lab.query.filter_by(is_active=True).all()
    consultations = Consultation.query.filter_by(patient_id=patient.id, status='completed').all()
    
    return render_template('patient/request_lab_report.html', 
                         patient=patient, 
                         doctors=doctors, 
                         labs=labs, 
                         consultations=consultations)

@patient_bp.route('/lab-requests')
@login_required
def lab_requests():
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    # Get patient's lab requests
    lab_requests = LabRequest.query.filter_by(patient_id=patient.id).order_by(LabRequest.created_at.desc()).all()
    
    return render_template('patient/lab_requests.html', patient=patient, lab_requests=lab_requests)

@patient_bp.route('/lab-request/<int:request_id>')
@login_required
def view_lab_request(request_id):
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash('Patient profile not found. Please contact support.', 'error')
        return redirect(url_for('main.home'))
    
    # Get lab request (only if it belongs to this patient)
    lab_request = LabRequest.query.filter_by(id=request_id, patient_id=patient.id).first()
    if not lab_request:
        flash('Lab request not found.', 'error')
        return redirect(url_for('patient.lab_requests'))
    
    return render_template('patient/view_lab_request.html', patient=patient, lab_request=lab_request) 