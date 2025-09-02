from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from app.models import db, User, Doctor, Patient, MedicalRecord, Consultation
from datetime import datetime, timedelta

admin_bp = Blueprint('admin', __name__)

@admin_bp.before_request
def require_admin():
    if not current_user.is_authenticated or current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('main.home'))

@admin_bp.route('/dashboard')
@login_required
def dashboard():
    # Get system statistics
    total_users = User.query.count()
    total_patients = Patient.query.count()
    total_doctors = Doctor.query.count()
    total_records = MedicalRecord.query.count()
    total_consultations = Consultation.query.count()
    
    # Get recent activity
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    recent_records = MedicalRecord.query.order_by(MedicalRecord.created_at.desc()).limit(5).all()
    
    # Get consultations for today
    today = datetime.now().date()
    today_consultations = Consultation.query.filter_by(date=today).all()
    
    return render_template('admin/dashboard.html',
                         total_users=total_users,
                         total_patients=total_patients,
                         total_doctors=total_doctors,
                         total_records=total_records,
                         total_consultations=total_consultations,
                         recent_users=recent_users,
                         recent_records=recent_records,
                         today_consultations=today_consultations)

@admin_bp.route('/users')
@login_required
def users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users.html', users=users)

@admin_bp.route('/user/<int:user_id>')
@login_required
def view_user(user_id):
    user = User.query.get_or_404(user_id)
    return render_template('admin/view_user.html', user=user)

@admin_bp.route('/user/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        user.name = request.form.get('name')
        user.email = request.form.get('email')
        user.role = request.form.get('role')
        user.updated_at = datetime.utcnow()
        
        db.session.commit()
        flash('User updated successfully!', 'success')
        return redirect(url_for('admin.view_user', user_id=user_id))
    
    return render_template('admin/edit_user.html', user=user)

@admin_bp.route('/user/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('Cannot delete your own account!', 'error')
        return redirect(url_for('admin.users'))
    
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully!', 'success')
    return redirect(url_for('admin.users'))

@admin_bp.route('/patients')
@login_required
def patients():
    patients = Patient.query.order_by(Patient.created_at.desc()).all()
    return render_template('admin/patients.html', patients=patients)

@admin_bp.route('/patient/<int:patient_id>')
@login_required
def view_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    records = MedicalRecord.query.filter_by(patient_id=patient_id).order_by(MedicalRecord.created_at.desc()).all()
    consultations = Consultation.query.filter_by(patient_id=patient_id).order_by(Consultation.date.desc()).all()
    
    return render_template('admin/view_patient.html', 
                         patient=patient, 
                         records=records, 
                         consultations=consultations)

@admin_bp.route('/doctors')
@login_required
def doctors():
    doctors = Doctor.query.order_by(Doctor.created_at.desc()).all()
    return render_template('admin/doctors.html', doctors=doctors)

@admin_bp.route('/doctor/<int:doctor_id>')
@login_required
def view_doctor(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    records = MedicalRecord.query.filter_by(doctor_id=doctor_id).order_by(MedicalRecord.created_at.desc()).all()
    consultations = Consultation.query.filter_by(doctor_id=doctor_id).order_by(Consultation.date.desc()).all()
    
    return render_template('admin/view_doctor.html', 
                         doctor=doctor, 
                         records=records, 
                         consultations=consultations)

@admin_bp.route('/records')
@login_required
def records():
    records = MedicalRecord.query.order_by(MedicalRecord.created_at.desc()).all()
    return render_template('admin/records.html', records=records)

@admin_bp.route('/record/<int:record_id>')
@login_required
def view_record(record_id):
    record = MedicalRecord.query.get_or_404(record_id)
    return render_template('admin/view_record.html', record=record)

@admin_bp.route('/consultations')
@login_required
def consultations():
    consultations = Consultation.query.order_by(Consultation.date.desc()).all()
    return render_template('admin/consultations.html', consultations=consultations)

@admin_bp.route('/consultation/<int:consultation_id>')
@login_required
def view_consultation(consultation_id):
    consultation = Consultation.query.get_or_404(consultation_id)
    return render_template('admin/view_consultation.html', consultation=consultation)

@admin_bp.route('/reports')
@login_required
def reports():
    # Generate various reports
    today = datetime.now().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    # Weekly statistics
    weekly_records = MedicalRecord.query.filter(
        MedicalRecord.created_at >= week_ago
    ).count()
    
    weekly_consultations = Consultation.query.filter(
        Consultation.date >= week_ago
    ).count()
    
    # Monthly statistics
    monthly_records = MedicalRecord.query.filter(
        MedicalRecord.created_at >= month_ago
    ).count()
    
    monthly_consultations = Consultation.query.filter(
        Consultation.date >= month_ago
    ).count()
    
    # Top doctors by records
    from sqlalchemy import func
    top_doctors = db.session.query(
        Doctor, func.count(MedicalRecord.id).label('record_count')
    ).join(MedicalRecord).group_by(Doctor.id).order_by(
        func.count(MedicalRecord.id).desc()
    ).limit(5).all()
    
    return render_template('admin/reports.html',
                         weekly_records=weekly_records,
                         weekly_consultations=weekly_consultations,
                         monthly_records=monthly_records,
                         monthly_consultations=monthly_consultations,
                         top_doctors=top_doctors)

@admin_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        # Handle system settings updates
        flash('Settings updated successfully!', 'success')
        return redirect(url_for('admin.settings'))
    
    return render_template('admin/settings.html') 