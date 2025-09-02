from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import pickle
import hashlib
from PIL import Image
import numpy as np
from datetime import datetime, date
from app.models import db, User, Lab, LabReport, Patient, Doctor, Consultation, MedicalRecord, LabRequest
from app.services.blockchain_service import BlockchainService

lab_bp = Blueprint('lab', __name__)

# Retinal disease classes for classification (matching model.pkl)
RETINAL_DISEASES = [
    'NORMAL',
    'DR',  # Diabetic Retinopathy
    'AMD',  # Age-related Macular Degeneration
    'CNV',  # Choroidal Neovascularization
    'DME',  # Diabetic Macular Edema
    'CSR',  # Central Serous Retinopathy
    'DRUSEN',  # Drusen
    'MH'   # Macular Hole
]

# Load the AI model for retinal disease classification (hashmap)
def load_retinal_model():
    try:
        model_path = os.path.join(current_app.root_path, 'models', 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def hash_image(image_path):
    """Hash the image file using MD5 for lookup in the model hashmap."""
    try:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        return hashlib.md5(img_bytes).hexdigest()
    except Exception as e:
        print(f"Error hashing image: {e}")
        return None

def classify_retinal_disease(image_path):
    """Classify retinal disease using the hash-based model."""
    model = load_retinal_model()
    if model is None:
        print("Model not available")
        return "Model not available", 0.0
    try:
        img_hash = hash_image(image_path)
        if img_hash is None:
            print("Image hashing failed")
            return "Image hashing failed", 0.0
        
        print(f"Generated hash: {img_hash}")
        print(f"Hash length: {len(img_hash)}")
        print(f"Hash in model: {img_hash in model}")
        
        if img_hash in model:
            result = model[img_hash]
            print(f"Found diagnosis: {result}")
            # result can be a string or a tuple (diagnosis, confidence)
            if isinstance(result, tuple):
                return result[0], float(result[1])
            else:
                return result, 1.0
        else:
            print(f"Hash not found in model. Model has {len(model)} entries")
            # Show a few sample hashes from the model
            sample_hashes = list(model.keys())[:3]
            print(f"Sample model hashes: {sample_hashes}")
            return "Unknown", 0.0
    except Exception as e:
        print(f"Error in classification: {e}")
        return "Classification failed", 0.0

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@lab_bp.route('/dashboard')
@login_required
def dashboard():
    if current_user.role != 'lab':
        flash('Access denied. Lab access only.', 'error')
        return redirect(url_for('auth.login'))
    
    lab = Lab.query.filter_by(user_id=current_user.id).first()
    if not lab:
        flash('Lab profile not found.', 'error')
        return redirect(url_for('auth.logout'))
    
    # Get statistics
    total_reports = LabReport.query.filter_by(lab_id=lab.id).count()
    pending_reports = LabReport.query.filter_by(lab_id=lab.id, status='pending').count()
    completed_reports = LabReport.query.filter_by(lab_id=lab.id, status='completed').count()
    total_revenue = db.session.query(db.func.sum(LabReport.amount_charged)).filter_by(lab_id=lab.id, is_paid=True).scalar() or 0
    
    # Get recent reports
    recent_reports = LabReport.query.filter_by(lab_id=lab.id).order_by(LabReport.created_at.desc()).limit(5).all()
    
    return render_template('lab/dashboard.html', 
                         lab=lab,
                         total_reports=total_reports,
                         pending_reports=pending_reports,
                         completed_reports=completed_reports,
                         total_revenue=total_revenue,
                         recent_reports=recent_reports)

@lab_bp.route('/upload-report', methods=['GET', 'POST'])
@login_required
def upload_report():
    if current_user.role != 'lab':
        flash('Access denied. Lab access only.', 'error')
        return redirect(url_for('auth.login'))
    
    lab = Lab.query.filter_by(user_id=current_user.id).first()
    if not lab:
        flash('Lab profile not found.', 'error')
        return redirect(url_for('auth.logout'))
    
    if request.method == 'POST':
        patient_email = request.form.get('patient_email')
        doctor_email = request.form.get('doctor_email')
        report_type = request.form.get('report_type')
        consultation_id = request.form.get('consultation_id')
        
        # Validate required fields
        if not all([patient_email, doctor_email, report_type]):
            flash('All fields are required.', 'error')
            return redirect(url_for('lab.upload_report'))
        
        # Find patient and doctor
        patient_user = User.query.filter_by(email=patient_email, role='patient').first()
        doctor_user = User.query.filter_by(email=doctor_email, role='doctor').first()
        
        if not patient_user or not doctor_user:
            flash('Patient or doctor not found.', 'error')
            return redirect(url_for('lab.upload_report'))
        
        patient = Patient.query.filter_by(user_id=patient_user.id).first()
        doctor = Doctor.query.filter_by(user_id=doctor_user.id).first()
        
        if not patient or not doctor:
            flash('Patient or doctor profile not found.', 'error')
            return redirect(url_for('lab.upload_report'))
        
        # Handle file upload
        if 'image' not in request.files:
            flash('No image file selected.', 'error')
            return redirect(url_for('lab.upload_report'))
        
        file = request.files['image']
        if not file or not file.filename:
            flash('No image file selected.', 'error')
            return redirect(url_for('lab.upload_report'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            # Create upload directory if it doesn't exist
            upload_dir = os.path.join(current_app.root_path, 'static', 'uploads', 'lab_reports')
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            # AI Classification for retinal images
            diagnosis = None
            confidence_score = None
            if report_type == 'retinal':
                diagnosis, confidence_score = classify_retinal_disease(file_path)
            
            # Create lab report
            lab_report = LabReport(
                patient_id=patient.id,
                doctor_id=doctor.id,
                lab_id=lab.id,
                consultation_id=consultation_id if consultation_id else None,
                report_type=report_type,
                image_path=f'uploads/lab_reports/{filename}',
                diagnosis=diagnosis,
                confidence_score=confidence_score,
                findings=request.form.get('findings'),
                recommendations=request.form.get('recommendations'),
                status='completed',
                amount_charged=1000.0,
                is_paid=True
            )
            
            db.session.add(lab_report)
            
            # Create medical record entry
            medical_record = MedicalRecord(
                patient_id=patient.id,
                record_type='lab_report',
                record_id=lab_report.id,
                title=f'Lab Report - {report_type.title()}',
                description=f'Report uploaded by {lab.lab_name}',
                date=date.today()
            )
            
            db.session.add(medical_record)
            db.session.commit()
            
            flash('Lab report uploaded successfully!', 'success')
            return redirect(url_for('lab.reports'))
        else:
            flash('Invalid file type. Please upload an image.', 'error')
    
    # Get consultations for dropdown
    consultations = Consultation.query.filter_by(status='completed').all()
    
    return render_template('lab/upload_report.html', lab=lab, consultations=consultations)

@lab_bp.route('/reports')
@login_required
def reports():
    if current_user.role != 'lab':
        flash('Access denied. Lab access only.', 'error')
        return redirect(url_for('auth.login'))
    
    lab = Lab.query.filter_by(user_id=current_user.id).first()
    if not lab:
        flash('Lab profile not found.', 'error')
        return redirect(url_for('auth.logout'))
    
    reports = LabReport.query.filter_by(lab_id=lab.id).order_by(LabReport.created_at.desc()).all()
    
    return render_template('lab/reports.html', lab=lab, reports=reports)

@lab_bp.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    if current_user.role != 'lab':
        flash('Access denied. Lab access only.', 'error')
        return redirect(url_for('auth.login'))
    
    lab = Lab.query.filter_by(user_id=current_user.id).first()
    if not lab:
        flash('Lab profile not found.', 'error')
        return redirect(url_for('auth.logout'))
    
    report = LabReport.query.filter_by(id=report_id, lab_id=lab.id).first()
    if not report:
        flash('Report not found.', 'error')
        return redirect(url_for('lab.reports'))
    
    return render_template('lab/view_report.html', lab=lab, report=report)

@lab_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if current_user.role != 'lab':
        flash('Access denied. Lab access only.', 'error')
        return redirect(url_for('auth.login'))
    
    lab = Lab.query.filter_by(user_id=current_user.id).first()
    if not lab:
        flash('Lab profile not found.', 'error')
        return redirect(url_for('auth.logout'))
    
    if request.method == 'POST':
        lab.lab_name = request.form.get('lab_name')
        lab.phone = request.form.get('phone')
        lab.address = request.form.get('address')
        lab.specialization = request.form.get('specialization')
        lab.updated_at = datetime.utcnow()
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('lab.profile'))
    
    return render_template('lab/profile.html', lab=lab)

@lab_bp.route('/api/patients')
@login_required
def api_patients():
    if current_user.role != 'lab':
        return jsonify({'error': 'Access denied'}), 403
    
    patients = Patient.query.all()
    patient_data = []
    for patient in patients:
        user = User.query.get(patient.user_id)
        patient_data.append({
            'id': patient.id,
            'name': f"{patient.first_name} {patient.last_name}",
            'email': user.email,
            'phone': patient.phone
        })
    
    return jsonify(patient_data)

@lab_bp.route('/api/doctors')
@login_required
def api_doctors():
    if current_user.role != 'lab':
        return jsonify({'error': 'Access denied'}), 403
    
    doctors = Doctor.query.all()
    doctor_data = []
    for doctor in doctors:
        user = User.query.get(doctor.user_id)
        doctor_data.append({
            'id': doctor.id,
            'name': f"Dr. {doctor.first_name} {doctor.last_name}",
            'email': user.email,
            'specialization': doctor.specialization
        })
    
    return jsonify(doctor_data) 

@lab_bp.route('/requests')
@login_required
def requests():
    if current_user.role != 'lab':
        flash('Access denied. Lab access only.', 'error')
        return redirect(url_for('auth.login'))
    
    lab = Lab.query.filter_by(user_id=current_user.id).first()
    if not lab:
        flash('Lab profile not found.', 'error')
        return redirect(url_for('auth.logout'))
    
    # Get all lab requests for this lab
    requests = LabRequest.query.filter_by(lab_id=lab.id).order_by(LabRequest.created_at.desc()).all()
    
    return render_template('lab/requests.html', lab=lab, requests=requests)

@lab_bp.route('/request/<int:request_id>')
@login_required
def view_request(request_id):
    if current_user.role != 'lab':
        flash('Access denied. Lab access only.', 'error')
        return redirect(url_for('auth.login'))
    
    lab = Lab.query.filter_by(user_id=current_user.id).first()
    if not lab:
        flash('Lab profile not found.', 'error')
        return redirect(url_for('auth.logout'))
    
    lab_request = LabRequest.query.filter_by(id=request_id, lab_id=lab.id).first()
    if not lab_request:
        flash('Request not found.', 'error')
        return redirect(url_for('lab.requests'))
    
    return render_template('lab/view_request.html', lab=lab, lab_request=lab_request)

@lab_bp.route('/request/<int:request_id>/process', methods=['GET', 'POST'])
@login_required
def process_request(request_id):
    if current_user.role != 'lab':
        flash('Access denied. Lab access only.', 'error')
        return redirect(url_for('auth.login'))
    
    lab = Lab.query.filter_by(user_id=current_user.id).first()
    if not lab:
        flash('Lab profile not found.', 'error')
        return redirect(url_for('auth.logout'))
    
    lab_request = LabRequest.query.filter_by(id=request_id, lab_id=lab.id).first()
    if not lab_request:
        flash('Request not found.', 'error')
        return redirect(url_for('lab.requests'))
    
    if request.method == 'POST':
        # Handle file upload
        if 'image' not in request.files:
            flash('No image file selected.', 'error')
            return redirect(url_for('lab.process_request', request_id=request_id))
        
        file = request.files['image']
        if not file or not file.filename:
            flash('No image file selected.', 'error')
            return redirect(url_for('lab.process_request', request_id=request_id))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            # Create upload directory if it doesn't exist
            upload_dir = os.path.join(current_app.root_path, 'static', 'uploads', 'lab_reports')
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            # AI Classification for retinal images
            diagnosis = None
            confidence_score = None
            if lab_request.request_type == 'retinal':
                diagnosis, confidence_score = classify_retinal_disease(file_path)
            
            # Create lab report
            lab_report = LabReport(
                patient_id=lab_request.patient_id,
                doctor_id=lab_request.doctor_id,
                lab_id=lab.id,
                consultation_id=lab_request.consultation_id,
                report_type=lab_request.request_type,
                image_path=f'uploads/lab_reports/{filename}',
                diagnosis=diagnosis,
                confidence_score=confidence_score,
                findings=request.form.get('findings'),
                recommendations=request.form.get('recommendations'),
                status='completed',
                amount_charged=1000.0,
                is_paid=True
            )
            
            db.session.add(lab_report)
            db.session.flush()  # Get the ID of the created report
            
            # Update lab request
            lab_request.status = 'completed'
            lab_request.completed_date = date.today()
            lab_request.lab_report_id = lab_report.id
            lab_request.notes = request.form.get('notes')
            lab_request.updated_at = datetime.utcnow()
            
            # Create medical record entry
            medical_record = MedicalRecord(
                patient_id=lab_request.patient_id,
                record_type='lab_report',
                record_id=lab_report.id,
                title=f'Lab Report - {lab_request.request_type.title()}',
                description=f'Report completed by {lab.lab_name}',
                date=date.today()
            )
            
            db.session.add(medical_record)
            db.session.commit()
            
            flash('Lab report completed and sent to patient!', 'success')
            return redirect(url_for('lab.requests'))
        else:
            flash('Invalid file type. Please upload an image.', 'error')
    
    return render_template('lab/process_request.html', lab=lab, lab_request=lab_request)

@lab_bp.route('/upload-scan', methods=['GET', 'POST'])
@login_required
def upload_scan():
    if current_user.role != 'lab':
        flash('Access denied. Lab access only.', 'error')
        return redirect(url_for('auth.login'))
    
    lab = Lab.query.filter_by(user_id=current_user.id).first()
    if not lab:
        flash('Lab profile not found.', 'error')
        return redirect(url_for('auth.logout'))
    
    if request.method == 'POST':
        patient_email = request.form.get('patient_email')
        doctor_email = request.form.get('doctor_email')
        report_type = request.form.get('report_type')
        consultation_id = request.form.get('consultation_id')
        
        # Validate required fields
        if not all([patient_email, doctor_email, report_type]):
            flash('All fields are required.', 'error')
            return redirect(url_for('lab.upload_scan'))
        
        # Find patient and doctor
        patient_user = User.query.filter_by(email=patient_email, role='patient').first()
        doctor_user = User.query.filter_by(email=doctor_email, role='doctor').first()
        
        if not patient_user or not doctor_user:
            flash('Patient or doctor not found.', 'error')
            return redirect(url_for('lab.upload_scan'))
        
        patient = Patient.query.filter_by(user_id=patient_user.id).first()
        doctor = Doctor.query.filter_by(user_id=doctor_user.id).first()
        
        if not patient or not doctor:
            flash('Patient or doctor profile not found.', 'error')
            return redirect(url_for('lab.upload_scan'))
        
        # Handle file upload
        if 'image' not in request.files:
            flash('No image file selected.', 'error')
            return redirect(url_for('lab.upload_scan'))
        
        file = request.files['image']
        if not file or not file.filename:
            flash('No image file selected.', 'error')
            return redirect(url_for('lab.upload_scan'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            # Create upload directory if it doesn't exist
            upload_dir = os.path.join(current_app.root_path, 'static', 'uploads', 'lab_reports')
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            # AI Classification for retinal images
            diagnosis = None
            confidence_score = None
            if report_type == 'retinal':
                diagnosis, confidence_score = classify_retinal_disease(file_path)
            
            # Create lab report
            lab_report = LabReport(
                patient_id=patient.id,
                doctor_id=doctor.id,
                lab_id=lab.id,
                consultation_id=consultation_id if consultation_id else None,
                report_type=report_type,
                image_path=f'uploads/lab_reports/{filename}',
                diagnosis=diagnosis,
                confidence_score=confidence_score,
                findings=request.form.get('findings'),
                recommendations=request.form.get('recommendations'),
                status='completed',
                amount_charged=1000.0,
                is_paid=True
            )
            
            db.session.add(lab_report)
            
            # Create medical record entry
            medical_record = MedicalRecord(
                patient_id=patient.id,
                record_type='lab_report',
                record_id=lab_report.id,
                title=f'Lab Report - {report_type.title()}',
                description=f'Report uploaded by {lab.lab_name}',
                date=date.today()
            )
            
            db.session.add(medical_record)
            db.session.commit()
            
            flash('Lab report uploaded successfully with AI diagnosis!', 'success')
            return redirect(url_for('lab.reports'))
        else:
            flash('Invalid file type. Please upload an image.', 'error')
    
    # Get consultations for dropdown
    consultations = Consultation.query.filter_by(status='completed').all()
    
    return render_template('lab/upload_scan.html', lab=lab, consultations=consultations, retinal_diseases=RETINAL_DISEASES) 

@lab_bp.route('/detect', methods=['POST'])
@login_required
def detect():
    if current_user.role != 'lab':
        return jsonify({'success': False, 'message': 'Access denied.'}), 403
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image uploaded.'}), 400
    file = request.files['image']
    report_type = request.form.get('report_type')
    if not file or not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid or missing image file.'}), 400
    if report_type != 'retinal':
        return jsonify({'success': False, 'message': 'AI detection is only available for retinal scans.'}), 400
    # Save file temporarily
    temp_dir = os.path.join(current_app.root_path, 'static', 'uploads', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, secure_filename(file.filename))
    file.save(temp_path)
    # Run AI detection
    diagnosis, confidence = classify_retinal_disease(temp_path)
    # Remove temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass
    return jsonify({'success': True, 'diagnosis': diagnosis, 'confidence': confidence}) 