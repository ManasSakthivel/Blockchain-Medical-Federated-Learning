from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from app.models import db, User, Patient, Doctor, Lab
from app.services.file_verification_service import FileVerificationService
import io

file_verification_bp = Blueprint('file_verification', __name__)

@file_verification_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    """Secure file upload with blockchain verification"""
    if current_user.role not in ['doctor', 'lab', 'admin']:
        flash('Access denied. Only doctors, labs, and admins can upload files.', 'error')
        return redirect(url_for('main.home'))
    
    file_service = FileVerificationService()
    
    if request.method == 'POST':
        # Get form data
        patient_email = request.form.get('patient_email')
        file_description = request.form.get('description')
        file_category = request.form.get('category', 'medical')
        
        # Validate required fields
        if not patient_email:
            flash('Patient email is required.', 'error')
            return redirect(url_for('file_verification.upload_file'))
        
        # Find patient
        patient_user = User.query.filter_by(email=patient_email, role='patient').first()
        if not patient_user:
            flash('Patient not found.', 'error')
            return redirect(url_for('file_verification.upload_file'))
        
        patient = Patient.query.filter_by(user_id=patient_user.id).first()
        if not patient:
            flash('Patient profile not found.', 'error')
            return redirect(url_for('file_verification.upload_file'))
        
        # Handle file upload
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(url_for('file_verification.upload_file'))
        
        file = request.files['file']
        if not file or not file.filename:
            flash('No file selected.', 'error')
            return redirect(url_for('file_verification.upload_file'))
        
        # Prepare metadata
        metadata = {
            "description": file_description,
            "category": file_category,
            "uploaded_by_user": current_user.email,
            "uploaded_by_role": current_user.role
        }
        
        # Upload file securely
        result = file_service.upload_file_secure(
            file=file,
            patient_id=patient.id,
            metadata=metadata
        )
        
        if result['success']:
            flash(f'File uploaded successfully! File ID: {result["file_id"]}', 'success')
            return redirect(url_for('file_verification.view_file', file_id=result['file_id']))
        else:
            flash(f'Upload failed: {result["error"]}', 'error')
    
    # Get patients for dropdown
    patients = Patient.query.join(User).filter(User.role == 'patient').all()
    
    return render_template('file_verification/upload.html', patients=patients)

@file_verification_bp.route('/verify', methods=['GET', 'POST'])
@login_required
def verify_file():
    """Verify file integrity"""
    if current_user.role not in ['doctor', 'lab', 'admin', 'patient']:
        flash('Access denied.', 'error')
        return redirect(url_for('main.home'))
    
    file_service = FileVerificationService()
    
    if request.method == 'POST':
        file_id = request.form.get('file_id')
        verification_notes = request.form.get('notes', '')
        
        if not file_id:
            flash('File ID is required.', 'error')
            return redirect(url_for('file_verification.verify_file'))
        
        # Handle file upload for verification
        if 'file' not in request.files:
            flash('No file selected for verification.', 'error')
            return redirect(url_for('file_verification.verify_file'))
        
        file = request.files['file']
        if not file or not file.filename:
            flash('No file selected for verification.', 'error')
            return redirect(url_for('file_verification.verify_file'))
        
        # Read file content
        file_content = file.read()
        
        # Verify file integrity
        result = file_service.verify_file_integrity(
            file_id=int(file_id),
            file_bytes=file_content
        )
        
        if result['success']:
            if result['is_match']:
                flash('File integrity verified successfully! File has not been tampered with.', 'success')
            else:
                flash('WARNING: File integrity check failed! File may have been tampered with.', 'error')
            
            return render_template('file_verification/verification_result.html', result=result)
        else:
            flash(f'Verification failed: {result["error"]}', 'error')
    
    return render_template('file_verification/verify.html')

@file_verification_bp.route('/file/<int:file_id>')
@login_required
def view_file(file_id):
    """View file details and verification history"""
    if current_user.role not in ['doctor', 'lab', 'admin', 'patient']:
        flash('Access denied.', 'error')
        return redirect(url_for('main.home'))
    
    file_service = FileVerificationService()
    
    # Get file record from blockchain
    file_record = file_service.blockchain_service.get_file_record(file_id)
    if not file_record:
        flash('File not found.', 'error')
        return redirect(url_for('file_verification.list_files'))
    
    # Get verification logs
    verification_logs = file_service.get_verification_logs(file_id)
    
    return render_template('file_verification/view_file.html', 
                         file_record=file_record, 
                         verification_logs=verification_logs)

@file_verification_bp.route('/files')
@login_required
def list_files():
    """List all files uploaded by the user"""
    if current_user.role not in ['doctor', 'lab', 'admin']:
        flash('Access denied.', 'error')
        return redirect(url_for('main.home'))
    
    file_service = FileVerificationService()
    
    # Get user's files using the current account address
    user_files = file_service.get_user_files(file_service.blockchain_service.account)
    
    # Get file details for each file ID
    files = []
    for file_id in user_files:
        file_record = file_service.blockchain_service.get_file_record(file_id)
        if file_record:
            files.append(file_record)
    
    return render_template('file_verification/list_files.html', files=files)

@file_verification_bp.route('/demo/tamper')
@login_required
def demo_tamper():
    """Demonstrate file tampering detection"""
    if current_user.role not in ['doctor', 'lab', 'admin']:
        flash('Access denied.', 'error')
        return redirect(url_for('main.home'))
    
    file_service = FileVerificationService()
    
    # Create a demo file
    demo_content = b"This is a demo medical record file for testing integrity verification."
    demo_filename = "demo_medical_record.txt"
    
    # Save demo file
    upload_dir = os.path.join(current_app.root_path, 'static', 'uploads', 'demo')
    os.makedirs(upload_dir, exist_ok=True)
    demo_path = os.path.join(upload_dir, demo_filename)
    
    with open(demo_path, 'wb') as f:
        f.write(demo_content)
    
    # Create tampered version
    tampered_path = file_service.create_tampered_file_demo(demo_path)
    
    if tampered_path:
        # Calculate hashes
        original_hash = file_service.calculate_file_hash(demo_path)
        tampered_hash = file_service.calculate_file_hash(tampered_path)
        
        demo_data = {
            'original_file': demo_path,
            'tampered_file': tampered_path,
            'original_hash': original_hash,
            'tampered_hash': tampered_hash,
            'hashes_match': original_hash == tampered_hash
        }
        
        return render_template('file_verification/demo_tamper.html', demo_data=demo_data)
    else:
        flash('Failed to create demo files.', 'error')
        return redirect(url_for('file_verification.upload_file'))

@file_verification_bp.route('/download/<int:file_id>')
@login_required
def download_file(file_id):
    """Download file from IPFS"""
    if current_user.role not in ['doctor', 'lab', 'admin', 'patient']:
        flash('Access denied.', 'error')
        return redirect(url_for('main.home'))
    
    file_service = FileVerificationService()
    
    # Get file record
    file_record = file_service.blockchain_service.get_file_record(file_id)
    if not file_record:
        flash('File not found.', 'error')
        return redirect(url_for('file_verification.list_files'))
    
    # Get file from IPFS
    file_content = file_service.get_file_from_ipfs(file_record['ipfs_hash'])
    if not file_content:
        flash('Failed to retrieve file from IPFS.', 'error')
        return redirect(url_for('file_verification.view_file', file_id=file_id))
    
    # Return file for download
    return send_file(
        io.BytesIO(file_content),
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=file_record['file_name']
    )

@file_verification_bp.route('/api/upload', methods=['POST'])
@login_required
def api_upload_file():
    """API endpoint for file upload"""
    if current_user.role not in ['doctor', 'lab', 'admin']:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    file_service = FileVerificationService()
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    patient_id = request.form.get('patient_id')
    metadata = request.form.get('metadata', '{}')
    
    try:
        metadata = json.loads(metadata)
    except:
        metadata = {}
    
    result = file_service.upload_file_secure(
        file=file,
        patient_id=patient_id,
        metadata=metadata
    )
    
    return jsonify(result)

@file_verification_bp.route('/api/verify', methods=['POST'])
@login_required
def api_verify_file():
    """API endpoint for file verification"""
    if current_user.role not in ['doctor', 'lab', 'admin', 'patient']:
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    
    file_service = FileVerificationService()
    
    file_id = request.form.get('file_id')
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_content = file.read()
    
    result = file_service.verify_file_integrity(
        file_id=int(file_id),
        file_bytes=file_content
    )
    
    return jsonify(result)
