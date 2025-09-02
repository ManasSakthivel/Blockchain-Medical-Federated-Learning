from flask import Blueprint, render_template, redirect, url_for
from flask_login import current_user

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    return render_template('main/home.html')

@main_bp.route('/dashboard')
def dashboard():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    
    if current_user.role == 'admin':
        return redirect(url_for('admin.dashboard'))
    elif current_user.role == 'doctor':
        return redirect(url_for('doctor.dashboard'))
    elif current_user.role == 'patient':
        return redirect(url_for('patient.dashboard'))
    
    return redirect(url_for('main.home')) 