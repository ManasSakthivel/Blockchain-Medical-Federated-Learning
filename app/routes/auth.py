from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user
from app.models import db, User, Patient, Doctor
from datetime import datetime

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('main.home'))
        else:
            flash('Invalid email or password.', 'error')
    
    return render_template('auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('auth.register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'error')
            return redirect(url_for('auth.register'))
        
        # Create user
        user = User(username=username, email=email, role=role)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        # Create profile based on role
        if role == 'patient':
            date_of_birth_str = request.form.get('date_of_birth')
            date_of_birth = None
            if date_of_birth_str:
                try:
                    date_of_birth = datetime.strptime(date_of_birth_str, '%Y-%m-%d').date()
                except ValueError:
                    flash('Invalid date format for date of birth. Please use YYYY-MM-DD format.', 'error')
                    return redirect(url_for('auth.register'))
            
            patient = Patient(
                user_id=user.id,
                first_name=request.form.get('first_name'),
                last_name=request.form.get('last_name'),
                date_of_birth=date_of_birth,
                gender=request.form.get('gender'),
                phone=request.form.get('phone'),
                address=request.form.get('address'),
                emergency_contact=request.form.get('emergency_contact')
            )
            db.session.add(patient)
        elif role == 'doctor':
            doctor = Doctor(
                user_id=user.id,
                first_name=request.form.get('first_name'),
                last_name=request.form.get('last_name'),
                specialization=request.form.get('specialization'),
                license_number=request.form.get('license_number'),
                phone=request.form.get('phone'),
                address=request.form.get('address'),
                experience_years=int(request.form.get('experience_years', 0)),
                education=request.form.get('education')
            )
            db.session.add(doctor)
        
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('main.home')) 