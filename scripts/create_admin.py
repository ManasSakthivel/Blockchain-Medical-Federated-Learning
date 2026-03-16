#!/usr/bin/env python3
"""
Script to create an admin user for the EHR system
"""

from app import create_app, db
from app.models.user import User
from app.models.doctor import Doctor
from werkzeug.security import generate_password_hash

def create_admin_user():
    app = create_app()
    
    with app.app_context():
        # Check if admin already exists
        admin = User.query.filter_by(email='admin@ehr.com').first()
        if admin:
            print("Admin user already exists!")
            return
        
        # Create admin user
        admin_user = User(
            name='System Administrator',
            email='admin@ehr.com',
            password_hash=generate_password_hash('admin123'),
            role='admin'
        )
        
        db.session.add(admin_user)
        db.session.commit()
        
        print("Admin user created successfully!")
        print("Email: admin@ehr.com")
        print("Password: admin123")

def create_sample_doctor():
    app = create_app()
    
    with app.app_context():
        # Check if doctor already exists
        doctor_user = User.query.filter_by(email='doctor@ehr.com').first()
        if doctor_user:
            print("Sample doctor already exists!")
            return
        
        # Create doctor user
        doctor_user = User(
            name='Dr. John Smith',
            email='doctor@ehr.com',
            password_hash=generate_password_hash('doctor123'),
            role='doctor'
        )
        
        db.session.add(doctor_user)
        db.session.flush()  # Get user ID
        
        # Create doctor profile
        doctor = Doctor(
            user_id=doctor_user.id,
            specialization='General Medicine',
            license_number='MD123456',
            phone='(555) 123-4567',
            address='123 Medical Center Dr, City, State 12345'
        )
        
        db.session.add(doctor)
        db.session.commit()
        
        print("Sample doctor created successfully!")
        print("Email: doctor@ehr.com")
        print("Password: doctor123")

if __name__ == '__main__':
    print("Creating admin user...")
    create_admin_user()
    
    print("\nCreating sample doctor...")
    create_sample_doctor()
    
    print("\nSetup complete!") 