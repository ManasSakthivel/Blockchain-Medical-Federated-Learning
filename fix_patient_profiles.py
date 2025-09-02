#!/usr/bin/env python3
"""
Script to fix missing patient profiles for existing users
"""

from app import create_app, db
from app.models.user import User
from app.models.patient import Patient
from app.models.doctor import Doctor

def fix_missing_profiles():
    app = create_app()
    
    with app.app_context():
        # Find users with role 'patient' but no patient profile
        patients_without_profile = []
        for user in User.query.filter_by(role='patient').all():
            patient = Patient.query.filter_by(user_id=user.id).first()
            if not patient:
                patients_without_profile.append(user)
        
        # Find users with role 'doctor' but no doctor profile
        doctors_without_profile = []
        for user in User.query.filter_by(role='doctor').all():
            doctor = Doctor.query.filter_by(user_id=user.id).first()
            if not doctor:
                doctors_without_profile.append(user)
        
        print(f"Found {len(patients_without_profile)} patients without profiles")
        print(f"Found {len(doctors_without_profile)} doctors without profiles")
        
        # Create missing patient profiles
        for user in patients_without_profile:
            patient = Patient(user_id=user.id)
            db.session.add(patient)
            print(f"Created patient profile for {user.name} ({user.email})")
        
        # Create missing doctor profiles
        for user in doctors_without_profile:
            doctor = Doctor(user_id=user.id, specialization='General Medicine', license_number='')
            db.session.add(doctor)
            print(f"Created doctor profile for {user.name} ({user.email})")
        
        if patients_without_profile or doctors_without_profile:
            db.session.commit()
            print("All missing profiles have been created!")
        else:
            print("No missing profiles found.")

def list_all_users():
    app = create_app()
    
    with app.app_context():
        print("\n=== All Users ===")
        for user in User.query.all():
            patient = Patient.query.filter_by(user_id=user.id).first()
            doctor = Doctor.query.filter_by(user_id=user.id).first()
            
            profile_status = "✅ Has profile"
            if user.role == 'patient' and not patient:
                profile_status = "❌ Missing patient profile"
            elif user.role == 'doctor' and not doctor:
                profile_status = "❌ Missing doctor profile"
            elif user.role == 'admin':
                profile_status = "✅ Admin (no profile needed)"
            
            print(f"{user.name} ({user.email}) - {user.role} - {profile_status}")

if __name__ == '__main__':
    print("Checking for missing user profiles...")
    list_all_users()
    fix_missing_profiles()
    print("\nFinal status:")
    list_all_users() 