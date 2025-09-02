#!/usr/bin/env python3
"""
Script to create the lab user with fixed credentials
"""

from app import create_app
from app.models import db, User, Lab
from datetime import datetime

def create_lab_user():
    app = create_app()
    with app.app_context():
        # Check if lab user already exists
        existing_user = User.query.filter_by(email='lab@gmail.com').first()
        if existing_user:
            print("Lab user already exists!")
            return
        # Create lab user
        lab_user = User()
        lab_user.username = 'lab'
        lab_user.email = 'lab@gmail.com'
        lab_user.role = 'lab'
        lab_user.created_at = datetime.utcnow()
        lab_user.updated_at = datetime.utcnow()
        lab_user.set_password('lab')
        db.session.add(lab_user)
        db.session.commit()
        # Create lab profile
        lab_profile = Lab(
            user_id=lab_user.id,
            lab_name='Advanced Medical Laboratory',
            license_number='LAB001',
            phone='+91-9876543210',
            address='123 Medical Center, Healthcare Street, City - 123456',
            specialization='Retinal Imaging, Blood Tests, Urine Analysis',
            is_active=True
        )
        db.session.add(lab_profile)
        db.session.commit()
        print("Lab user created successfully!")
        print("Email: lab@gmail.com")
        print("Password: lab")
        print("Lab Name: Advanced Medical Laboratory")

if __name__ == '__main__':
    create_lab_user() 