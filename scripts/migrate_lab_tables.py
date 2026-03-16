#!/usr/bin/env python3
"""
Migration script to add lab-related tables
"""

from app import create_app, db
from app.models import Lab, LabReport, Prescription, MedicalRecord

def migrate_lab_tables():
    app = create_app()
    
    with app.app_context():
        print("Creating lab-related tables...")
        
        # Create tables
        db.create_all()
        
        print("Migration completed successfully!")
        print("New tables created:")
        print("- Lab")
        print("- LabReport") 
        print("- Prescription")
        print("- MedicalRecord")

if __name__ == '__main__':
    migrate_lab_tables() 