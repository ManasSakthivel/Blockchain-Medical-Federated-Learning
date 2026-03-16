#!/usr/bin/env python3
"""
Migration script to add LabRequest table
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import LabRequest

def migrate_lab_request():
    app = create_app()
    
    with app.app_context():
        try:
            # Create the LabRequest table
            db.create_all()
            print("✅ LabRequest table created successfully!")
            
            # Verify the table exists
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'lab_request' in tables:
                print("✅ LabRequest table verified in database")
            else:
                print("❌ LabRequest table not found in database")
                
        except Exception as e:
            print(f"❌ Error creating LabRequest table: {e}")
            return False
    
    return True

if __name__ == '__main__':
    migrate_lab_request() 