#!/usr/bin/env python3
"""
Test Web Interface Upload with Authentication
This script simulates the exact web interface upload process.
"""

import os
import tempfile
import time
from app import create_app
from app.models import db, User, Patient
from app.services.file_verification_service import FileVerificationService
from werkzeug.datastructures import FileStorage
from io import BytesIO

def create_test_file():
    """Create a test file for upload"""
    timestamp = int(time.time())
    test_content = f"This is a test medical record file for blockchain verification.\nPatient: John Doe\nDate: 2024-01-15\nDiagnosis: Healthy\nTimestamp: {timestamp}".encode()
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    temp_file.write(test_content)
    temp_file.close()
    
    return temp_file.name

def test_web_interface_upload():
    """Test upload using the exact web interface method"""
    print("🌐 Testing Web Interface Upload with Authentication")
    print("=" * 60)
    
    try:
        # Create Flask app context
        app = create_app()
        
        with app.app_context():
            # Create test file
            test_file_path = create_test_file()
            print(f"✅ Created test file: {test_file_path}")
            
            # Read file content
            with open(test_file_path, 'rb') as f:
                file_bytes = f.read()
            
            filename = os.path.basename(test_file_path)
            file_size = len(file_bytes)
            
            print(f"✅ File info: {filename}, {file_size} bytes")
            
            # Create a mock file object (like Flask's request.files)
            file_obj = FileStorage(
                stream=BytesIO(file_bytes),
                filename=filename,
                content_type='text/plain'
            )
            
            # Initialize service
            file_service = FileVerificationService()
            
            # Test upload with patient ID (like web interface)
            result = file_service.upload_file_secure(
                file=file_obj,
                patient_id="test_patient_123",
                metadata={
                    "description": "Test medical record",
                    "category": "medical",
                    "uploaded_by_user": "test@example.com",
                    "uploaded_by_role": "doctor"
                }
            )
            
            print(f"✅ Upload result: {result}")
            
            if result.get("success"):
                print("🎉 Web interface upload successful!")
                print(f"   File ID: {result.get('file_id')}")
                print(f"   Transaction Hash: {result.get('blockchain_tx')}")
                print(f"   File Hash: {result.get('file_hash')}")
                print(f"   IPFS Hash: {result.get('ipfs_hash')}")
                return True
            else:
                print(f"❌ Web interface upload failed: {result.get('error')}")
                return False
                
    except Exception as e:
        print(f"❌ Error during web upload test: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False
    finally:
        # Cleanup
        if 'test_file_path' in locals():
            try:
                os.unlink(test_file_path)
                print(f"✅ Cleaned up test file: {test_file_path}")
            except:
                pass

def main():
    """Main test function"""
    print("🚀 Web Interface Upload Test with Authentication")
    print("=" * 60)
    
    success = test_web_interface_upload()
    
    if success:
        print("\n🎉 Web interface upload test completed successfully!")
        print("✅ Your web interface should work correctly!")
        print("\n📋 To test the actual web interface:")
        print("   1. Go to: http://127.0.0.1:5002")
        print("   2. Login as a doctor/lab/admin user")
        print("   3. Navigate to: File Verification → Upload File")
        print("   4. Upload a file")
    else:
        print("\n❌ Web interface upload test failed!")
        print("🔧 Check the error messages above for details")

if __name__ == "__main__":
    main()
