#!/usr/bin/env python3
"""
Test Web Interface Upload
This script simulates the web interface upload to identify issues.
"""

import os
import tempfile
import hashlib
import time
from app import create_app
from app.services.file_verification_service import FileVerificationService
from app.services.blockchain_service import BlockchainService

# Mock IPFS service for testing
class MockIPFSService:
    def __init__(self, ipfs_url="http://127.0.0.1:5001"):
        self.ipfs_url = ipfs_url
        
    def upload_file(self, file_path):
        """Mock upload file to IPFS"""
        # Return a mock IPFS hash
        return f"QmMockHash{int(time.time())}"
    
    def pin_file(self, ipfs_hash):
        """Mock pin file"""
        return True

def create_test_file():
    """Create a test file for upload"""
    timestamp = int(time.time())
    test_content = f"This is a test medical record file for blockchain verification.\nPatient: John Doe\nDate: 2024-01-15\nDiagnosis: Healthy\nTimestamp: {timestamp}".encode()
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    temp_file.write(test_content)
    temp_file.close()
    
    return temp_file.name

def test_web_upload():
    """Test upload using the same method as web interface"""
    print("🌐 Testing Web Interface Upload")
    print("=" * 50)
    
    try:
        # Create Flask app context
        app = create_app()
        
        with app.app_context():
            # Create test file
            test_file_path = create_test_file()
            print(f"✅ Created test file: {test_file_path}")
            
            # Initialize service with mock IPFS
            file_service = FileVerificationService()
            file_service.ipfs_service = MockIPFSService()  # Use mock IPFS
            
            # Test the secure_upload_file method (same as web interface)
            with open(test_file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Get file info
            filename = os.path.basename(test_file_path)
            file_size = len(file_bytes)
            
            print(f"✅ File info: {filename}, {file_size} bytes")
            
            # Create a mock file object (like Flask's request.files)
            from werkzeug.datastructures import FileStorage
            from io import BytesIO
            
            # Create mock file object
            file_obj = FileStorage(
                stream=BytesIO(file_bytes),
                filename=filename,
                content_type='text/plain'
            )
            
            # Test upload (same as web interface)
            result = file_service.upload_file_secure(
                file=file_obj,
                patient_id="test_patient_123",
                metadata={"test": True, "source": "web_interface_test"}
            )
            
            print(f"✅ Upload result: {result}")
            
            if result.get("success"):
                print("🎉 Web interface upload successful!")
                print(f"   File ID: {result.get('file_id')}")
                print(f"   Transaction Hash: {result.get('blockchain_tx')}")
                print(f"   File Hash: {result.get('file_hash')}")
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
    print("🚀 Web Interface Upload Test")
    print("=" * 50)
    
    success = test_web_upload()
    
    if success:
        print("\n🎉 Web interface upload test completed successfully!")
        print("✅ Your web interface should work correctly!")
    else:
        print("\n❌ Web interface upload test failed!")
        print("🔧 Check the error messages above for details")

if __name__ == "__main__":
    main()
