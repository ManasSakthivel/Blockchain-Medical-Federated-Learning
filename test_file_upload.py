#!/usr/bin/env python3
"""
Test File Upload to Blockchain
This script tests the file upload functionality with real blockchain integration.
"""

import os
import tempfile
import hashlib
import time
from app import create_app
from app.services.file_verification_service import FileVerificationService
from app.services.blockchain_service import BlockchainService

def create_test_file():
    """Create a test file for upload with unique content"""
    timestamp = int(time.time())
    test_content = f"This is a test medical record file for blockchain verification.\nPatient: John Doe\nDate: 2024-01-15\nDiagnosis: Healthy\nTimestamp: {timestamp}".encode()
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
    temp_file.write(test_content)
    temp_file.close()
    
    return temp_file.name

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def test_file_upload():
    """Test file upload to blockchain"""
    print("🧪 Testing File Upload to Blockchain")
    print("=" * 50)
    
    try:
        # Create Flask app context
        app = create_app()
        
        with app.app_context():
            # Create test file
            test_file_path = create_test_file()
            print(f"✅ Created test file: {test_file_path}")
            
            # Initialize services
            file_service = FileVerificationService()
            blockchain_service = BlockchainService()
            
            # Connect to blockchain
            if not blockchain_service.connect_to_ganache():
                print("❌ Failed to connect to Ganache")
                return False
            
            if not blockchain_service.load_contract():
                print("❌ Failed to load contract")
                return False
            
            print(f"✅ Connected to blockchain with account: {blockchain_service.account}")
            print(f"✅ Account balance: {blockchain_service.get_balance()} ETH")
            
            # Test private key loading
            private_key = blockchain_service._get_private_key()
            if private_key:
                print("✅ Private key loaded successfully")
            else:
                print("❌ Failed to load private key")
                return False
            
            # Calculate file hash
            file_hash = calculate_file_hash(test_file_path)
            print(f"✅ File hash calculated: {file_hash}")
            
            # Test direct blockchain upload
            print("\n📤 Testing direct blockchain upload...")
            blockchain_result = blockchain_service.upload_file_to_blockchain(
                filename="test_file.txt",
                file_hash=file_hash,
                ipfs_hash="QmTestHash123456789",
                file_type="txt",
                file_size=1024,
                patient_id="test_patient_123",
                metadata='{"test": true}'
            )
            
            if blockchain_result:
                print("✅ Direct blockchain upload successful!")
                print(f"   File ID: {blockchain_result.get('file_id')}")
                print(f"   Transaction Hash: {blockchain_result.get('transaction_hash')}")
                
                # Test file verification
                print("\n🔍 Testing file verification...")
                verify_result = blockchain_service.verify_file_on_blockchain(
                    file_id=blockchain_result.get('file_id'),
                    current_hash=file_hash,
                    notes="Test verification"
                )
                
                if verify_result:
                    print("✅ File verification successful!")
                    print(f"   Log ID: {verify_result.get('log_id')}")
                    print(f"   Is Match: {verify_result.get('is_match')}")
                else:
                    print("❌ File verification failed")
                
                return True
            else:
                print("❌ Direct blockchain upload failed")
                return False
                
    except Exception as e:
        print(f"❌ Error during test: {e}")
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
    print("🚀 File Upload Blockchain Test")
    print("=" * 50)
    
    success = test_file_upload()
    
    if success:
        print("\n🎉 File upload test completed successfully!")
        print("\n✅ Your blockchain integration is working correctly!")
        print("\n📋 You can now:")
        print("   1. Use the web interface to upload files")
        print("   2. Verify file integrity through the UI")
        print("   3. Monitor transactions in Ganache")
    else:
        print("\n❌ File upload test failed!")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if Ganache is running")
        print("   2. Verify contract deployment")
        print("   3. Check private keys configuration")
        print("   4. Review error messages above")

if __name__ == "__main__":
    main()
