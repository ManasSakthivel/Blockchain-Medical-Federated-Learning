#!/usr/bin/env python3
"""
Test script for the Secure File Upload & Verification System
This script demonstrates the key features of the blockchain-based file verification system.
"""

import os
import hashlib
import tempfile
from datetime import datetime

def create_test_file(content, filename="test_file.txt"):
    """Create a test file with given content"""
    with open(filename, 'w') as f:
        f.write(content)
    return filename

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def create_tampered_file(original_file, tampered_file):
    """Create a tampered version of the original file"""
    with open(original_file, 'rb') as f:
        content = f.read()
    
    # Add a single byte to tamper with the file
    tampered_content = content + b'\x00'
    
    with open(tampered_file, 'wb') as f:
        f.write(tampered_content)
    
    return tampered_file

def demonstrate_file_verification():
    """Demonstrate the file verification system"""
    print("🔐 Secure File Upload & Verification System Demo")
    print("=" * 60)
    
    # Create test files
    print("\n1. Creating test files...")
    original_content = "This is a medical record file for patient John Doe.\nDiagnosis: Healthy\nDate: 2024-01-15"
    original_file = create_test_file(original_content, "medical_record_original.txt")
    tampered_file = create_tampered_file(original_file, "medical_record_tampered.txt")
    
    print(f"✅ Created original file: {original_file}")
    print(f"✅ Created tampered file: {tampered_file}")
    
    # Calculate hashes
    print("\n2. Calculating file hashes...")
    original_hash = calculate_file_hash(original_file)
    tampered_hash = calculate_file_hash(tampered_file)
    
    print(f"📄 Original file hash: {original_hash}")
    print(f"⚠️  Tampered file hash: {tampered_hash}")
    
    # Demonstrate hash comparison
    print("\n3. Hash comparison results...")
    hashes_match = original_hash == tampered_hash
    
    if hashes_match:
        print("❌ ERROR: Hashes match - this should not happen!")
    else:
        print("✅ SUCCESS: Hashes do not match - tampering detected!")
        print("🔍 Even a single byte change produces a completely different hash")
    
    # Show hash differences
    print(f"\n4. Hash analysis...")
    print(f"Original hash length: {len(original_hash)} characters")
    print(f"Tampered hash length: {len(tampered_hash)} characters")
    print(f"Hash similarity: {sum(1 for a, b in zip(original_hash, tampered_hash) if a == b)}/{len(original_hash)} characters match")
    
    # Demonstrate blockchain simulation
    print("\n5. Blockchain simulation...")
    print("📋 Storing original hash on blockchain...")
    print(f"   File ID: 1")
    print(f"   File Name: {original_file}")
    print(f"   File Hash: {original_hash}")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print(f"   Status: ✅ Stored on blockchain")
    
    # Simulate verification process
    print("\n6. File verification process...")
    print("🔍 Verifying original file...")
    current_hash = calculate_file_hash(original_file)
    verification_result = current_hash == original_hash
    
    if verification_result:
        print("✅ VERIFICATION SUCCESS: File integrity confirmed")
        print("   - Current hash matches blockchain record")
        print("   - File has not been tampered with")
    else:
        print("❌ VERIFICATION FAILED: File integrity compromised")
        print("   - Current hash does not match blockchain record")
        print("   - File may have been modified")
    
    print("\n🔍 Verifying tampered file...")
    current_hash = calculate_file_hash(tampered_file)
    verification_result = current_hash == original_hash
    
    if verification_result:
        print("❌ ERROR: Verification should have failed!")
    else:
        print("✅ TAMPERING DETECTED: File integrity compromised")
        print("   - Current hash does not match blockchain record")
        print("   - File has been modified since upload")
    
    # Cleanup
    print("\n7. Cleanup...")
    os.remove(original_file)
    os.remove(tampered_file)
    print("✅ Test files cleaned up")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)
    print("✅ SHA-256 hashing provides cryptographic file integrity")
    print("✅ Blockchain storage ensures immutable hash records")
    print("✅ Tampering detection works with minimal file changes")
    print("✅ Verification process is automated and reliable")
    print("✅ Audit trail is maintained on blockchain")
    print("\n🎯 Key Benefits:")
    print("   - Prevents unauthorized file modifications")
    print("   - Provides verifiable file authenticity")
    print("   - Creates immutable audit trail")
    print("   - Enables independent verification")
    print("   - Suitable for healthcare and legal compliance")

def demonstrate_security_features():
    """Demonstrate additional security features"""
    print("\n🔒 Additional Security Features")
    print("=" * 40)
    
    # File type validation
    print("\n1. File Type Validation:")
    allowed_extensions = {'.txt', '.pdf', '.doc', '.jpg', '.png'}
    test_files = ['document.txt', 'image.jpg', 'script.exe', 'data.csv']
    
    for file in test_files:
        ext = os.path.splitext(file)[1].lower()
        is_allowed = ext in allowed_extensions
        status = "✅ ALLOWED" if is_allowed else "❌ BLOCKED"
        print(f"   {file}: {status}")
    
    # File size validation
    print("\n2. File Size Validation:")
    max_size = 50 * 1024 * 1024  # 50MB
    test_sizes = [1024, 1024*1024, 100*1024*1024, 60*1024*1024]
    
    for size in test_sizes:
        is_valid = size <= max_size
        status = "✅ ACCEPTED" if is_valid else "❌ REJECTED"
        size_mb = size / (1024 * 1024)
        print(f"   {size_mb:.1f}MB: {status}")
    
    # Access control simulation
    print("\n3. Access Control:")
    user_roles = ['admin', 'doctor', 'lab', 'patient']
    permissions = {
        'admin': ['upload', 'verify', 'view_all', 'invalidate'],
        'doctor': ['upload', 'verify', 'view_patient'],
        'lab': ['upload', 'verify', 'view_patient'],
        'patient': ['verify', 'view_own']
    }
    
    for role in user_roles:
        perms = permissions.get(role, [])
        print(f"   {role.title()}: {', '.join(perms)}")

if __name__ == "__main__":
    try:
        demonstrate_file_verification()
        demonstrate_security_features()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        print("\nTo use the full system:")
        print("1. Start Ganache blockchain")
        print("2. Start IPFS daemon")
        print("3. Run: python run.py")
        print("4. Navigate to: http://localhost:5000")
        print("5. Login and access File Verification features")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check your setup and try again.")
