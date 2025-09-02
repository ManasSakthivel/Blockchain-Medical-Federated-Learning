import hashlib
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import current_app
from app.services.blockchain_service import BlockchainService
from app.services.ipfs_service import IPFSService

class FileVerificationService:
    def __init__(self):
        self.blockchain_service = BlockchainService()
        # Initialize blockchain connection
        if not self.blockchain_service.connect_to_ganache():
            print("❌ Failed to connect to Ganache")
        if not self.blockchain_service.load_contract():
            print("❌ Failed to load contract")
        
        self.ipfs_service = IPFSService()
        self.allowed_extensions = {
            'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf', 'doc', 'docx', 
            'txt', 'csv', 'xlsx', 'xls', 'zip', 'rar', 'mp4', 'avi', 'mov'
        }
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
    
    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of a file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating file hash: {e}")
            return None
    
    def calculate_file_hash_from_bytes(self, file_bytes):
        """Calculate SHA-256 hash from file bytes"""
        try:
            sha256_hash = hashlib.sha256()
            sha256_hash.update(file_bytes)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating file hash from bytes: {e}")
            return None
    
    def upload_file_secure(self, file, patient_id=None, metadata=None):
        """
        Upload file with integrity verification
        Returns: dict with upload status and blockchain info
        """
        try:
            # Validate file
            if not file or not file.filename:
                return {"success": False, "error": "No file provided"}
            
            if not self.allowed_file(file.filename):
                return {"success": False, "error": "File type not allowed"}
            
            # Read file content and calculate hash
            file_content = file.read()
            file_size = len(file_content)
            
            if file_size > self.max_file_size:
                return {"success": False, "error": "File too large (max 50MB)"}
            
            # Calculate hash
            file_hash = self.calculate_file_hash_from_bytes(file_content)
            if not file_hash:
                return {"success": False, "error": "Failed to calculate file hash"}
            
            # Save file locally first
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            upload_dir = os.path.join(current_app.root_path, 'static', 'uploads', 'verified_files')
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, filename)
            
            # Write file content
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Upload to IPFS
            print(f"🔍 Uploading to IPFS: {file_path}")
            ipfs_hash = self.ipfs_service.upload_file(file_path)
            print(f"🔍 IPFS hash result: {ipfs_hash}")
            if not ipfs_hash:
                return {"success": False, "error": "Failed to upload to IPFS"}
            
            # Pin file on IPFS
            print(f"🔍 Pinning file on IPFS: {ipfs_hash}")
            pin_result = self.ipfs_service.pin_file(ipfs_hash)
            print(f"🔍 Pin result: {pin_result}")
            
            # Prepare metadata
            file_metadata = {
                "original_filename": file.filename,
                "file_size": file_size,
                "upload_timestamp": datetime.now().isoformat(),
                "uploaded_by": "system",  # Will be updated with actual user
                "patient_id": patient_id,
                "custom_metadata": metadata or {}
            }
            
            # Store on blockchain
            try:
                blockchain_result = self.blockchain_service.upload_file_to_blockchain(
                    filename=filename,
                    file_hash=file_hash,
                    ipfs_hash=ipfs_hash,
                    file_type=filename.rsplit('.', 1)[1].lower(),
                    file_size=file_size,
                    patient_id=patient_id,
                    metadata=json.dumps(file_metadata)
                )
                    
                if not blockchain_result:
                    return {"success": False, "error": "Failed to store on blockchain"}
                    
            except Exception as e:
                import traceback
                print(f"Blockchain upload error: {e}")
                print(f"Full traceback: {traceback.format_exc()}")
                return {"success": False, "error": f"Failed to store on blockchain: {str(e)}"}
            
            return {
                "success": True,
                "file_id": blockchain_result.get("file_id"),
                "file_hash": file_hash,
                "ipfs_hash": ipfs_hash,
                "local_path": f"uploads/verified_files/{filename}",
                "blockchain_tx": blockchain_result.get("transaction_hash"),
                "metadata": file_metadata
            }
            
        except Exception as e:
            print(f"Error in secure file upload: {e}")
            return {"success": False, "error": str(e)}
    
    def verify_file_integrity(self, file_id, file_path=None, file_bytes=None):
        """
        Verify file integrity against blockchain record
        Returns: dict with verification status
        """
        try:
            # Get file record from blockchain
            file_record = self.blockchain_service.get_file_record(file_id)
            if not file_record:
                return {"success": False, "error": "File record not found on blockchain"}
            
            # Calculate current hash
            current_hash = None
            if file_path and os.path.exists(file_path):
                current_hash = self.calculate_file_hash(file_path)
            elif file_bytes:
                current_hash = self.calculate_file_hash_from_bytes(file_bytes)
            else:
                return {"success": False, "error": "No file provided for verification"}
            
            if not current_hash:
                return {"success": False, "error": "Failed to calculate current file hash"}
            
            # Compare hashes
            original_hash = file_record.get("file_hash")
            is_match = current_hash.lower() == original_hash.lower()
            
            # Log verification on blockchain
            verification_result = self.blockchain_service.verify_file_on_blockchain(
                file_id=file_id,
                current_hash=current_hash,
                notes=f"Verification performed at {datetime.now().isoformat()}"
            )
            
            return {
                "success": True,
                "is_match": is_match,
                "original_hash": original_hash,
                "current_hash": current_hash,
                "file_record": file_record,
                "verification_log_id": verification_result.get("log_id") if verification_result else None
            }
            
        except Exception as e:
            print(f"Error in file integrity verification: {e}")
            return {"success": False, "error": str(e)}
    
    def get_file_from_ipfs(self, ipfs_hash):
        """Retrieve file from IPFS"""
        try:
            return self.ipfs_service.get_file(ipfs_hash)
        except Exception as e:
            print(f"Error getting file from IPFS: {e}")
            return None
    
    def get_user_files(self, user_address):
        """Get all files uploaded by a user"""
        try:
            return self.blockchain_service.get_user_files(user_address)
        except Exception as e:
            print(f"Error getting user files: {e}")
            return []
    
    def get_verification_logs(self, file_id):
        """Get verification logs for a file"""
        try:
            return self.blockchain_service.get_file_verification_logs(file_id)
        except Exception as e:
            print(f"Error getting verification logs: {e}")
            return []
    
    def create_tampered_file_demo(self, original_file_path):
        """
        Create a tampered version of a file for demonstration
        Returns: path to tampered file
        """
        try:
            # Read original file
            with open(original_file_path, 'rb') as f:
                content = f.read()
            
            # Create tampered content (add a byte at the end)
            tampered_content = content + b'\x00'
            
            # Save tampered file
            tampered_path = original_file_path.replace('.', '_tampered.')
            with open(tampered_path, 'wb') as f:
                f.write(tampered_content)
            
            return tampered_path
            
        except Exception as e:
            print(f"Error creating tampered file: {e}")
            return None
