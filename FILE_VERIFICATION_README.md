# Secure File Upload & Verification System

## Overview

This system implements a secure file upload mechanism with blockchain-based integrity verification for the EHR (Electronic Health Record) project. It ensures file authenticity, prevents tampering, and provides an immutable audit trail using blockchain technology.

## Features

### 🔐 Security Features
- **SHA-256 Hashing**: Cryptographic hash function for file integrity
- **Blockchain Storage**: Immutable storage of file hashes on blockchain
- **IPFS Integration**: Decentralized file storage
- **Tamper Detection**: Automatic detection of file modifications
- **Audit Trail**: Complete verification history on blockchain

### 📁 File Management
- **Secure Upload**: Files are hashed and stored with metadata
- **Verification System**: Compare current file hash with blockchain record
- **File Categories**: Medical records, lab reports, prescriptions, etc.
- **Access Control**: Role-based permissions (admin, doctor, lab)

### 🔍 Verification Process
- **Hash Calculation**: SHA-256 hash of uploaded file
- **Blockchain Lookup**: Retrieve original hash from blockchain
- **Comparison**: Compare hashes to detect tampering
- **Logging**: Record verification results on blockchain

## Technical Architecture

### Smart Contracts

#### FileVerificationContract.sol
```solidity
// Main contract for file verification
contract FileVerificationContract {
    struct FileRecord {
        uint256 fileId;
        string fileName;
        string fileHash;        // SHA-256 hash
        string ipfsHash;        // IPFS storage hash
        string fileType;
        uint256 fileSize;
        address uploadedBy;
        address patientId;
        uint256 timestamp;
        bool isValid;
        string metadata;
    }
    
    struct VerificationLog {
        uint256 logId;
        uint256 fileId;
        string originalHash;
        string verifiedHash;
        bool isMatch;
        address verifiedBy;
        uint256 timestamp;
        string notes;
    }
}
```

### Services

#### FileVerificationService
- Handles file upload with integrity checking
- Manages IPFS storage and blockchain interactions
- Provides verification functionality
- Creates tampered file demos for testing

#### BlockchainService (Enhanced)
- Upload file information to blockchain
- Retrieve file records and verification logs
- Manage user file lists
- Handle verification transactions

#### IPFSService
- Upload files to IPFS network
- Retrieve files from IPFS
- Pin files to prevent garbage collection

## API Endpoints

### File Upload
```
POST /file-verification/upload
POST /file-verification/api/upload
```

### File Verification
```
POST /file-verification/verify
POST /file-verification/api/verify
```

### File Management
```
GET /file-verification/files
GET /file-verification/file/<file_id>
GET /file-verification/download/<file_id>
```

### Demo & Testing
```
GET /file-verification/demo/tamper
```

## Usage Guide

### 1. Upload a File

1. Navigate to **File Verification > Upload File**
2. Select a patient from the dropdown
3. Choose file category (medical, lab_report, etc.)
4. Add description and select file
5. Click "Upload Securely"

The system will:
- Calculate SHA-256 hash of the file
- Upload file to IPFS
- Store hash and metadata on blockchain
- Return file ID for future reference

### 2. Verify File Integrity

1. Navigate to **File Verification > Verify File**
2. Enter the File ID
3. Upload the file you want to verify
4. Add optional verification notes
5. Click "Verify Integrity"

The system will:
- Calculate hash of uploaded file
- Retrieve original hash from blockchain
- Compare hashes and show result
- Log verification on blockchain

### 3. View File Details

1. Navigate to **File Verification > My Files**
2. Click on any file to view details
3. See verification history and blockchain information

### 4. Demo Tampering Detection

1. Navigate to **File Verification > Tampering Demo**
2. View original vs tampered file hashes
3. Understand how tampering detection works

## Security Implementation

### SHA-256 Hashing
```python
def calculate_file_hash(self, file_path):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
```

### Blockchain Integration
```python
def upload_file_to_blockchain(self, filename, file_hash, ipfs_hash, file_type, file_size, patient_id, metadata):
    """Upload file information to blockchain"""
    # Smart contract call to store file record
    return contract.functions.uploadFile(
        filename, file_hash, ipfs_hash, file_type, file_size, patient_id, metadata
    ).transact()
```

### IPFS Storage
```python
def upload_file(self, file_path):
    """Upload file to IPFS"""
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(f"{self.ipfs_url}/api/v0/add", files=files)
        return response.json()['Hash']
```

## File Types Supported

- **Images**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **Documents**: PDF, DOC, DOCX, TXT, CSV, XLSX, XLS
- **Archives**: ZIP, RAR
- **Videos**: MP4, AVI, MOV
- **Maximum Size**: 50MB per file

## Blockchain Network

### Ganache (Local Development)
- Network: Local blockchain for development
- Contract: FileVerificationContract
- Features: File storage, verification logging, access control

### Production Considerations
- Use Ethereum mainnet or testnet
- Implement proper gas management
- Consider Layer 2 solutions for cost optimization

## IPFS Configuration

### Local IPFS Node
```python
IPFSService(ipfs_url="http://127.0.0.1:5001")
```

### Public IPFS Gateway
- Use public gateways for production
- Implement redundancy and backup strategies
- Consider IPFS pinning services

## Access Control

### Role-Based Permissions
- **Admin**: Full access to all features
- **Doctor**: Upload and verify files for patients
- **Lab**: Upload lab reports and medical images
- **Patient**: View and verify their own files

### Smart Contract Access Control
```solidity
modifier onlyAuthorized() {
    require(admin.has(msg.sender) || doctor.has(msg.sender) || lab.has(msg.sender), "Not authorized");
    _;
}
```

## Audit Trail

### Verification Logs
Every file verification is logged on the blockchain with:
- Verification timestamp
- Verifier address
- Hash comparison result
- Verification notes
- Log ID for tracking

### Blockchain Events
```solidity
event FileUploaded(uint256 indexed fileId, string fileName, string fileHash, address indexed uploadedBy);
event FileVerified(uint256 indexed fileId, bool isMatch, address indexed verifiedBy);
event FileInvalidated(uint256 indexed fileId, address indexed invalidatedBy);
```

## Testing & Demo

### Tampering Demo
The system includes a demonstration that:
1. Creates an original file
2. Creates a tampered version (adds one byte)
3. Shows hash differences
4. Demonstrates tampering detection

### Test Scenarios
- Upload various file types
- Verify file integrity
- Test tampering detection
- Check audit trail
- Verify access controls

## Deployment

### Prerequisites
1. Ganache running on localhost:7545
2. IPFS daemon running on localhost:5001
3. Truffle framework installed
4. Python dependencies installed

### Migration Steps
```bash
# Deploy smart contracts
truffle migrate --reset

# Start IPFS daemon
ipfs daemon

# Run Flask application
python run.py
```

### Environment Variables
```bash
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///ehr.db
IPFS_URL=http://127.0.0.1:5001
BLOCKCHAIN_URL=http://127.0.0.1:7545
```

## Monitoring & Maintenance

### Blockchain Monitoring
- Monitor gas usage for transactions
- Track contract events
- Monitor file upload/verification rates

### IPFS Monitoring
- Monitor storage usage
- Check file availability
- Monitor pinning status

### System Health Checks
- Blockchain connectivity
- IPFS connectivity
- Database performance
- File upload success rates

## Security Best Practices

### File Security
- Validate file types and sizes
- Scan for malware (consider implementing)
- Encrypt sensitive files
- Implement file retention policies

### Blockchain Security
- Use secure private keys
- Implement proper access controls
- Monitor for suspicious transactions
- Regular security audits

### Network Security
- Use HTTPS for all communications
- Implement rate limiting
- Monitor for DDoS attacks
- Regular security updates

## Future Enhancements

### Planned Features
- **Digital Signatures**: Add cryptographic signatures
- **Batch Verification**: Verify multiple files at once
- **Advanced Analytics**: File usage and verification statistics
- **Mobile App**: Native mobile application
- **API Rate Limiting**: Prevent abuse
- **File Encryption**: End-to-end encryption

### Scalability Improvements
- **Layer 2 Solutions**: Reduce blockchain costs
- **IPFS Clustering**: Improve file availability
- **Caching**: Implement Redis caching
- **Load Balancing**: Multiple server instances

## Troubleshooting

### Common Issues

#### File Upload Fails
- Check file size (max 50MB)
- Verify file type is supported
- Ensure IPFS daemon is running
- Check blockchain connectivity

#### Verification Fails
- Verify File ID is correct
- Check file hasn't been modified
- Ensure blockchain is accessible
- Check network connectivity

#### IPFS Issues
- Restart IPFS daemon
- Check IPFS configuration
- Verify network connectivity
- Check disk space

### Error Messages
- "File too large": Reduce file size
- "File type not allowed": Use supported format
- "Blockchain connection failed": Check Ganache
- "IPFS upload failed": Check IPFS daemon

## Support & Documentation

### Additional Resources
- [Blockchain Documentation](https://ethereum.org/developers/)
- [IPFS Documentation](https://docs.ipfs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Truffle Documentation](https://www.trufflesuite.com/docs)

### Contact
For technical support or questions about the file verification system, please refer to the main project documentation or contact the development team.

---

**Note**: This system is designed for educational and demonstration purposes. For production use in healthcare environments, additional security measures, compliance checks, and regulatory approvals may be required.
