// SPDX-License-Identifier: MIT
pragma solidity >=0.4.21;

import "./Roles.sol";

contract FileVerificationContract {
    using Roles for Roles.Role;

    Roles.Role private admin;
    Roles.Role private doctor;
    Roles.Role private lab;

    struct FileRecord {
        uint256 fileId;
        string fileName;
        string fileHash;        // SHA-256 hash of the file
        string ipfsHash;        // IPFS hash for file storage
        string fileType;
        uint256 fileSize;
        address uploadedBy;
        address patientId;
        uint256 timestamp;
        bool isValid;
        string metadata;        // Additional metadata as JSON string
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

    mapping(uint256 => FileRecord) public files;
    mapping(uint256 => VerificationLog) public verificationLogs;
    mapping(address => uint256[]) public userFiles;
    mapping(string => uint256) public hashToFileId;

    uint256 public fileCounter;
    uint256 public verificationCounter;

    event FileUploaded(uint256 indexed fileId, string fileName, string fileHash, address indexed uploadedBy);
    event FileVerified(uint256 indexed fileId, bool isMatch, address indexed verifiedBy);
    event FileInvalidated(uint256 indexed fileId, address indexed invalidatedBy);

    constructor() {
        admin.add(msg.sender);
    }

    // Modifiers
    modifier onlyAdmin() {
        require(admin.has(msg.sender), "Only admin can perform this action");
        _;
    }

    modifier onlyAuthorized() {
        require(admin.has(msg.sender) || doctor.has(msg.sender) || lab.has(msg.sender), "Not authorized");
        _;
    }

    // Admin functions
    function addAdmin(address newAdmin) public onlyAdmin {
        admin.add(newAdmin);
    }

    function addDoctor(address doctorAddress) public onlyAdmin {
        doctor.add(doctorAddress);
    }

    function addLab(address labAddress) public onlyAdmin {
        lab.add(labAddress);
    }

    // File upload function
    function uploadFile(
        string memory _fileName,
        string memory _fileHash,
        string memory _ipfsHash,
        string memory _fileType,
        uint256 _fileSize,
        address _patientId,
        string memory _metadata
    ) public onlyAuthorized returns (uint256) {
        require(bytes(_fileHash).length > 0, "File hash cannot be empty");
        require(hashToFileId[_fileHash] == 0, "File with this hash already exists");

        fileCounter++;
        
        FileRecord storage newFile = files[fileCounter];
        newFile.fileId = fileCounter;
        newFile.fileName = _fileName;
        newFile.fileHash = _fileHash;
        newFile.ipfsHash = _ipfsHash;
        newFile.fileType = _fileType;
        newFile.fileSize = _fileSize;
        newFile.uploadedBy = msg.sender;
        newFile.patientId = _patientId;
        newFile.timestamp = block.timestamp;
        newFile.isValid = true;
        newFile.metadata = _metadata;

        userFiles[msg.sender].push(fileCounter);
        hashToFileId[_fileHash] = fileCounter;

        emit FileUploaded(fileCounter, _fileName, _fileHash, msg.sender);
        
        return fileCounter;
    }

    // File verification function
    function verifyFile(
        uint256 _fileId,
        string memory _currentHash,
        string memory _notes
    ) public onlyAuthorized returns (bool) {
        require(files[_fileId].isValid, "File does not exist or is invalid");
        
        FileRecord storage file = files[_fileId];
        bool isMatch = keccak256(abi.encodePacked(file.fileHash)) == keccak256(abi.encodePacked(_currentHash));
        
        verificationCounter++;
        
        VerificationLog storage log = verificationLogs[verificationCounter];
        log.logId = verificationCounter;
        log.fileId = _fileId;
        log.originalHash = file.fileHash;
        log.verifiedHash = _currentHash;
        log.isMatch = isMatch;
        log.verifiedBy = msg.sender;
        log.timestamp = block.timestamp;
        log.notes = _notes;

        emit FileVerified(_fileId, isMatch, msg.sender);
        
        return isMatch;
    }

    // Get file information
    function getFile(uint256 _fileId) public view returns (
        string memory fileName,
        string memory fileHash,
        string memory ipfsHash,
        string memory fileType,
        uint256 fileSize,
        address uploadedBy,
        address patientId,
        uint256 timestamp,
        bool isValid,
        string memory metadata
    ) {
        FileRecord storage file = files[_fileId];
        return (
            file.fileName,
            file.fileHash,
            file.ipfsHash,
            file.fileType,
            file.fileSize,
            file.uploadedBy,
            file.patientId,
            file.timestamp,
            file.isValid,
            file.metadata
        );
    }

    // Get verification log
    function getVerificationLog(uint256 _logId) public view returns (
        uint256 fileId,
        string memory originalHash,
        string memory verifiedHash,
        bool isMatch,
        address verifiedBy,
        uint256 timestamp,
        string memory notes
    ) {
        VerificationLog storage log = verificationLogs[_logId];
        return (
            log.fileId,
            log.originalHash,
            log.verifiedHash,
            log.isMatch,
            log.verifiedBy,
            log.timestamp,
            log.notes
        );
    }

    // Get files uploaded by a user
    function getUserFiles(address _user) public view returns (uint256[] memory) {
        return userFiles[_user];
    }

    // Get file ID by hash
    function getFileIdByHash(string memory _hash) public view returns (uint256) {
        return hashToFileId[_hash];
    }

    // Invalidate file (admin only)
    function invalidateFile(uint256 _fileId) public onlyAdmin {
        require(files[_fileId].isValid, "File is already invalid");
        files[_fileId].isValid = false;
        emit FileInvalidated(_fileId, msg.sender);
    }

    // Check if address is authorized
    function isAuthorized(address _address) public view returns (bool) {
        return admin.has(_address) || doctor.has(_address) || lab.has(_address);
    }

    // Get total file count
    function getTotalFiles() public view returns (uint256) {
        return fileCounter;
    }

    // Get total verification count
    function getTotalVerifications() public view returns (uint256) {
        return verificationCounter;
    }
}
