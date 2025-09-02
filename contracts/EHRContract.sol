// SPDX-License-Identifier: MIT
pragma solidity >=0.4.21;

import "./Roles.sol";

contract EHRContract {
    using Roles for Roles.Role;

    Roles.Role private admin;
    Roles.Role private doctor;
    Roles.Role private patient;

    struct Doctor {
        address id;
        string drHash;
        string specialization;
        string licenseNumber;
    }

    struct Patient {
        address id;
        string patientHash;
        string medicalHistory;
    }

    struct MedicalRecord {
        uint256 recordId;
        address patientId;
        address doctorId;
        string recordHash;
        string ipfsHash;
        uint256 timestamp;
        bool isValid;
    }

    mapping(address => Doctor) public doctors;
    mapping(address => Patient) public patients;
    mapping(uint256 => MedicalRecord) public medicalRecords;
    mapping(address => uint256[]) public patientRecords;

    address[] public doctorIds;
    address[] public patientIds;
    uint256 public recordCounter;

    event DoctorAdded(address indexed doctorId, string drHash);
    event PatientAdded(address indexed patientId, string patientHash);
    event RecordAdded(uint256 indexed recordId, address indexed patientId, address indexed doctorId);
    event RecordUpdated(uint256 indexed recordId);

    constructor() {
        admin.add(msg.sender);
    }

    // Admin functions
    function isAdmin() public view returns (bool) {
        return admin.has(msg.sender);
    }

    function addAdmin(address newAdmin) public {
        require(admin.has(msg.sender), "Only admin can add new admin");
        admin.add(newAdmin);
    }

    // Doctor functions
    function addDoctor(address drId, string memory _drHash, string memory _specialization, string memory _licenseNumber) public {
        require(admin.has(msg.sender), "Only admin can add doctors");

        Doctor storage drInfo = doctors[drId];
        drInfo.id = drId;
        drInfo.drHash = _drHash;
        drInfo.specialization = _specialization;
        drInfo.licenseNumber = _licenseNumber;
        
        doctorIds.push(drId);
        doctor.add(drId);

        emit DoctorAdded(drId, _drHash);
    }

    function getAllDoctors() public view returns (address[] memory) {
        return doctorIds;
    }

    function getDoctor(address _id) public view returns (string memory, string memory, string memory) {
        Doctor storage dr = doctors[_id];
        return (dr.drHash, dr.specialization, dr.licenseNumber);
    }

    function isDoctor(address id) public view returns (bool) {
        return doctor.has(id);
    }

    // Patient functions
    function addPatient(address patientId, string memory _patientHash, string memory _medicalHistory) public {
        require(admin.has(msg.sender) || doctor.has(msg.sender), "Only admin or doctor can add patients");

        Patient storage patientInfo = patients[patientId];
        patientInfo.id = patientId;
        patientInfo.patientHash = _patientHash;
        patientInfo.medicalHistory = _medicalHistory;
        
        patientIds.push(patientId);
        patient.add(patientId);
    }

    function getAllPatients() public view returns (address[] memory) {
        return patientIds;
    }

    function getPatient(address _id) public view returns (string memory, string memory) {
        Patient storage p = patients[_id];
        return (p.patientHash, p.medicalHistory);
    }

    function isPatient(address id) public view returns (bool) {
        return patient.has(id);
    }

    // Medical Record functions
    function addMedicalRecord(address _patientId, address _doctorId, string memory _recordHash, string memory _ipfsHash) public {
        require(doctor.has(msg.sender), "Only doctors can add medical records");
        require(patient.has(_patientId), "Patient must be registered");

        recordCounter++;
        
        MedicalRecord storage record = medicalRecords[recordCounter];
        record.recordId = recordCounter;
        record.patientId = _patientId;
        record.doctorId = _doctorId;
        record.recordHash = _recordHash;
        record.ipfsHash = _ipfsHash;
        record.timestamp = block.timestamp;
        record.isValid = true;

        patientRecords[_patientId].push(recordCounter);

        emit RecordAdded(recordCounter, _patientId, _doctorId);
    }

    function getMedicalRecord(uint256 _recordId) public view returns (
        address patientId,
        address doctorId,
        string memory recordHash,
        string memory ipfsHash,
        uint256 timestamp,
        bool isValid
    ) {
        MedicalRecord storage record = medicalRecords[_recordId];
        return (
            record.patientId,
            record.doctorId,
            record.recordHash,
            record.ipfsHash,
            record.timestamp,
            record.isValid
        );
    }

    function getPatientRecords(address _patientId) public view returns (uint256[] memory) {
        return patientRecords[_patientId];
    }

    function updateMedicalRecord(uint256 _recordId, string memory _newRecordHash, string memory _newIpfsHash) public {
        require(doctor.has(msg.sender), "Only doctors can update medical records");
        
        MedicalRecord storage record = medicalRecords[_recordId];
        require(record.isValid, "Record does not exist or is invalid");
        require(record.doctorId == msg.sender, "Only the original doctor can update the record");

        record.recordHash = _newRecordHash;
        record.ipfsHash = _newIpfsHash;
        record.timestamp = block.timestamp;

        emit RecordUpdated(_recordId);
    }

    function invalidateRecord(uint256 _recordId) public {
        require(admin.has(msg.sender), "Only admin can invalidate records");
        
        MedicalRecord storage record = medicalRecords[_recordId];
        require(record.isValid, "Record is already invalid");

        record.isValid = false;
    }
} 