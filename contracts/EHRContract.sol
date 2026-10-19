// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./Roles.sol";

interface IConsent {
    function hasConsent(address patient, address recipient, string calldata dataType)
        external
        view
        returns (bool);
}

/**
 * @title EHRContract
 * @notice Registry of doctor/patient pseudonyms and record *digests*.
 *
 * Nothing human-readable about a patient is stored: no medical history, name or
 * licence number. Only 32-byte digests (computed off-chain, ideally keyed) and
 * an optional content id of an ENCRYPTED off-chain object.
 *
 * Not used by the Flask application in this repository (the app anchors via
 * ConsentContract / FileVerificationContract / ErasureLogContract); kept as a
 * reference design with its authorisation rules covered by tests.
 */
contract EHRContract {
    using Roles for Roles.Role;

    Roles.Role private admin;
    Roles.Role private doctor;
    Roles.Role private patient;

    struct MedicalRecord {
        uint256 recordId;
        address patientId;
        address doctorId;
        bytes32 recordHash;
        string ipfsHash; // CID of an encrypted object; may be empty
        uint256 timestamp;
        bool isValid;
    }

    mapping(address => bytes32) public doctorHash;
    mapping(address => bytes32) public patientHash;
    mapping(uint256 => MedicalRecord) public medicalRecords;
    mapping(address => uint256[]) private patientRecords;

    address[] private doctorIds;
    address[] private patientIds;
    uint256 public recordCounter;

    /// Consent gate. MUST be set before records can be added (fail closed).
    address public consentContractAddress;

    event DoctorAdded(address indexed doctorId);
    event PatientAdded(address indexed patientId);
    event RecordAdded(uint256 indexed recordId, address indexed patientId, address indexed doctorId);
    event RecordUpdated(uint256 indexed recordId);
    event RecordInvalidated(uint256 indexed recordId);

    constructor() {
        admin.add(msg.sender);
    }

    function isAdmin() public view returns (bool) {
        return admin.has(msg.sender);
    }

    function addAdmin(address newAdmin) public {
        require(admin.has(msg.sender), "Only admin can add new admin");
        admin.add(newAdmin);
    }

    function setConsentContract(address _addr) public {
        require(admin.has(msg.sender), "Only admin can set consent contract");
        require(_addr != address(0), "Consent contract required");
        consentContractAddress = _addr;
    }

    function addDoctor(address drId, bytes32 _drHash) public {
        require(admin.has(msg.sender), "Only admin can add doctors");
        doctor.add(drId); // reverts on zero address / duplicate
        doctorHash[drId] = _drHash;
        doctorIds.push(drId);
        emit DoctorAdded(drId);
    }

    function addPatient(address patientId, bytes32 _patientHash) public {
        require(admin.has(msg.sender) || doctor.has(msg.sender), "Only admin or doctor can add patients");
        patient.add(patientId); // reverts on zero address / duplicate
        patientHash[patientId] = _patientHash;
        patientIds.push(patientId);
        emit PatientAdded(patientId);
    }

    function getAllDoctors() public view returns (address[] memory) {
        return doctorIds;
    }

    function getAllPatients() public view returns (address[] memory) {
        return patientIds;
    }

    function isDoctor(address id) public view returns (bool) {
        return doctor.has(id);
    }

    function isPatient(address id) public view returns (bool) {
        return patient.has(id);
    }

    /// The recording doctor is always msg.sender (cannot be spoofed).
    function addMedicalRecord(address _patientId, bytes32 _recordHash, string memory _ipfsHash)
        public
        returns (uint256 id)
    {
        require(doctor.has(msg.sender), "Only doctors can add medical records");
        require(patient.has(_patientId), "Patient must be registered");
        require(consentContractAddress != address(0), "Consent contract not configured");
        require(
            IConsent(consentContractAddress).hasConsent(_patientId, msg.sender, "medical_record"),
            "Patient has not granted consent for medical record access"
        );

        id = ++recordCounter;
        medicalRecords[id] = MedicalRecord(id, _patientId, msg.sender, _recordHash, _ipfsHash, block.timestamp, true);
        patientRecords[_patientId].push(id);
        emit RecordAdded(id, _patientId, msg.sender);
    }

    function getPatientRecords(address _patientId) public view returns (uint256[] memory) {
        return patientRecords[_patientId];
    }

    function updateMedicalRecord(uint256 _recordId, bytes32 _newRecordHash, string memory _newIpfsHash) public {
        MedicalRecord storage record = medicalRecords[_recordId];
        require(record.isValid, "Record does not exist or is invalid");
        require(doctor.has(msg.sender), "Only doctors can update medical records");
        require(record.doctorId == msg.sender, "Only the original doctor can update the record");
        require(
            IConsent(consentContractAddress).hasConsent(record.patientId, msg.sender, "medical_record"),
            "Consent no longer active"
        );
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
        emit RecordInvalidated(_recordId);
    }
}
