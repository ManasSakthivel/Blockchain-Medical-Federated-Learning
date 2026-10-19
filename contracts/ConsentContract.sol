// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ConsentContract
 * @notice Append-only log of consent grants/revocations, written by ONE trusted
 *         operator (the application server).
 *
 * Trust model (read this before relying on it):
 *  - Only `operator` can write. A transaction therefore proves "the operator
 *    recorded this", NOT "the patient signed this". Patients hold no keys here.
 *  - The application database is authoritative for access decisions; this
 *    contract is a tamper-evident audit trail of what the operator recorded.
 *  - `patient` is a pseudonymous address derived from a throw-away key (the
 *    DID); no name, e-mail or health data is stored. Everything on a public
 *    chain is readable by anyone and cannot be deleted.
 *  - This contract does not implement or evidence GDPR/HIPAA compliance.
 */
contract ConsentContract {
    struct ConsentRecord {
        address patient;
        address recipient;
        string dataType;
        bytes32 purposeHash; // SHA-256 of the purpose text (text itself is off-chain)
        uint256 grantedAt;
        uint256 revokedAt; // 0 = still active
        bool isActive;
    }

    address public operator;
    uint256 public consentCounter;
    mapping(uint256 => ConsentRecord) public consents;
    mapping(address => uint256[]) private patientConsents;
    // patient => recipient => keccak256(dataType) => consentId (0 = none)
    mapping(address => mapping(address => mapping(bytes32 => uint256))) private activeConsentId;

    event ConsentGranted(
        uint256 indexed consentId,
        address indexed patient,
        address indexed recipient,
        string dataType,
        bytes32 purposeHash
    );
    event ConsentRevoked(uint256 indexed consentId, address indexed patient, address indexed recipient);
    event OperatorTransferred(address indexed previousOperator, address indexed newOperator);

    modifier onlyOperator() {
        require(msg.sender == operator, "Consent: caller is not operator");
        _;
    }

    constructor() {
        operator = msg.sender;
        emit OperatorTransferred(address(0), msg.sender);
    }

    function transferOperator(address newOperator) external onlyOperator {
        require(newOperator != address(0), "Consent: zero operator");
        emit OperatorTransferred(operator, newOperator);
        operator = newOperator;
    }

    function grantConsent(
        address patient,
        address recipient,
        string calldata dataType,
        bytes32 purposeHash
    ) external onlyOperator returns (uint256 consentId) {
        require(patient != address(0) && recipient != address(0), "Consent: zero address");
        require(bytes(dataType).length > 0 && bytes(dataType).length <= 64, "Consent: bad dataType");

        bytes32 typeKey = keccak256(bytes(dataType));
        uint256 existing = activeConsentId[patient][recipient][typeKey];
        if (existing != 0) {
            _revoke(existing);
        }

        consentId = ++consentCounter;
        consents[consentId] = ConsentRecord({
            patient: patient,
            recipient: recipient,
            dataType: dataType,
            purposeHash: purposeHash,
            grantedAt: block.timestamp,
            revokedAt: 0,
            isActive: true
        });
        patientConsents[patient].push(consentId);
        activeConsentId[patient][recipient][typeKey] = consentId;
        emit ConsentGranted(consentId, patient, recipient, dataType, purposeHash);
    }

    function revokeConsent(uint256 consentId) external onlyOperator {
        require(consents[consentId].isActive, "Consent: not active");
        _revoke(consentId);
    }

    function hasConsent(address patient, address recipient, string calldata dataType)
        external
        view
        returns (bool)
    {
        uint256 id = activeConsentId[patient][recipient][keccak256(bytes(dataType))];
        return id != 0 && consents[id].isActive;
    }

    function getPatientConsentIds(address patient) external view returns (uint256[] memory) {
        return patientConsents[patient];
    }

    function _revoke(uint256 consentId) internal {
        ConsentRecord storage c = consents[consentId];
        c.isActive = false;
        c.revokedAt = block.timestamp;
        activeConsentId[c.patient][c.recipient][keccak256(bytes(c.dataType))] = 0;
        emit ConsentRevoked(consentId, c.patient, c.recipient);
    }
}
