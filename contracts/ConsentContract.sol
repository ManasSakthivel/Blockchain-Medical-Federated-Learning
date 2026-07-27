// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ConsentContract
 * @notice Patient-centric consent management using Decentralised Identifiers (DIDs).
 *
 * Each patient has a DID (did:ethr:<address>) stored off-chain; their Ethereum
 * address is the on-chain identity anchor. Patients grant and revoke data-sharing
 * consent per (recipient, dataType) pair. All consent actions are immutably logged.
 *
 * GDPR Compliance:
 *  - Article 7: consent can be withdrawn at any time via revokeConsent()
 *  - Article 17: data erasure requests are emitted as DataErasureRequested events
 *  - Article 20: data portability request events are emitted
 */
contract ConsentContract {

    // ── Structs ────────────────────────────────────────────────────────────
    struct ConsentRecord {
        address patient;
        address recipient;    // doctor / lab / researcher
        string  dataType;     // e.g. "lab_report", "prescription", "all"
        uint256 grantedAt;
        uint256 revokedAt;    // 0 = still active
        bool    isActive;
        string  purposeHash;  // IPFS CID of purpose statement (GDPR Art. 13)
    }

    // ── State ──────────────────────────────────────────────────────────────
    uint256 public consentCounter;
    mapping(uint256 => ConsentRecord)          public consents;
    mapping(address => uint256[])              public patientConsents;   // patient → consent IDs
    mapping(address => mapping(address => mapping(string => uint256))) public activeConsentId;
    // patient → recipient → dataType → consentId (0 = none)

    // ── Events ─────────────────────────────────────────────────────────────
    event ConsentGranted(
        uint256 indexed consentId,
        address indexed patient,
        address indexed recipient,
        string  dataType,
        uint256 timestamp,
        string  purposeHash
    );
    event ConsentRevoked(
        uint256 indexed consentId,
        address indexed patient,
        address indexed recipient,
        string  dataType,
        uint256 timestamp
    );
    event DataErasureRequested(
        address indexed patient,
        uint256 timestamp,
        string  ipfsCidToErase
    );
    event DataPortabilityRequested(
        address indexed patient,
        uint256 timestamp
    );

    // ── Consent management ─────────────────────────────────────────────────

    /**
     * @notice Grant consent for a recipient to access a specific data type.
     * @param recipient   Address of the doctor/lab/researcher
     * @param dataType    String identifier of the data category
     * @param purposeHash IPFS CID of the purpose statement document
     */
    function grantConsent(
        address recipient,
        string calldata dataType,
        string calldata purposeHash
    ) external {
        require(recipient != address(0), "Consent: zero recipient");
        require(bytes(dataType).length > 0, "Consent: empty dataType");

        // Revoke any existing consent for the same (patient, recipient, dataType)
        uint256 existingId = activeConsentId[msg.sender][recipient][dataType];
        if (existingId != 0 && consents[existingId].isActive) {
            _revoke(existingId);
        }

        consentCounter++;
        consents[consentCounter] = ConsentRecord({
            patient:     msg.sender,
            recipient:   recipient,
            dataType:    dataType,
            grantedAt:   block.timestamp,
            revokedAt:   0,
            isActive:    true,
            purposeHash: purposeHash
        });
        patientConsents[msg.sender].push(consentCounter);
        activeConsentId[msg.sender][recipient][dataType] = consentCounter;

        emit ConsentGranted(consentCounter, msg.sender, recipient, dataType, block.timestamp, purposeHash);
    }

    /**
     * @notice Revoke a previously granted consent.
     * Only the patient who granted it may revoke it.
     */
    function revokeConsent(uint256 consentId) external {
        ConsentRecord storage c = consents[consentId];
        require(c.patient == msg.sender, "Consent: not the patient");
        require(c.isActive, "Consent: already revoked");
        _revoke(consentId);
    }

    /**
     * @notice Revoke all active consents for a patient (e.g. before account erasure).
     * Only callable by the patient themselves.
     */
    function revokeAllConsents() external {
        uint256[] storage ids = patientConsents[msg.sender];
        for (uint256 i = 0; i < ids.length; i++) {
            if (consents[ids[i]].isActive) {
                _revoke(ids[i]);
            }
        }
    }

    // ── GDPR data subject rights ───────────────────────────────────────────

    /**
     * @notice Emit a data erasure request event (GDPR Art. 17 — Right to Erasure).
     * The off-chain system listens for this event and deletes the referenced data.
     * @param ipfsCidToErase CID of the IPFS data artifact to delete
     */
    function requestDataErasure(string calldata ipfsCidToErase) external {
        emit DataErasureRequested(msg.sender, block.timestamp, ipfsCidToErase);
    }

    /**
     * @notice Emit a data portability request event (GDPR Art. 20).
     */
    function requestDataPortability() external {
        emit DataPortabilityRequested(msg.sender, block.timestamp);
    }

    // ── View functions ─────────────────────────────────────────────────────

    /**
     * @notice Check whether a recipient currently has active consent
     * to access a specific data type for a given patient.
     */
    function hasConsent(
        address patient,
        address recipient,
        string calldata dataType
    ) external view returns (bool) {
        uint256 id = activeConsentId[patient][recipient][dataType];
        return id != 0 && consents[id].isActive;
    }

    function getPatientConsentIds(address patient) external view returns (uint256[] memory) {
        return patientConsents[patient];
    }

    function getConsent(uint256 consentId) external view returns (
        address patient,
        address recipient,
        string memory dataType,
        uint256 grantedAt,
        uint256 revokedAt,
        bool    isActive,
        string memory purposeHash
    ) {
        ConsentRecord storage c = consents[consentId];
        return (c.patient, c.recipient, c.dataType, c.grantedAt, c.revokedAt, c.isActive, c.purposeHash);
    }

    // ── Internal ───────────────────────────────────────────────────────────

    function _revoke(uint256 consentId) internal {
        ConsentRecord storage c = consents[consentId];
        c.isActive  = false;
        c.revokedAt = block.timestamp;
        activeConsentId[c.patient][c.recipient][c.dataType] = 0;
        emit ConsentRevoked(consentId, c.patient, c.recipient, c.dataType, block.timestamp);
    }
}
