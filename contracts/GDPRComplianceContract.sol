// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title GDPRComplianceContract
 * @notice Immutable audit trail for GDPR compliance actions.
 *
 * Records:
 *  - Data erasure completions (functional erasure model)
 *  - Consent audit events
 *  - Data breach notifications (GDPR Art. 33)
 *  - Data Processing Agreements (DPA) hashes
 *
 * Functional Erasure Model (GDPR Art. 17 vs. Blockchain Immutability):
 *   Off-chain data + decryption keys are deleted. The on-chain pointer
 *   (IPFS CID or record hash) is rendered useless — "functional erasure".
 *   This contract records that the erasure was completed, satisfying
 *   the audit requirement while respecting blockchain immutability.
 */
contract GDPRComplianceContract {

    // ── Enums ──────────────────────────────────────────────────────────────
    enum ErasureStatus { Requested, InProgress, Completed, Failed }

    // ── Structs ────────────────────────────────────────────────────────────
    struct ErasureRecord {
        address patient;
        string  offChainDataRef;   // IPFS CID or record identifier (now erased)
        uint256 requestedAt;
        uint256 completedAt;       // 0 = not yet completed
        ErasureStatus status;
        string  processorDID;      // DID of the data processor who executed erasure
    }

    struct BreachNotification {
        uint256 notifiedAt;
        string  breachDescription; // IPFS CID of breach report (encrypted)
        address reportedBy;
        uint256 affectedCount;
    }

    // ── State ──────────────────────────────────────────────────────────────
    address public dataController;  // GDPR Data Controller address

    uint256 public erasureCounter;
    uint256 public breachCounter;

    mapping(uint256 => ErasureRecord)       public erasureRecords;
    mapping(address => uint256[])           public patientErasures;
    mapping(uint256 => BreachNotification)  public breachNotifications;

    // DPA hashes: processor address → IPFS CID of the signed DPA
    mapping(address => string) public dpaHashes;

    // ── Events ─────────────────────────────────────────────────────────────
    event ErasureRequested(
        uint256 indexed erasureId,
        address indexed patient,
        string  offChainDataRef,
        uint256 timestamp
    );
    event ErasureCompleted(
        uint256 indexed erasureId,
        address indexed patient,
        uint256 completedAt,
        string  processorDID
    );
    event ErasureFailed(uint256 indexed erasureId, string reason);
    event BreachNotified(
        uint256 indexed breachId,
        uint256 notifiedAt,
        uint256 affectedCount
    );
    event DPARegistered(address indexed processor, string ipfsCID);

    // ── Modifiers ──────────────────────────────────────────────────────────
    modifier onlyController() {
        require(msg.sender == dataController, "GDPR: not the data controller");
        _;
    }

    // ── Constructor ────────────────────────────────────────────────────────
    constructor() {
        dataController = msg.sender;
    }

    // ── Erasure lifecycle ──────────────────────────────────────────────────

    /**
     * @notice Log the start of a GDPR Art. 17 erasure request.
     * @param offChainDataRef  IPFS CID or record ID being erased
     */
    function requestErasure(string calldata offChainDataRef) external {
        erasureCounter++;
        erasureRecords[erasureCounter] = ErasureRecord({
            patient:          msg.sender,
            offChainDataRef:  offChainDataRef,
            requestedAt:      block.timestamp,
            completedAt:      0,
            status:           ErasureStatus.Requested,
            processorDID:     ""
        });
        patientErasures[msg.sender].push(erasureCounter);
        emit ErasureRequested(erasureCounter, msg.sender, offChainDataRef, block.timestamp);
    }

    /**
     * @notice Record completion of a functional erasure.
     * Only callable by the data controller once off-chain deletion is confirmed.
     * @param erasureId    ID returned by requestErasure
     * @param processorDID DID of the processor who performed the erasure
     */
    function confirmErasure(uint256 erasureId, string calldata processorDID) external onlyController {
        ErasureRecord storage r = erasureRecords[erasureId];
        require(r.requestedAt != 0, "GDPR: erasure not found");
        require(r.status == ErasureStatus.Requested || r.status == ErasureStatus.InProgress,
                "GDPR: erasure already finalised");
        r.completedAt  = block.timestamp;
        r.status       = ErasureStatus.Completed;
        r.processorDID = processorDID;
        emit ErasureCompleted(erasureId, r.patient, block.timestamp, processorDID);
    }

    function markErasureFailed(uint256 erasureId, string calldata reason) external onlyController {
        erasureRecords[erasureId].status = ErasureStatus.Failed;
        emit ErasureFailed(erasureId, reason);
    }

    // ── Breach notification ────────────────────────────────────────────────

    /**
     * @notice GDPR Art. 33 — notify supervisory authority of a data breach.
     * Must be reported within 72 hours of becoming aware.
     * @param breachReportCID  IPFS CID of the encrypted breach report
     * @param affectedCount    Approximate number of affected data subjects
     */
    function notifyBreach(string calldata breachReportCID, uint256 affectedCount) external onlyController {
        breachCounter++;
        breachNotifications[breachCounter] = BreachNotification({
            notifiedAt:        block.timestamp,
            breachDescription: breachReportCID,
            reportedBy:        msg.sender,
            affectedCount:     affectedCount
        });
        emit BreachNotified(breachCounter, block.timestamp, affectedCount);
    }

    // ── DPA registration ───────────────────────────────────────────────────

    /**
     * @notice Register a Data Processing Agreement for a processor.
     * @param processor  Address of the data processor
     * @param ipfsCID    IPFS CID of the signed DPA document
     */
    function registerDPA(address processor, string calldata ipfsCID) external onlyController {
        dpaHashes[processor] = ipfsCID;
        emit DPARegistered(processor, ipfsCID);
    }

    // ── View functions ─────────────────────────────────────────────────────

    function getPatientErasureIds(address patient) external view returns (uint256[] memory) {
        return patientErasures[patient];
    }

    function getErasureRecord(uint256 erasureId) external view returns (
        address patient,
        string memory offChainDataRef,
        uint256 requestedAt,
        uint256 completedAt,
        uint8   status,
        string memory processorDID
    ) {
        ErasureRecord storage r = erasureRecords[erasureId];
        return (r.patient, r.offChainDataRef, r.requestedAt, r.completedAt, uint8(r.status), r.processorDID);
    }
}
