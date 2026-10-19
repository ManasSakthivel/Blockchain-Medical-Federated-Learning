// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./Roles.sol";

/**
 * @title FederatedLearning
 * @notice On-chain governance contract for the Blockchain-FL framework.
 *
 * Implements:
 *  - Participant registration by an admin (no ETH staking: staked funds could
 *    never be withdrawn, so staking was removed)
 *  - Per-round model update recording (bytes32 hash of DP-noised weights)
 *  - Reputation-based consensus: reputation updated per round from accuracy
 *  - Tamper detection: flagged updates recorded immutably
 *  - Aggregator role: only the designated aggregator may record global hashes
 *
 * Reputation formula (on-chain exponential moving average):
 *   rep_new = (7 * accuracy_bps + 3 * rep_old) / 10
 * where accuracy_bps is accuracy * 10000 (basis points, 0-10000).
 *
 * Flagged nodes (decided off-chain by the aggregator) are deactivated and marked
 * slashed; no funds are involved. Everything is recorded by trusted roles (admin,
 * aggregator, registered participants): the contract is an audit log, not a
 * trustless consensus mechanism.
 */
contract FederatedLearning {
    using Roles for Roles.Role;

    // ── Roles ──────────────────────────────────────────────────────────────
    Roles.Role private adminRole;
    Roles.Role private participantRole;

    // ── Constants ──────────────────────────────────────────────────────────
    uint256 public constant MAX_FLAGGED_PER_ROUND = 64;
    uint256 public constant MAX_BPS               = 10000;

    // ── Structs ────────────────────────────────────────────────────────────
    struct ModelUpdate {
        address contributor;
        bytes32 modelHash;
        uint256 round;
        uint256 timestamp;
        bool    isTampered;     // flagged by aggregator
        uint256 accuracyBps;    // local accuracy in basis points (0-10000)
    }

    struct Participant {
        address addr;
        uint256 stake;          // always 0 (staking removed); kept for ABI stability
        uint256 reputation;     // 0-10000 basis points
        uint256 roundsContributed;
        bool    isActive;
        bool    isSlashed;
    }

    struct RoundSummary {
        uint256 round;
        bytes32 globalHash;     // hash of the SMPC-aggregated global model
        uint256 timestamp;
        uint256 nParticipants;
        uint256 avgAccuracyBps;
        address[] flaggedNodes;
    }

    // ── State ──────────────────────────────────────────────────────────────
    ModelUpdate[]  public updates;
    RoundSummary[] public roundSummaries;

    mapping(address => Participant) public participants;
    address[] public participantList;

    address public aggregator;   // address authorised to record global hashes
    uint256 public currentRound;

    // ── Events ─────────────────────────────────────────────────────────────
    event ParticipantRegistered(address indexed participant, uint256 stake);
    event ModelUpdated(
        address indexed contributor,
        bytes32 modelHash,
        uint256 round,
        uint256 timestamp,
        uint256 accuracyBps
    );
    event RoundCompleted(
        uint256 indexed round,
        bytes32 globalHash,
        uint256 avgAccuracyBps
    );
    event NodeFlagged(address indexed node, uint256 round, string reason);
    event ParticipantSlashed(address indexed participant, uint256 amount); // amount is always 0
    event ReputationUpdated(address indexed participant, uint256 oldRep, uint256 newRep);

    // ── Modifiers ──────────────────────────────────────────────────────────
    modifier onlyAdmin() {
        require(adminRole.has(msg.sender), "FL: caller is not admin");
        _;
    }

    modifier onlyAggregator() {
        require(msg.sender == aggregator, "FL: caller is not aggregator");
        _;
    }

    modifier onlyParticipant() {
        require(participantRole.has(msg.sender), "FL: caller is not a registered participant");
        require(participants[msg.sender].isActive, "FL: participant is not active");
        _;
    }

    // ── Constructor ────────────────────────────────────────────────────────
    constructor() {
        adminRole.add(msg.sender);
        aggregator   = msg.sender;
        currentRound = 0;
    }

    // ── Admin functions ────────────────────────────────────────────────────

    function addAdmin(address newAdmin) external onlyAdmin {
        adminRole.add(newAdmin);
    }

    function setAggregator(address newAggregator) external onlyAdmin {
        aggregator = newAggregator;
    }

    /**
     * @notice Register a participant without requiring staking
     * (for testnet / simulation environments).
     */
    function registerParticipant(address participant) external onlyAdmin {
        require(!participantRole.has(participant), "FL: already registered");
        participantRole.add(participant);
        participants[participant] = Participant({
            addr:               participant,
            stake:              0,
            reputation:         8000,   // start at 80% reputation
            roundsContributed:  0,
            isActive:           true,
            isSlashed:          false
        });
        participantList.push(participant);
        emit ParticipantRegistered(participant, 0);
    }

    // ── Model update recording ─────────────────────────────────────────────

    /**
     * @notice Record a local model update hash (called by each hospital node).
     * @param modelHash  SHA-256 of the DP-noised weight vector
     * @param round      FL round number
     * @param accuracyBps  Local model accuracy * 10000 (e.g. 0.82 → 8200)
     */
    function recordModelUpdate(
        bytes32 modelHash,
        uint256 round,
        uint256 accuracyBps
    ) external {
        recordModelUpdate(modelHash, round, accuracyBps, address(0));
    }

    /**
     * @notice Record a local model update hash with optional on-behalf-of attribution.
     * @param modelHash    SHA-256 of the DP-noised weight vector
     * @param round        FL round number
     * @param accuracyBps  Local model accuracy * 10000 (e.g. 0.82 → 8200)
     * @param onBehalfOf   Participant address if aggregator is recording on their behalf,
     *                     or address(0) for self-attribution.
     */
    function recordModelUpdate(
        bytes32 modelHash,
        uint256 round,
        uint256 accuracyBps,
        address onBehalfOf
    ) public {
        require(accuracyBps <= MAX_BPS, "FL: accuracy out of range");
        require(round <= currentRound + 1, "FL: round too far ahead");
        // Only an active registered participant, or the aggregator recording on
        // behalf of nodes (simulation), may write. Anyone else is rejected.
        bool isParticipantCaller = participantRole.has(msg.sender) && participants[msg.sender].isActive;
        require(isParticipantCaller || msg.sender == aggregator, "FL: caller not authorised");

        address contributor;
        if (msg.sender == aggregator) {
            if (onBehalfOf != address(0)) {
                require(participantRole.has(onBehalfOf), "FL: participant not registered");
                contributor = onBehalfOf;
            } else {
                contributor = aggregator;
            }
        } else {
            require(onBehalfOf == address(0) || onBehalfOf == msg.sender, "FL: caller cannot act on behalf of others");
            contributor = msg.sender;
        }

        updates.push(ModelUpdate({
            contributor:  contributor,
            modelHash:    modelHash,
            round:        round,
            timestamp:    block.timestamp,
            isTampered:   false,
            accuracyBps:  accuracyBps
        }));

        // Update reputation for registered participants
        if (participantRole.has(contributor)) {
            _updateReputation(contributor, accuracyBps);
        }

        emit ModelUpdated(contributor, modelHash, round, block.timestamp, accuracyBps);
    }

    /**
     * @notice Record the aggregated global model hash at the end of a round.
     * Only callable by the designated aggregator.
     * @param globalHash     SHA-256 of the SMPC-aggregated global weight vector
     * @param round          FL round number
     * @param avgAccuracyBps Average accuracy across all participants * 10000
     * @param flaggedNodes   Addresses of nodes whose reputation fell below threshold
     */
    function recordRoundSummary(
        bytes32 globalHash,
        uint256 round,
        uint256 avgAccuracyBps,
        address[] calldata flaggedNodes
    ) external onlyAggregator {
        require(avgAccuracyBps <= MAX_BPS, "FL: accuracy out of range");
        require(flaggedNodes.length <= MAX_FLAGGED_PER_ROUND, "FL: too many flagged nodes");
        require(round >= currentRound, "FL: round cannot go backwards");
        for (uint256 i = 0; i < flaggedNodes.length; i++) {
            require(participantRole.has(flaggedNodes[i]), "FL: flagged node not registered");
        }
        roundSummaries.push(RoundSummary({
            round:         round,
            globalHash:    globalHash,
            timestamp:     block.timestamp,
            nParticipants: participantList.length,
            avgAccuracyBps: avgAccuracyBps,
            flaggedNodes:  flaggedNodes
        }));

        // Deactivate flagged nodes
        for (uint256 i = 0; i < flaggedNodes.length; i++) {
            _slash(flaggedNodes[i], "reputation_below_threshold");
        }

        currentRound = round;
        emit RoundCompleted(round, globalHash, avgAccuracyBps);
    }

    /**
     * @notice Flag a specific update as tampered (callable by aggregator only).
     */
    function flagUpdate(uint256 updateIndex, string calldata reason) external onlyAggregator {
        require(updateIndex < updates.length, "FL: index out of bounds");
        updates[updateIndex].isTampered = true;
        emit NodeFlagged(updates[updateIndex].contributor, updates[updateIndex].round, reason);
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    function _updateReputation(address participant, uint256 accuracyBps) internal {
        Participant storage p = participants[participant];
        uint256 oldRep = p.reputation;
        // EMA: new_rep = 0.7 * accuracy + 0.3 * old_rep
        p.reputation = (7 * accuracyBps + 3 * oldRep) / 10;
        p.roundsContributed += 1;
        emit ReputationUpdated(participant, oldRep, p.reputation);
    }

    function _slash(address participant, string memory reason) internal {
        Participant storage p = participants[participant];
        if (!p.isActive || p.isSlashed) return;
        emit NodeFlagged(participant, currentRound, reason);
        p.isActive = false;
        p.isSlashed = true;
        emit ParticipantSlashed(participant, 0);
    }

    // ── View functions ─────────────────────────────────────────────────────

    function getUpdateCount() external view returns (uint256) {
        return updates.length;
    }

    function getUpdate(uint256 index) external view returns (
        address contributor,
        bytes32 modelHash,
        uint256 round,
        uint256 timestamp,
        bool    isTampered,
        uint256 accuracyBps
    ) {
        require(index < updates.length, "FL: index out of bounds");
        ModelUpdate storage u = updates[index];
        return (u.contributor, u.modelHash, u.round, u.timestamp, u.isTampered, u.accuracyBps);
    }

    function getRoundSummaryCount() external view returns (uint256) {
        return roundSummaries.length;
    }

    function getRoundSummary(uint256 index) external view returns (
        uint256 round,
        bytes32 globalHash,
        uint256 timestamp,
        uint256 nParticipants,
        uint256 avgAccuracyBps
    ) {
        require(index < roundSummaries.length, "FL: index out of bounds");
        RoundSummary storage rs = roundSummaries[index];
        return (rs.round, rs.globalHash, rs.timestamp, rs.nParticipants, rs.avgAccuracyBps);
    }

    function getReputation(address participant) external view returns (uint256) {
        return participants[participant].reputation;
    }

    function getParticipantCount() external view returns (uint256) {
        return participantList.length;
    }

    function isParticipant(address addr) external view returns (bool) {
        return participantRole.has(addr);
    }
}
