// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FileVerificationContract
 * @notice Append-only anchor of keyed file digests, written by ONE trusted operator.
 *
 * What is stored: an opaque 32-byte id and a 32-byte digest. The digest is an
 * HMAC-SHA256 computed with a secret key held by the application, so it cannot
 * be used to confirm a guessed file and reveals nothing about the content.
 * No file name, type, size, patient address or IPFS CID is written here.
 *
 * Trust model: only `operator` can write. The contract proves that the operator
 * anchored a digest at a given block time; it does NOT compare file contents
 * (the HMAC key is off-chain) and `recordVerification` only logs the operator's
 * off-chain comparison result.
 */
contract FileVerificationContract {
    struct Anchor {
        bytes32 publicId;
        bytes32 digest;
        uint256 anchoredAt;
        uint256 verifications;
        uint256 mismatches;
    }

    address public operator;
    uint256 public fileCounter;
    mapping(uint256 => Anchor) private anchors;
    mapping(bytes32 => uint256) public idByPublicId; // 0 = unknown

    event FileAnchored(uint256 indexed fileId, bytes32 indexed publicId, bytes32 digest);
    event VerificationRecorded(uint256 indexed fileId, bool isMatch);
    event OperatorTransferred(address indexed previousOperator, address indexed newOperator);

    modifier onlyOperator() {
        require(msg.sender == operator, "FileVerification: caller is not operator");
        _;
    }

    constructor() {
        operator = msg.sender;
        emit OperatorTransferred(address(0), msg.sender);
    }

    function transferOperator(address newOperator) external onlyOperator {
        require(newOperator != address(0), "FileVerification: zero operator");
        emit OperatorTransferred(operator, newOperator);
        operator = newOperator;
    }

    function anchorFile(bytes32 publicId, bytes32 digest) external onlyOperator returns (uint256 fileId) {
        require(publicId != bytes32(0) && digest != bytes32(0), "FileVerification: empty value");
        require(idByPublicId[publicId] == 0, "FileVerification: already anchored");
        fileId = ++fileCounter;
        anchors[fileId] = Anchor(publicId, digest, block.timestamp, 0, 0);
        idByPublicId[publicId] = fileId;
        emit FileAnchored(fileId, publicId, digest);
    }

    function recordVerification(uint256 fileId, bool isMatch) external onlyOperator {
        Anchor storage a = anchors[fileId];
        require(a.anchoredAt != 0, "FileVerification: unknown file");
        a.verifications += 1;
        if (!isMatch) a.mismatches += 1;
        emit VerificationRecorded(fileId, isMatch);
    }

    function getAnchor(uint256 fileId)
        external
        view
        returns (bytes32 publicId, bytes32 digest, uint256 anchoredAt, uint256 verifications, uint256 mismatches)
    {
        Anchor storage a = anchors[fileId];
        require(a.anchoredAt != 0, "FileVerification: unknown file");
        return (a.publicId, a.digest, a.anchoredAt, a.verifications, a.mismatches);
    }
}
