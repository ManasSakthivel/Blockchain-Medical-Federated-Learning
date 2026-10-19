// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ErasureLogContract
 * @notice Append-only accountability log: "an erasure workflow ran".
 *
 * Each entry holds only a random 32-byte reference chosen by the application.
 * The reference is not derived from, and cannot be linked to, any person,
 * address, record or file. The matching off-chain row carries the details.
 *
 * This contract does NOT erase anything and cannot evidence that data was
 * deleted; blockchain data is immutable. Erasure happens off-chain (key
 * destruction + row deletion). Written by one trusted operator only.
 */
contract ErasureLogContract {
    address public operator;
    uint256 public entryCount;
    mapping(bytes32 => uint256) public recordedAt; // erasureRef => block timestamp

    event ErasureRecorded(bytes32 indexed erasureRef, uint256 timestamp);
    event OperatorTransferred(address indexed previousOperator, address indexed newOperator);

    modifier onlyOperator() {
        require(msg.sender == operator, "ErasureLog: caller is not operator");
        _;
    }

    constructor() {
        operator = msg.sender;
        emit OperatorTransferred(address(0), msg.sender);
    }

    function transferOperator(address newOperator) external onlyOperator {
        require(newOperator != address(0), "ErasureLog: zero operator");
        emit OperatorTransferred(operator, newOperator);
        operator = newOperator;
    }

    function recordErasure(bytes32 erasureRef) external onlyOperator {
        require(erasureRef != bytes32(0), "ErasureLog: empty ref");
        require(recordedAt[erasureRef] == 0, "ErasureLog: already recorded");
        recordedAt[erasureRef] = block.timestamp;
        entryCount += 1;
        emit ErasureRecorded(erasureRef, block.timestamp);
    }
}
