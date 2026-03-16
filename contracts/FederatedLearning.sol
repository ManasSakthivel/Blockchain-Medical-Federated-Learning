// SPDX-License-Identifier: MIT
pragma solidity ^0.8.11;

contract FederatedLearning {
    struct ModelUpdate {
        address contributor;
        bytes32 modelHash;
        uint round;
        uint timestamp;
    }

    ModelUpdate[] public updates;

    event ModelUpdated(address indexed contributor, bytes32 modelHash, uint round, uint timestamp);

    function recordModelUpdate(bytes32 modelHash, uint round) public {
        updates.push(ModelUpdate(msg.sender, modelHash, round, block.timestamp));
        emit ModelUpdated(msg.sender, modelHash, round, block.timestamp);
    }

    function getUpdateCount() public view returns (uint) {
        return updates.length;
    }

    function getUpdate(uint index) public view returns (address, bytes32, uint, uint) {
        require(index < updates.length, "Index out of bounds");
        ModelUpdate memory update = updates[index];
        return (update.contributor, update.modelHash, update.round, update.timestamp);
    }
}
