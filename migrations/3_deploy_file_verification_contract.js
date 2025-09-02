const FileVerificationContract = artifacts.require("FileVerificationContract");

module.exports = function(deployer) {
  deployer.deploy(FileVerificationContract);
};
