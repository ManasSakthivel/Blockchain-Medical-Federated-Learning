const EHRContract = artifacts.require("EHRContract");

module.exports = function (deployer) {
  deployer.deploy(EHRContract);
}; 