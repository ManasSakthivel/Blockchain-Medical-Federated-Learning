// Deploys the five application contracts. The deploying account becomes the
// operator/owner of each; the Flask app must use that same account
// (BLOCKCHAIN_OPERATOR_PRIVATE_KEY). Not executed in the audit environment
// (no Truffle/Ganache); the contracts themselves are tested with eth-tester.
const ConsentContract = artifacts.require("ConsentContract");
const FileVerificationContract = artifacts.require("FileVerificationContract");
const ErasureLogContract = artifacts.require("ErasureLogContract");
const EHRContract = artifacts.require("EHRContract");
const FederatedLearning = artifacts.require("FederatedLearning");

module.exports = async function (deployer) {
  await deployer.deploy(ConsentContract);
  await deployer.deploy(FileVerificationContract);
  await deployer.deploy(ErasureLogContract);
  await deployer.deploy(FederatedLearning);
  await deployer.deploy(EHRContract);
  const ehr = await EHRContract.deployed();
  const consent = await ConsentContract.deployed();
  await ehr.setConsentContract(consent.address);
};
