require("dotenv").config();

module.exports = {
  networks: {
    development: {
      host: "127.0.0.1",
      port: 7545,
      network_id: "*", // Match any network id
    },
    test: {
      host: "127.0.0.1",
      port: 7545,
      network_id: "*",
    },
    // Ethereum Sepolia testnet — set SEPOLIA_RPC_URL and DEPLOYER_PRIVATE_KEY in .env
    sepolia: {
      provider: () => {
        const HDWalletProvider = require("@truffle/hdwallet-provider");
        return new HDWalletProvider(
          process.env.DEPLOYER_PRIVATE_KEY,
          process.env.SEPOLIA_RPC_URL || "https://rpc.sepolia.org"
        );
      },
      network_id: 11155111,
      gas: 4_500_000,
      gasPrice: 10_000_000_000, // 10 gwei
      confirmations: 2,
      timeoutBlocks: 200,
      skipDryRun: true,
    },
  },
  compilers: {
    solc: {
      version: "0.8.20",
      settings: {
        optimizer: {
          enabled: true,
          runs: 200,
        },
      },
    },
  },
};