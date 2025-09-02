# Ganache Setup Guide for EHR Blockchain Project

This guide will help you set up your EHR blockchain project to use actual ETH from Ganache instead of mock implementations.

## Prerequisites

1. **Ganache** - Download and install from [https://trufflesuite.com/ganache/](https://trufflesuite.com/ganache/)
2. **Node.js and npm** - For Truffle installation
3. **Truffle** - Install globally: `npm install -g truffle`
4. **Python dependencies** - Already installed in your project

## Step 1: Start Ganache

1. Open Ganache application
2. Click "Quickstart" to start a new workspace
3. Make sure it's running on the default port: **http://127.0.0.1:7545**
4. Note the accounts and their ETH balances (each account starts with 100 ETH)

## Step 2: Deploy Contracts

Run the automated setup script:

```bash
python setup_ganache.py
```

This script will:
- ✅ Check if Ganache is running
- ✅ Get all available accounts
- ✅ Create a private keys file for development
- ✅ Compile and deploy contracts using Truffle
- ✅ Extract contract addresses
- ✅ Update the BlockchainService
- ✅ Test the connection

## Step 3: Verify Setup

Test the connection and deployment:

```bash
python test_ganache_connection.py
```

This will verify:
- ✅ Ganache connection
- ✅ Contract deployment
- ✅ BlockchainService functionality

## Step 4: Start Your Application

```bash
python run.py
```

Navigate to `http://localhost:5000` and test the file verification features.

## How It Works

### 1. Blockchain Connection

The `BlockchainService` now connects to Ganache using `web3.py`:

```python
self.web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
```

### 2. Contract Interaction

All blockchain operations now use the deployed `FileVerificationContract`:

- **File Upload**: `uploadFile()` function stores file hashes on blockchain
- **File Verification**: `verifyFile()` function compares hashes and logs results
- **File Retrieval**: `getFile()` function retrieves file records
- **Audit Trail**: `getVerificationLog()` function gets verification history

### 3. Transaction Signing

Transactions are signed using private keys from the `private_keys.json` file:

```python
signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key)
```

### 4. Gas Management

The system automatically:
- Estimates gas requirements
- Uses current gas prices from Ganache
- Handles transaction nonces

## File Structure

After setup, you'll have these new files:

```
ZIDP/
├── private_keys.json          # Private keys for development (DO NOT commit)
├── contract_address.txt       # Deployed contract address
├── build/                     # Truffle build artifacts
│   └── contracts/
│       └── FileVerificationContract.json
├── setup_ganache.py          # Setup script
└── test_ganache_connection.py # Connection test script
```

## Security Considerations

### Development Environment

⚠️ **IMPORTANT**: The current setup is for development only!

- Private keys are stored in plain text (`private_keys.json`)
- Using default Ganache private keys
- No encryption or secure key management

### Production Environment

For production, implement:

1. **Secure Key Management**:
   ```python
   # Use environment variables or secure key stores
   private_key = os.environ.get('PRIVATE_KEY')
   ```

2. **Hardware Security Modules (HSM)**:
   ```python
   # Use HSM for key storage and signing
   from cryptography.hazmat.primitives import serialization
   ```

3. **Multi-signature Wallets**:
   ```solidity
   // Implement multi-sig for critical operations
   contract MultiSigWallet {
       mapping(address => bool) public isOwner;
       uint public required;
   }
   ```

## Testing the System

### 1. File Upload Test

1. Login as a doctor or lab user
2. Navigate to "File Verification" → "Upload File"
3. Select a patient and upload a file
4. Check the transaction in Ganache

### 2. File Verification Test

1. Navigate to "File Verification" → "Verify File"
2. Enter the File ID from the upload
3. Upload the same file for verification
4. Check the verification result

### 3. Tampering Demo

1. Navigate to "File Verification" → "Tampering Demo"
2. See how even a single byte change affects the hash
3. Understand the security benefits

## Troubleshooting

### Ganache Not Running

```
❌ Ganache is not running on http://127.0.0.1:7545
```

**Solution**: Start Ganache and ensure it's running on the correct port.

### Contract Not Deployed

```
❌ Contract address file not found
```

**Solution**: Run `python setup_ganache.py` to deploy contracts.

### Truffle Not Found

```
❌ Truffle not found
```

**Solution**: Install Truffle globally:
```bash
npm install -g truffle
```

### Private Key Issues

```
❌ Error getting private key
```

**Solution**: Ensure `private_keys.json` exists and contains valid keys.

### Gas Issues

```
❌ Out of gas
```

**Solution**: Increase gas limit in transaction calls or ensure account has sufficient ETH.

## Monitoring Transactions

### Ganache Interface

1. Open Ganache
2. Go to "Transactions" tab
3. View all blockchain transactions
4. Check gas usage and transaction status

### Contract Events

The `FileVerificationContract` emits events for:

- `FileUploaded`: When a file is uploaded
- `FileVerified`: When a file is verified
- `FileInvalidated`: When a file is invalidated

### Logs

Check the application logs for transaction details:

```python
print(f"Transaction hash: {tx_receipt.transactionHash.hex()}")
print(f"Gas used: {tx_receipt.gasUsed}")
print(f"Block number: {tx_receipt.blockNumber}")
```

## Performance Optimization

### Gas Optimization

1. **Batch Operations**: Group multiple operations in single transactions
2. **Gas Estimation**: Use `estimate_gas()` before transactions
3. **Gas Price**: Monitor and adjust gas prices based on network conditions

### Caching

1. **Contract Instance**: Reuse contract instances
2. **Account Nonce**: Track and manage nonces properly
3. **Connection Pooling**: Use connection pooling for multiple requests

## Next Steps

1. **Test thoroughly** with different file types and sizes
2. **Monitor gas usage** and optimize transactions
3. **Implement error handling** for network issues
4. **Add transaction confirmation** UI feedback
5. **Consider upgrading** to a testnet (Sepolia, Goerli) for more realistic testing

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify Ganache is running and accessible
3. Ensure all dependencies are installed
4. Check the application logs for detailed error messages
5. Test with the provided test scripts

---

🎉 **Congratulations!** Your EHR blockchain project is now using real ETH from Ganache for secure file verification!
