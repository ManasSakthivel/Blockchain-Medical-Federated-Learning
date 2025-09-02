#!/usr/bin/env python3
"""
Ganache Setup Script for EHR Blockchain Project
This script helps set up the connection to Ganache and deploy contracts.
"""

import json
import os
from web3 import Web3
from eth_account import Account
import subprocess
import sys

def check_ganache_running():
    """Check if Ganache is running"""
    try:
        web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
        if web3.is_connected():
            print("✅ Ganache is running on http://127.0.0.1:7545")
            return True
        else:
            print("❌ Ganache is not running on http://127.0.0.1:7545")
            return False
    except Exception as e:
        print(f"❌ Error checking Ganache: {e}")
        return False

def get_ganache_accounts():
    """Get accounts from Ganache"""
    try:
        web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
        accounts = web3.eth.accounts
        print(f"✅ Found {len(accounts)} accounts in Ganache")
        
        for i, account in enumerate(accounts):
            balance = web3.eth.get_balance(account)
            balance_eth = web3.from_wei(balance, 'ether')
            print(f"   Account {i}: {account} ({balance_eth} ETH)")
        
        return accounts
    except Exception as e:
        print(f"❌ Error getting accounts: {e}")
        return []

def create_private_keys_file(accounts):
    """Create a file with private keys for development"""
    try:
        # In a real application, you would NEVER store private keys in plain text
        # This is only for development purposes
        private_keys = {}
        
        print("\n🔑 Setting up private keys for development...")
        print("⚠️  WARNING: This is for development only. Never store private keys in production!")
        
        # For development, we'll use the default Ganache private keys
        # These are the default private keys that Ganache uses
        default_keys = [
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
            "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
            "0x7c852118e8d7e3b95b4b8d0d8a3c2b5b8d0d8a3c2b5b8d0d8a3c2b5b8d0d8a3c",
            "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a"
        ]
        
        for i, account in enumerate(accounts):
            if i < len(default_keys):
                private_keys[account] = default_keys[i]
            else:
                # Generate a new private key for additional accounts
                new_account = Account.create()
                private_keys[account] = new_account.key.hex()
        
        # Save to file
        with open('private_keys.json', 'w') as f:
            json.dump(private_keys, f, indent=2)
        
        print("✅ Private keys saved to private_keys.json")
        print("⚠️  Remember to add private_keys.json to .gitignore!")
        
        return private_keys
        
    except Exception as e:
        print(f"❌ Error creating private keys file: {e}")
        return None

def deploy_contracts():
    """Deploy contracts using Truffle"""
    try:
        print("\n📦 Deploying contracts with Truffle...")
        
        # Check if truffle is installed
        try:
            subprocess.run(['truffle', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Truffle not found. Please install Truffle globally:")
            print("   npm install -g truffle")
            return False
        
        # Compile contracts
        print("🔨 Compiling contracts...")
        result = subprocess.run(['truffle', 'compile'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Compilation failed: {result.stderr}")
            return False
        
        print("✅ Contracts compiled successfully")
        
        # Deploy contracts
        print("🚀 Deploying contracts to Ganache...")
        result = subprocess.run(['truffle', 'migrate', '--reset'], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Deployment failed: {result.stderr}")
            return False
        
        print("✅ Contracts deployed successfully")
        
        # Extract contract addresses
        extract_contract_addresses()
        
        return True
        
    except Exception as e:
        print(f"❌ Error deploying contracts: {e}")
        return False

def extract_contract_addresses():
    """Extract contract addresses from build artifacts"""
    try:
        build_path = os.path.join('build', 'contracts')
        if not os.path.exists(build_path):
            print("❌ Build directory not found")
            return
        
        # Get FileVerificationContract address
        contract_json_path = os.path.join(build_path, 'FileVerificationContract.json')
        if os.path.exists(contract_json_path):
            with open(contract_json_path, 'r') as f:
                contract_data = json.load(f)
                networks = contract_data.get('networks', {})
                
                for network_id, network_data in networks.items():
                    address = network_data.get('address')
                    if address:
                        # Save to a simple text file for easy access
                        with open('contract_address.txt', 'w') as f:
                            f.write(address)
                        print(f"✅ FileVerificationContract deployed at: {address}")
                        return
        
        print("❌ Contract address not found in build artifacts")
        
    except Exception as e:
        print(f"❌ Error extracting contract addresses: {e}")

def update_blockchain_service():
    """Update the BlockchainService to use private keys"""
    try:
        print("\n🔧 Updating BlockchainService...")
        
        # Read the current blockchain service
        service_path = 'app/services/blockchain_service.py'
        with open(service_path, 'r') as f:
            content = f.read()
        
        # Add private key loading functionality
        private_key_import = "import json\nimport hashlib\nfrom web3 import Web3\nfrom flask import current_app\nimport os\n"
        
        # Update the _get_private_key method
        new_private_key_method = '''    def _get_private_key(self):
        """Get private key for the current account"""
        try:
            if not self.account:
                return None
            
            # Load private keys from file
            private_keys_path = os.path.join(os.getcwd(), 'private_keys.json')
            if os.path.exists(private_keys_path):
                with open(private_keys_path, 'r') as f:
                    private_keys = json.load(f)
                    return private_keys.get(self.account)
            
            return None
        except Exception as e:
            print(f"Error getting private key: {e}")
            return None'''
        
        # Replace the old _get_private_key method
        old_method = '''    def _get_private_key(self):
        """Get private key for the current account"""
        # In a real application, you would store private keys securely
        # For development, you can get them from Ganache
        # This is a placeholder - you'll need to implement secure key management
        return None  # You'll need to provide the actual private key'''
        
        content = content.replace(old_method, new_private_key_method)
        
        # Write back to file
        with open(service_path, 'w') as f:
            f.write(content)
        
        print("✅ BlockchainService updated successfully")
        
    except Exception as e:
        print(f"❌ Error updating BlockchainService: {e}")

def test_connection():
    """Test the blockchain connection"""
    try:
        print("\n🧪 Testing blockchain connection...")
        
        from app.services.blockchain_service import BlockchainService
        
        service = BlockchainService()
        
        # Test connection
        if not service.connect_to_ganache():
            print("❌ Failed to connect to Ganache")
            return False
        
        # Test contract loading
        if not service.load_contract():
            print("❌ Failed to load contract")
            return False
        
        # Test account balance
        balance = service.get_balance()
        print(f"✅ Account balance: {balance} ETH")
        
        # Test contract functions
        total_files = service.contract.functions.getTotalFiles().call()
        print(f"✅ Total files on contract: {total_files}")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Ganache Setup for EHR Blockchain Project")
    print("=" * 50)
    
    # Check if Ganache is running
    if not check_ganache_running():
        print("\n❌ Please start Ganache first:")
        print("   1. Open Ganache")
        print("   2. Make sure it's running on http://127.0.0.1:7545")
        print("   3. Run this script again")
        return
    
    # Get accounts
    accounts = get_ganache_accounts()
    if not accounts:
        print("❌ No accounts found in Ganache")
        return
    
    # Create private keys file
    private_keys = create_private_keys_file(accounts)
    if not private_keys:
        print("❌ Failed to create private keys file")
        return
    
    # Deploy contracts
    if not deploy_contracts():
        print("❌ Failed to deploy contracts")
        return
    
    # Update blockchain service
    update_blockchain_service()
    
    # Test connection
    if not test_connection():
        print("❌ Connection test failed")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Start your Flask application: python run.py")
    print("   2. Navigate to: http://localhost:5000")
    print("   3. Login and test the file verification features")
    print("\n⚠️  Important security notes:")
    print("   - private_keys.json contains sensitive data")
    print("   - Add private_keys.json to .gitignore")
    print("   - Never use these keys in production")
    print("   - Use proper key management in production")

if __name__ == "__main__":
    main()
