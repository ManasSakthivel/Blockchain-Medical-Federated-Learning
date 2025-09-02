#!/usr/bin/env python3
"""
Test Ganache Connection
This script tests the connection to Ganache and the deployed contracts.
"""

from web3 import Web3
import json
import os

def test_ganache_connection():
    """Test basic Ganache connection"""
    print("🔍 Testing Ganache Connection...")
    
    try:
        # Connect to Ganache
        web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
        
        if web3.is_connected():
            print("✅ Successfully connected to Ganache")
            
            # Get accounts
            accounts = web3.eth.accounts
            print(f"✅ Found {len(accounts)} accounts")
            
            # Show account balances
            for i, account in enumerate(accounts[:3]):  # Show first 3 accounts
                balance = web3.eth.get_balance(account)
                balance_eth = web3.from_wei(balance, 'ether')
                print(f"   Account {i}: {account} ({balance_eth} ETH)")
            
            return True
        else:
            print("❌ Failed to connect to Ganache")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to Ganache: {e}")
        return False

def test_contract_deployment():
    """Test if contracts are deployed"""
    print("\n🔍 Testing Contract Deployment...")
    
    try:
        # Check if contract address file exists
        contract_address_path = 'contract_address.txt'
        if not os.path.exists(contract_address_path):
            print("❌ Contract address file not found")
            print("   Run: python setup_ganache.py")
            return False
        
        # Read contract address
        with open(contract_address_path, 'r') as f:
            contract_address = f.read().strip()
        
        print(f"✅ Contract address: {contract_address}")
        
        # Check if build artifacts exist
        build_path = os.path.join('build', 'contracts', 'FileVerificationContract.json')
        if not os.path.exists(build_path):
            print("❌ Contract build artifacts not found")
            print("   Run: truffle compile")
            return False
        
        # Load contract ABI
        with open(build_path, 'r') as f:
            contract_data = json.load(f)
            abi = contract_data.get('abi')
        
        if not abi:
            print("❌ Contract ABI not found")
            return False
        
        print("✅ Contract ABI loaded successfully")
        
        # Test contract interaction
        web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
        contract = web3.eth.contract(address=contract_address, abi=abi)
        
        # Test basic contract functions
        total_files = contract.functions.getTotalFiles().call()
        total_verifications = contract.functions.getTotalVerifications().call()
        
        print(f"✅ Total files on contract: {total_files}")
        print(f"✅ Total verifications on contract: {total_verifications}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing contract deployment: {e}")
        return False

def test_blockchain_service():
    """Test the BlockchainService"""
    print("\n🔍 Testing BlockchainService...")
    
    try:
        from app.services.blockchain_service import BlockchainService
        
        service = BlockchainService()
        
        # Test connection
        if not service.connect_to_ganache():
            print("❌ BlockchainService failed to connect to Ganache")
            return False
        
        # Test contract loading
        if not service.load_contract():
            print("❌ BlockchainService failed to load contract")
            return False
        
        # Test account balance
        balance = service.get_balance()
        print(f"✅ Account balance: {balance} ETH")
        
        # Test contract functions
        total_files = service.contract.functions.getTotalFiles().call()
        print(f"✅ Total files: {total_files}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing BlockchainService: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Ganache Connection Test")
    print("=" * 40)
    
    # Test 1: Basic Ganache connection
    if not test_ganache_connection():
        print("\n❌ Ganache connection test failed")
        print("   Please make sure Ganache is running on http://127.0.0.1:7545")
        return
    
    # Test 2: Contract deployment
    if not test_contract_deployment():
        print("\n❌ Contract deployment test failed")
        print("   Please run: python setup_ganache.py")
        return
    
    # Test 3: BlockchainService
    if not test_blockchain_service():
        print("\n❌ BlockchainService test failed")
        return
    
    print("\n🎉 All tests passed!")
    print("\n✅ Your project is ready to use with Ganache!")
    print("\n📋 Next steps:")
    print("   1. Start your Flask app: python run.py")
    print("   2. Navigate to: http://localhost:5000")
    print("   3. Test the file verification features")

if __name__ == "__main__":
    main()
