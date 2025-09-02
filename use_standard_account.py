#!/usr/bin/env python3
"""
Use Standard Ganache Account
This script sets up the project to use a standard Ganache account.
"""

from web3 import Web3
from eth_account import Account
import json

def setup_standard_account():
    """Set up to use a standard Ganache account"""
    print("🔧 Setting Up Standard Ganache Account")
    print("=" * 40)
    
    try:
        # The standard Ganache private keys and their corresponding addresses
        standard_accounts = [
            {
                "address": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
                "private_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
            },
            {
                "address": "0x70997970C51812dc3A010C7d01b50e0d17dc79C8",
                "private_key": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"
            },
            {
                "address": "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC",
                "private_key": "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
            }
        ]
        
        # Connect to Ganache
        web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
        
        if not web3.is_connected():
            print("❌ Failed to connect to Ganache")
            return None
        
        print("✅ Connected to Ganache")
        
        # Get accounts from Ganache
        ganache_accounts = web3.eth.accounts
        print(f"✅ Found {len(ganache_accounts)} accounts in Ganache")
        
        # Find which standard account is available in your Ganache
        available_standard_account = None
        
        for standard_account in standard_accounts:
            if standard_account["address"] in ganache_accounts:
                available_standard_account = standard_account
                break
        
        if not available_standard_account:
            print("❌ No standard Ganache accounts found")
            print("Available accounts in your Ganache:")
            for i, account in enumerate(ganache_accounts):
                balance = web3.eth.get_balance(account)
                balance_eth = web3.from_wei(balance, 'ether')
                print(f"   {i}: {account} ({balance_eth} ETH)")
            return None
        
        print(f"✅ Found standard account: {available_standard_account['address']}")
        print(f"   Private key: {available_standard_account['private_key']}")
        
        # Create private keys mapping
        private_keys = {}
        for i, account in enumerate(ganache_accounts):
            if i < len(standard_accounts):
                private_keys[account] = standard_accounts[i]["private_key"]
            else:
                # For additional accounts, we'll need to handle them differently
                print(f"   Account {i}: {account} (no standard private key)")
        
        # Save to file
        with open('private_keys.json', 'w') as f:
            json.dump(private_keys, f, indent=2)
        
        print("✅ Updated private_keys.json")
        
        # Update the blockchain service to use the standard account
        print(f"✅ Will use account: {available_standard_account['address']}")
        
        return available_standard_account
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def main():
    """Main function"""
    standard_account = setup_standard_account()
    
    if standard_account:
        print("\n🎉 Successfully set up standard account!")
        print(f"\n📋 Using account: {standard_account['address']}")
        print("\n📋 Next steps:")
        print("   1. Update your BlockchainService to use this account")
        print("   2. Run: python test_file_upload.py")
        print("   3. Test file upload functionality")
        
        # Update the blockchain service configuration
        print(f"\n🔧 To use this account, update your BlockchainService:")
        print(f"   Set account to: {standard_account['address']}")
    else:
        print("\n❌ Failed to set up standard account")
        print("\n🔧 Alternative solutions:")
        print("   1. Restart Ganache with default settings")
        print("   2. Use a different account from your Ganache instance")
        print("   3. Check Ganache configuration")

if __name__ == "__main__":
    main()
