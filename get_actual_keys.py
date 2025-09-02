#!/usr/bin/env python3
"""
Get Actual Private Keys from Ganache
This script finds the correct private key for your specific Ganache account.
"""

from web3 import Web3
from eth_account import Account
import json

def find_correct_private_key():
    """Find the correct private key for the target account"""
    print("🔍 Finding Correct Private Key")
    print("=" * 40)
    
    try:
        # Connect to Ganache
        web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
        
        if not web3.is_connected():
            print("❌ Failed to connect to Ganache")
            return None
        
        print("✅ Connected to Ganache")
        
        # Get accounts from Ganache
        accounts = web3.eth.accounts
        print(f"✅ Found {len(accounts)} accounts")
        
        target_account = "0x77AD9fCE8CeA9A19541AF7d889448e0eeC017efD"
        
        # Check if target account is in the list
        if target_account not in accounts:
            print(f"❌ Target account {target_account} not found in Ganache accounts")
            print("Available accounts:")
            for i, account in enumerate(accounts):
                print(f"   {i}: {account}")
            return None
        
        print(f"✅ Target account {target_account} found in Ganache")
        
        # The standard Ganache private keys
        standard_keys = [
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
            "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
            "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",
            "0x7c852118e8d7e3b95b4b8d0d8a3c2b5b8d0d8a3c2b5b8d0d8a3c2b5b8d0d8a3c",
            "0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a",
            "0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba",
            "0x92db14e403b83dfe3df233f83dfa2a13d7c63471eb57b95839df4b239f1624d9",
            "0x4bbbf85ce3377467afe5d46f804f221813b2bb87f24d81f60f1fcdbf7cbf4356",
            "0xdbda1821b80551c9d65939329250298aa3472ba22feea921c0cf5d620ea67b97",
            "0x2a871d0798f97d79848a013d4936a73bf4cc922c825d33c1cf7073dff6ad409a"
        ]
        
        # Find which private key corresponds to the target account
        correct_private_key = None
        correct_index = None
        
        for i, private_key in enumerate(standard_keys):
            account = Account.from_key(private_key)
            if account.address.lower() == target_account.lower():
                correct_private_key = private_key
                correct_index = i
                break
        
        if correct_private_key:
            print(f"✅ Found correct private key!")
            print(f"   Index: {correct_index}")
            print(f"   Private key: {correct_private_key}")
            print(f"   Account: {target_account}")
            
            # Create new private keys mapping
            private_keys = {}
            for i, account in enumerate(accounts):
                if i < len(standard_keys):
                    private_keys[account] = standard_keys[i]
                    balance = web3.eth.get_balance(account)
                    balance_eth = web3.from_wei(balance, 'ether')
                    print(f"   Account {i}: {account} ({balance_eth} ETH)")
            
            # Save to file
            with open('private_keys.json', 'w') as f:
                json.dump(private_keys, f, indent=2)
            
            print("✅ Updated private_keys.json with correct mapping")
            return private_keys
        else:
            print("❌ Could not find correct private key")
            print("This suggests your Ganache instance is using non-standard private keys")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return None

def main():
    """Main function"""
    private_keys = find_correct_private_key()
    
    if private_keys:
        print("\n🎉 Successfully found correct private keys!")
        print("\n📋 Next steps:")
        print("   1. Run: python test_file_upload.py")
        print("   2. Test file upload functionality")
        print("   3. Use the web interface to upload files")
    else:
        print("\n❌ Failed to find correct private keys")
        print("\n🔧 Alternative solutions:")
        print("   1. Restart Ganache with default settings")
        print("   2. Check Ganache configuration")
        print("   3. Use a different account from the available ones")

if __name__ == "__main__":
    main()
