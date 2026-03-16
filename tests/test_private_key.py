#!/usr/bin/env python3
"""
Test Private Key Loading
This script tests if the private key is being loaded correctly.
"""

from web3 import Web3
from eth_account import Account
import json
import os

def test_private_key():
    """Test private key loading and account derivation"""
    print("🔑 Testing Private Key Loading")
    print("=" * 40)
    
    try:
        # Load private keys
        with open('private_keys.json', 'r') as f:
            private_keys = json.load(f)
        
        target_account = "0x77AD9fCE8CeA9A19541AF7d889448e0eeC017efD"
        
        if target_account in private_keys:
            private_key = private_keys[target_account]
            print(f"✅ Found private key for {target_account}")
            print(f"   Private key: {private_key}")
            
            # Derive account from private key
            account = Account.from_key(private_key)
            derived_address = account.address
            print(f"   Derived address: {derived_address}")
            
            # Check if they match
            if derived_address.lower() == target_account.lower():
                print("✅ Private key matches account address!")
                
                # Test transaction signing
                web3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
                
                # Create a simple transaction
                transaction = {
                    'to': '0x0000000000000000000000000000000000000000',
                    'value': 0,
                    'gas': 21000,
                    'gasPrice': web3.eth.gas_price,
                    'nonce': web3.eth.get_transaction_count(target_account),
                    'chainId': web3.eth.chain_id
                }
                
                # Sign transaction
                signed_txn = web3.eth.account.sign_transaction(transaction, private_key)
                print("✅ Transaction signed successfully!")
                print(f"   Transaction hash: {signed_txn.hash.hex()}")
                
                return True
            else:
                print("❌ Private key does not match account address!")
                print(f"   Expected: {target_account}")
                print(f"   Got: {derived_address}")
                return False
        else:
            print(f"❌ Account {target_account} not found in private keys")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Main function"""
    success = test_private_key()
    
    if success:
        print("\n🎉 Private key test passed!")
        print("✅ Your private key configuration is correct!")
    else:
        print("\n❌ Private key test failed!")
        print("🔧 Please check your private key configuration")

if __name__ == "__main__":
    main()
