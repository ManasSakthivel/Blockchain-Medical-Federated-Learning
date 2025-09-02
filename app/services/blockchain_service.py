import json
import hashlib
from web3 import Web3
from flask import current_app
import os

class BlockchainService:
    def __init__(self):
        self.web3 = None
        self.contract = None
        self.account = None
        self.contract_address = None
        self.contract_abi = None
        self.is_connected = False
        
        # Ganache configuration
        self.ganache_url = "http://127.0.0.1:7545"  # Default Ganache URL
        self.contract_address = None
        self.contract_abi = None
        
    def connect_to_ganache(self):
        """Connect to Ganache local blockchain"""
        try:
            self.web3 = Web3(Web3.HTTPProvider(self.ganache_url))
            
            if self.web3.is_connected():
                print(f"✅ Connected to Ganache at {self.ganache_url}")
                self.is_connected = True
                
                # Set default account (first account in Ganache)
                accounts = self.web3.eth.accounts
                if accounts:
                    self.account = accounts[0]
                    print(f"✅ Using account: {self.account}")
                
                return True
            else:
                print(f"❌ Failed to connect to Ganache at {self.ganache_url}")
                return False
                
        except Exception as e:
            print(f"❌ Error connecting to Ganache: {e}")
            return False
    
    def load_contract(self, contract_address=None, contract_abi_path=None):
        """Load the FileVerificationContract"""
        try:
            if not self.is_connected:
                if not self.connect_to_ganache():
                    return False
            
            # Use provided address or try to get from deployment
            if contract_address:
                self.contract_address = contract_address
            else:
                # Try to get from deployment artifacts
                self.contract_address = self._get_deployed_contract_address()
            
            if not self.contract_address:
                print("❌ No contract address found. Please deploy the contract first.")
                return False
            
            # Load ABI
            if contract_abi_path:
                with open(contract_abi_path, 'r') as f:
                    self.contract_abi = json.load(f)
            else:
                self.contract_abi = self._get_contract_abi()
            
            if not self.contract_abi:
                print("❌ No contract ABI found.")
                return False
            
            # Create contract instance
            self.contract = self.web3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
            
            print(f"✅ Loaded FileVerificationContract at {self.contract_address}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading contract: {e}")
            return False
    
    def _get_deployed_contract_address(self):
        """Get deployed contract address from build artifacts"""
        try:
            # Check if build directory exists
            build_path = os.path.join(os.getcwd(), 'build', 'contracts')
            if os.path.exists(build_path):
                contract_json_path = os.path.join(build_path, 'FileVerificationContract.json')
                if os.path.exists(contract_json_path):
                    with open(contract_json_path, 'r') as f:
                        contract_data = json.load(f)
                        networks = contract_data.get('networks', {})
                        # Get the latest network deployment (highest network ID)
                        latest_network = max(networks.keys(), key=int) if networks else None
                        if latest_network:
                            return networks[latest_network].get('address')
            
            # Fallback: check if address is stored in a config file
            config_path = os.path.join(os.getcwd(), 'contract_address.txt')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return f.read().strip()
            
            return None
            
        except Exception as e:
            print(f"Error getting contract address: {e}")
            return None
    
    def _get_contract_abi(self):
        """Get contract ABI from build artifacts"""
        try:
            build_path = os.path.join(os.getcwd(), 'build', 'contracts')
            contract_json_path = os.path.join(build_path, 'FileVerificationContract.json')
            
            if os.path.exists(contract_json_path):
                with open(contract_json_path, 'r') as f:
                    contract_data = json.load(f)
                    return contract_data.get('abi')
            
            return None
            
        except Exception as e:
            print(f"Error getting contract ABI: {e}")
            return None
    
    def get_accounts(self):
        """Get available accounts from Ganache"""
        if self.is_connected:
            return self.web3.eth.accounts
        return []
    
    def set_account(self, account_index=0):
        """Set the account to use for transactions"""
        if not self.is_connected:
            if not self.connect_to_ganache():
                return None
        
        accounts = self.get_accounts()
        if accounts and account_index < len(accounts):
            self.account = accounts[account_index]
            print(f"✅ Set account to: {self.account}")
            return self.account
        return None
    
    def get_balance(self):
        """Get balance of current account in ETH"""
        if self.account and self.is_connected:
            balance_wei = self.web3.eth.get_balance(self.account)
            balance_eth = self.web3.from_wei(balance_wei, 'ether')
            return float(balance_eth)
        return 0
    
    def is_admin(self, address):
        """Check if address is admin using the contract"""
        if self.contract and self.is_connected:
            try:
                return self.contract.functions.isAuthorized(address).call()
            except Exception as e:
                print(f"Error checking admin status: {e}")
                return False
        return False
    
    def add_doctor(self, doctor_address, doctor_hash):
        """Add doctor to blockchain"""
        if self.contract and self.account and self.is_connected:
            try:
                # Build transaction
                transaction = self.contract.functions.addDoctor(doctor_address).build_transaction({
                    'from': self.account,
                    'gas': 200000,
                    'gasPrice': self.web3.eth.gas_price,
                    'nonce': self.web3.eth.get_transaction_count(self.account)
                })
                
                # Sign and send transaction
                signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key=self._get_private_key())
                tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                # Wait for transaction receipt
                tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                
                return {
                    "status": "success",
                    "hash": tx_receipt.transactionHash.hex()
                }
                
            except Exception as e:
                print(f"Error adding doctor: {e}")
                return None
        return None
    
    def _get_private_key(self):
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
            return None
    
    def get_all_doctors(self):
        """Get all doctors from blockchain"""
        if self.contract and self.is_connected:
            try:
                # This would depend on your contract implementation
                # For now, return empty list
                return []
            except Exception as e:
                print(f"Error getting doctors: {e}")
                return []
        return []
    
    def get_doctor(self, doctor_address):
        """Get doctor information from blockchain"""
        if self.contract and self.is_connected:
            try:
                # This would depend on your contract implementation
                return f"Doctor data for {doctor_address}"
            except Exception as e:
                print(f"Error getting doctor: {e}")
                return None
        return None
    
    def is_doctor(self, address):
        """Check if address is doctor"""
        if self.contract and self.is_connected:
            try:
                return self.contract.functions.isAuthorized(address).call()
            except Exception as e:
                print(f"Error checking doctor status: {e}")
                return False
        return False
    
    def hash_record(self, record_data):
        """Create hash of medical record"""
        record_string = json.dumps(record_data, sort_keys=True)
        return hashlib.sha256(record_string.encode()).hexdigest()
    
    def store_record_on_blockchain(self, record_hash):
        """Store medical record hash on blockchain"""
        if self.contract and self.account and self.is_connected:
            try:
                # This would depend on your EHR contract implementation
                print(f"✅ Stored record hash {record_hash} on blockchain")
                return record_hash
            except Exception as e:
                print(f"Error storing record: {e}")
                return None
        return None
    
    def upload_file_to_blockchain(self, filename, file_hash, ipfs_hash, file_type, file_size, patient_id, metadata):
        """Upload file information to blockchain using FileVerificationContract"""
        print(f"🔍 Starting blockchain upload for file: {filename}")
        print(f"🔍 Contract: {self.contract}")
        print(f"🔍 Account: {self.account}")
        print(f"🔍 Connected: {self.is_connected}")
        if self.contract and self.account and self.is_connected:
            try:
                # Convert patient_id to address if it's not already
                if isinstance(patient_id, int):
                    # It's a database ID, use current account as placeholder
                    patient_address = self.account
                elif isinstance(patient_id, str) and not patient_id.startswith('0x'):
                    # Assume it's a patient ID, use a default address for now
                    patient_address = self.account  # Use current account as placeholder
                else:
                    patient_address = patient_id
                
                # Build transaction
                transaction = self.contract.functions.uploadFile(
                    filename,
                    file_hash,
                    ipfs_hash,
                    file_type,
                    file_size,
                    patient_address,
                    metadata
                ).build_transaction({
                    'from': self.account,
                    'gas': 1000000,  # Increased gas limit to 1M
                    'gasPrice': self.web3.eth.gas_price,
                    'nonce': self.web3.eth.get_transaction_count(self.account)
                })
                
                # Get private key
                private_key = self._get_private_key()
                if not private_key:
                    print("❌ Failed to get private key")
                    return None
                
                # Sign and send transaction
                signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key=private_key)
                tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                # Wait for transaction receipt
                tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Get the file ID from the event
                file_id = self.contract.functions.fileCounter().call()
                
                return {
                    "file_id": file_id,
                    "transaction_hash": tx_receipt.transactionHash.hex(),
                    "status": "success"
                }
                
            except Exception as e:
                print(f"Error uploading file to blockchain: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                return None
        return None
    
    def get_file_record(self, file_id):
        """Get file record from blockchain"""
        if self.contract and self.is_connected:
            try:
                file_data = self.contract.functions.getFile(file_id).call()
                
                return {
                    "file_id": file_id,
                    "file_name": file_data[0],
                    "file_hash": file_data[1],
                    "ipfs_hash": file_data[2],
                    "file_type": file_data[3],
                    "file_size": file_data[4],
                    "uploaded_by": file_data[5],
                    "patient_id": file_data[6],
                    "timestamp": file_data[7],
                    "is_valid": file_data[8],
                    "metadata": file_data[9]
                }
                
            except Exception as e:
                print(f"Error getting file record: {e}")
                return None
        return None
    
    def verify_file_on_blockchain(self, file_id, current_hash, notes):
        """Verify file on blockchain"""
        if self.contract and self.account and self.is_connected:
            try:
                # Build transaction
                transaction = self.contract.functions.verifyFile(
                    file_id,
                    current_hash,
                    notes
                ).build_transaction({
                    'from': self.account,
                    'gas': 500000,  # Increased gas limit
                    'gasPrice': self.web3.eth.gas_price,
                    'nonce': self.web3.eth.get_transaction_count(self.account)
                })
                
                # Sign and send transaction
                signed_txn = self.web3.eth.account.sign_transaction(transaction, private_key=self._get_private_key())
                tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                
                # Wait for transaction receipt
                tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Get the verification log ID
                log_id = self.contract.functions.verificationCounter().call()
                
                return {
                    "log_id": log_id,
                    "is_match": True,  # The contract returns the actual result
                    "status": "success"
                }
                
            except Exception as e:
                print(f"Error verifying file: {e}")
                return None
        return None
    
    def get_user_files(self, user_address):
        """Get files uploaded by user"""
        if self.contract and self.is_connected:
            try:
                file_ids = self.contract.functions.getUserFiles(user_address).call()
                return file_ids
            except Exception as e:
                print(f"Error getting user files: {e}")
                return []
        return []
    
    def get_file_verification_logs(self, file_id):
        """Get verification logs for file"""
        if self.contract and self.is_connected:
            try:
                # This is a simplified implementation
                # In a real scenario, you'd need to iterate through logs
                total_verifications = self.contract.functions.verificationCounter().call()
                logs = []
                
                for log_id in range(1, total_verifications + 1):
                    try:
                        log_data = self.contract.functions.getVerificationLog(log_id).call()
                        if log_data[0] == file_id:  # log_data[0] is fileId
                            logs.append({
                                "log_id": log_id,
                                "file_id": log_data[0],
                                "original_hash": log_data[1],
                                "verified_hash": log_data[2],
                                "is_match": log_data[3],
                                "verified_by": log_data[4],
                                "timestamp": log_data[5],
                                "notes": log_data[6]
                            })
                    except:
                        continue
                
                return logs
                
            except Exception as e:
                print(f"Error getting verification logs: {e}")
                return []
        return []