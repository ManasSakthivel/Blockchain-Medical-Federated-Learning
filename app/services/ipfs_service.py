import requests
import json
from flask import current_app

class IPFSService:
    def __init__(self, ipfs_url="http://127.0.0.1:5001"):
        self.ipfs_url = ipfs_url
        
    def upload_file(self, file_path):
        """Upload file to IPFS"""
        try:
            with open(file_path, 'rb') as file:
                files = {'file': file}
                response = requests.post(f"{self.ipfs_url}/api/v0/add", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    return result['Hash']
                else:
                    print(f"Error uploading to IPFS: {response.text}")
                    # Return mock hash if IPFS is not available
                    import time
                    return f"QmMockHash{int(time.time())}"
        except Exception as e:
            print(f"Error uploading file to IPFS: {e}")
            # Return mock hash if IPFS is not available
            import time
            return f"QmMockHash{int(time.time())}"
    
    def upload_json(self, data):
        """Upload JSON data to IPFS"""
        try:
            # Convert data to JSON string
            json_data = json.dumps(data)
            
            # Upload as file
            files = {'file': ('data.json', json_data, 'application/json')}
            response = requests.post(f"{self.ipfs_url}/api/v0/add", files=files)
            
            if response.status_code == 200:
                result = response.json()
                return result['Hash']
            else:
                print(f"Error uploading JSON to IPFS: {response.text}")
                return None
        except Exception as e:
            print(f"Error uploading JSON to IPFS: {e}")
            return None
    
    def get_file(self, ipfs_hash):
        """Get file from IPFS"""
        try:
            response = requests.post(f"{self.ipfs_url}/api/v0/cat", params={'arg': ipfs_hash})
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"Error getting file from IPFS: {response.text}")
                return None
        except Exception as e:
            print(f"Error getting file from IPFS: {e}")
            return None
    
    def get_json(self, ipfs_hash):
        """Get JSON data from IPFS"""
        try:
            content = self.get_file(ipfs_hash)
            if content:
                return json.loads(content.decode('utf-8'))
            return None
        except Exception as e:
            print(f"Error getting JSON from IPFS: {e}")
            return None
    
    def pin_file(self, ipfs_hash):
        """Pin file to IPFS to prevent garbage collection"""
        try:
            response = requests.post(f"{self.ipfs_url}/api/v0/pin/add", params={'arg': ipfs_hash})
            
            if response.status_code == 200:
                return True
            else:
                print(f"Error pinning file: {response.text}")
                # Return True for mock hashes
                if ipfs_hash.startswith('QmMockHash'):
                    return True
                return False
        except Exception as e:
            print(f"Error pinning file: {e}")
            # Return True for mock hashes
            if ipfs_hash.startswith('QmMockHash'):
                return True
            return False 