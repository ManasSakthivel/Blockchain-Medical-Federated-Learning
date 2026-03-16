#!/usr/bin/env python3
"""
Setup script for EHR Blockchain System
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_virtual_environment():
    """Create a virtual environment."""
    if os.path.exists("venv"):
        print("✓ Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def install_dependencies():
    """Install Python dependencies."""
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
    
    return run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies")

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    env_content = """# Flask Configuration
SECRET_KEY=your-super-secret-key-change-this-in-production
DATABASE_URL=sqlite:///ehr.db

# Blockchain Configuration
GANACHE_URL=http://127.0.0.1:7545
IPFS_URL=http://127.0.0.1:5001
CONTRACT_ADDRESS=your-deployed-contract-address

# Environment
FLASK_ENV=development
FLASK_DEBUG=1
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✓ Created .env file")
        return True
    except Exception as e:
        print(f"✗ Failed to create .env file: {e}")
        return False

def create_uploads_directory():
    """Create uploads directory."""
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        print("✓ Uploads directory already exists")
        return True
    
    try:
        uploads_dir.mkdir()
        print("✓ Created uploads directory")
        return True
    except Exception as e:
        print(f"✗ Failed to create uploads directory: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up EHR Blockchain System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Create uploads directory
    if not create_uploads_directory():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    
    print("2. Set up Ganache (local Ethereum blockchain)")
    print("3. Set up IPFS (decentralized storage)")
    print("4. Deploy smart contracts using Truffle")
    print("5. Initialize database: flask init-db")
    print("6. Run the application: python run.py")
    print("\nFor detailed instructions, see README.md")

if __name__ == "__main__":
    main() 