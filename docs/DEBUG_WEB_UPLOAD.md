# Debugging Web Interface Upload Issues

## ✅ **System Status: WORKING**

The backend system is working correctly:
- ✅ Blockchain connection: Active
- ✅ Smart contract: Deployed and accessible
- ✅ IPFS service: Working (real IPFS hashes being generated)
- ✅ File upload: Successfully storing files on blockchain
- ✅ Gas limits: Properly configured (1,000,000 gas)

## 🔍 **Troubleshooting Steps**

### 1. **Check Authentication**
Make sure you're logged in as a user with the correct role:
- **Required roles**: `doctor`, `lab`, or `admin`
- **Access URL**: http://127.0.0.1:5002/file-verification/upload
- **If redirected to login**: You need to authenticate first

### 2. **Check Console Logs**
When you try to upload a file, check the Flask application console for these logs:
```
🔍 Uploading to IPFS: [file_path]
🔍 IPFS hash result: [hash]
🔍 Pinning file on IPFS: [hash]
🔍 Pin result: [True/False]
🔍 Starting blockchain upload for file: [filename]
🔍 Contract: [contract_object]
🔍 Account: [account_address]
🔍 Connected: [True/False]
```

### 3. **Common Issues and Solutions**

#### **Issue: "Upload failed: Failed to store on blockchain"**
**Possible causes:**
- Not logged in with correct role
- IPFS not running (but this should use mock hash)
- Network connectivity issues
- Form data not properly formatted

**Solutions:**
1. **Login as doctor/lab/admin**: Go to http://127.0.0.1:5002/auth/login
2. **Check Flask console**: Look for error messages
3. **Verify Ganache**: Make sure Ganache is running
4. **Check file size**: Ensure file is under 50MB

#### **Issue: "No file selected"**
**Solution:**
- Make sure you're selecting a file in the upload form
- Check that the file extension is allowed

#### **Issue: "Patient not found"**
**Solution:**
- Make sure you're selecting a valid patient email from the dropdown
- The patient must exist in the database

### 4. **Test the System**

Run this test to verify everything is working:
```bash
python test_web_interface_upload.py
```

Expected output:
```
🎉 Web interface upload successful!
   File ID: [number]
   Transaction Hash: [hash]
   File Hash: [hash]
   IPFS Hash: [hash]
```

### 5. **Manual Testing Steps**

1. **Start Ganache**: Make sure it's running
2. **Start Flask app**: `python run.py`
3. **Login**: Go to http://127.0.0.1:5002/auth/login
4. **Navigate**: Go to File Verification → Upload File
5. **Upload**: Select a file and patient, then upload
6. **Check logs**: Look at Flask console for detailed logs

### 6. **If Still Not Working**

1. **Check Flask console**: Look for any error messages
2. **Verify Ganache**: Make sure it's running and accessible
3. **Check database**: Ensure patients exist in the database
4. **Test with simple file**: Try uploading a small text file first

## 🎯 **Expected Behavior**

When working correctly, you should see:
1. File uploads successfully
2. IPFS hash is generated
3. File is stored on blockchain
4. Success message with File ID
5. Transaction visible in Ganache

## 📞 **Need Help?**

If you're still experiencing issues:
1. Share the exact error message
2. Share the Flask console logs
3. Describe the steps you're taking
4. Mention your user role and authentication status
