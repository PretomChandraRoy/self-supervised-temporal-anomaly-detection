# NumPy Compatibility Issue - Quick Fix Guide

## Problem

You're seeing this error:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.2
```

This happens because you have NumPy 2.x installed, but PyTorch and matplotlib were compiled with NumPy 1.x.

---

## Solution

### **Option 1: Quick Fix (Recommended)**

Run the provided fix script:

**Windows:**
```bash
cd anomaly_detection
fix_install.bat
```

**Linux/Mac:**
```bash
cd anomaly_detection
chmod +x fix_install.sh
./fix_install.sh
```

---

### **Option 2: Manual Fix**

Run these commands in order:

```bash
# 1. Uninstall NumPy 2.x
pip uninstall -y numpy

# 2. Install NumPy 1.x (compatible version)
pip install "numpy<2.0.0"

# 3. Reinstall PyTorch (optional but recommended)
pip install --force-reinstall torch torchvision

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

### **Option 3: Create Fresh Virtual Environment**

This is the cleanest solution:

```bash
# Create new environment
python -m venv anomaly_env

# Activate it
# Windows:
anomaly_env\Scripts\activate
# Linux/Mac:
source anomaly_env/bin/activate

# Install dependencies
cd anomaly_detection
pip install -r requirements.txt
```

---

## Verify Installation

After fixing, verify everything works:

```bash
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import matplotlib; print('Matplotlib OK')"
```

**Expected output:**
```
NumPy version: 1.26.x  (should be < 2.0)
PyTorch version: 2.x.x
Matplotlib OK
```

---

## Test the Framework

```bash
python example.py
```

If successful, you should see:
```
================================================================================
Anomaly Detection Framework - Example Usage
================================================================================

[1/6] Loading data...
✓ Loaded X sequences with Y features
...
```

---

## Why This Happened

- Your system has multiple packages with conflicting requirements:
  - **opencv-python 4.13.x** requires NumPy >= 2.0
  - **PyTorch 2.10.0** doesn't match torchvision 0.21.0 (needs 2.6.0)
  - **tslearn 0.6.4** requires scipy < 1.17
  - **scipy 1.17.0** is incompatible with tslearn

The solution is to use compatible versions of all packages.

---

## Permanent Fix

The `requirements.txt` has been updated with compatible versions:
```
numpy>=2.0.0          # Required by opencv-python
torch==2.6.0          # Compatible with torchvision
torchvision==0.21.0   # Compatible with torch
scipy>=1.10.0,<1.17.0 # Compatible with tslearn
```

---

## Still Having Issues?

### Error: "torch not found"
```bash
pip install torch torchvision
```

### Error: "ta not found"
```bash
pip install ta
```

### Error: Multiple imports
Close all Python processes and try again:
```bash
# Windows
taskkill /F /IM python.exe
# Linux/Mac
killall python
```

Then run the fix script again.

---

## Next Steps

Once fixed:
1. ✅ Run `python example.py` to test
2. ✅ Try `python main.py --help` to see options
3. ✅ Read `GETTING_STARTED.md` for full guide

---

**The requirements.txt has been updated to prevent this issue in the future!**

