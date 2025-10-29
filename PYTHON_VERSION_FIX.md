# Python Version Fix Guide

## Problem
You have Python 3.9.13, but this service requires Python 3.10+.

---

## ✅ Solution: Install Python 3.11 (Recommended)

Install Python 3.11 **alongside** your existing Python 3.9 - they won't conflict!

### Step 1: Download Python 3.11

1. Go to: **https://www.python.org/downloads/**
2. Download **Python 3.11.9** (or latest 3.11.x)
3. Run the installer

### Step 2: Installation Settings

**IMPORTANT - Check these boxes:**
- ✅ **"Add Python 3.11 to PATH"**
- ✅ **"Install launcher for all users (recommended)"**
- Click **"Customize installation"**
- ✅ Make sure **"py launcher"** is checked
- Complete the installation

### Step 3: Verify Installation

Open **new** Command Prompt or PowerShell:

```powershell
py -3.11 --version
```

Should show: `Python 3.11.x`

Also check:
```powershell
py --list
```

Should show both versions:
```
 -3.11-64        Python 3.11.9
 -3.9-64 *       Python 3.9.13
```

---

## 🚀 Running the Service with Python 3.11

### Option A: Use the Special Script (Easiest)

I created a script that uses Python 3.11 specifically:

```powershell
.\setup_and_run_py311.bat
```

This script uses `py -3.11` launcher to target Python 3.11.

### Option B: Manual Setup

```powershell
# Create virtual environment with Python 3.11
py -3.11 -m venv venv

# Activate it
.\venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run the service
python quick_run.py
```

---

## 🔍 Alternative Solutions

### Option 1: Update Conda Base Environment

If you're using Anaconda (I see `(base)` in your prompt):

```powershell
# Create new conda environment with Python 3.11
conda create -n video311 python=3.11 -y

# Activate it
conda activate video311

# Navigate to project
cd E:\Lora2\local_video_service

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Run service
python quick_run.py
```

### Option 2: Update Your Base Conda Environment

⚠️ **Warning**: This updates your base environment - may affect other projects!

```powershell
# Update conda base to Python 3.11
conda install python=3.11 -y

# Then run the original script
.\setup_and_run.bat
```

---

## 📊 Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Install Python 3.11** | ✅ Keeps Python 3.9<br>✅ Easy to switch<br>✅ No conda needed | Need to download installer |
| **New Conda Env** | ✅ Isolated environment<br>✅ Easy to manage | Only works if using Anaconda |
| **Update Base Conda** | ✅ Quick | ⚠️ Affects all projects using base |

---

## 🎯 Recommended Approach

### For You (Using Conda):

**Create a dedicated conda environment** (best practice):

```powershell
# 1. Create environment
conda create -n video_gen python=3.11 -y

# 2. Activate
conda activate video_gen

# 3. Go to project
cd E:\Lora2\local_video_service

# 4. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install requirements
pip install -r requirements.txt

# 6. Run service
python quick_run.py
```

### Future Runs:

```powershell
conda activate video_gen
cd E:\Lora2\local_video_service
python quick_run.py
```

---

## ❓ FAQ

**Q: Will this break my existing Python 3.9 projects?**
A: No! Multiple Python versions can coexist. Use `py -3.9` for old projects and `py -3.11` for this one.

**Q: Should I uninstall Python 3.9?**
A: No! Keep it. Just install 3.11 alongside it.

**Q: Which Python will be default?**
A: The one installed last (3.11), but you can always specify: `py -3.9` or `py -3.11`

**Q: Can I use Python 3.12?**
A: Yes! Python 3.10, 3.11, or 3.12 all work. Just change the script to `py -3.12`.

**Q: I'm using Conda, what should I do?**
A: Best practice: Create a new environment with `conda create -n video_gen python=3.11`

---

## 🆘 Still Having Issues?

### Check Python Installation:

```powershell
# List all Python versions
py --list

# Check specific version
py -3.11 --version
py -3.10 --version
py -3.12 --version
```

### Conda Environments:

```powershell
# List conda environments
conda env list

# Check current environment
conda info

# Check Python version in current environment
python --version
```

---

## 🎬 Quick Start (After Installing Python 3.11)

```powershell
# Using the special script
.\setup_and_run_py311.bat
```

or

```powershell
# Using conda (recommended)
conda create -n video_gen python=3.11 -y
conda activate video_gen
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python quick_run.py
```

**Done!** Service will start on http://localhost:8765 🎉
