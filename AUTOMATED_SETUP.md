# Automated Setup Scripts

**One-click setup and run scripts** that handle everything automatically!

These scripts will:
- ✅ Kill any existing process on the port
- ✅ Create Python virtual environment
- ✅ Install PyTorch with CUDA support
- ✅ Install all dependencies
- ✅ Download CogVideoX model (optional)
- ✅ Start the service automatically
- ✅ Open your browser to the web UI

**Port**: Uses port **8765** to avoid conflicts with other services.

---

## 🪟 Windows Users

### First Time Setup + Run

**Double-click** or run in Command Prompt:

```cmd
setup_and_run.bat
```

That's it! The script handles everything automatically.

### What It Does

1. Checks Python version (requires 3.10+)
2. Kills any process using port 8765
3. Creates virtual environment (if needed)
4. Installs PyTorch with CUDA 12.1
5. Installs all dependencies from requirements.txt
6. Asks if you want to download CogVideoX models
7. Starts the service on http://127.0.0.1:8765
8. Opens your browser automatically

### Subsequent Runs

After first setup, you can use the quick run script:

```cmd
# Activate venv
venv\Scripts\activate

# Quick run (kills port + starts service)
python quick_run.py
```

Or just run the full script again:
```cmd
setup_and_run.bat
```

---

## 🐧 Linux / 🍎 Mac Users

### First Time Setup + Run

Make executable and run:

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

Or run directly:

```bash
bash setup_and_run.sh
```

### What It Does

1. Checks Python 3.10+ installation
2. Kills any process using port 8765 (using lsof or fuser)
3. Creates virtual environment (if needed)
4. Installs PyTorch with CUDA 12.1
5. Installs all dependencies from requirements.txt
6. Asks if you want to download CogVideoX models
7. Starts the service on http://127.0.0.1:8765
8. Opens your browser automatically

### Subsequent Runs

After first setup:

```bash
# Activate venv
source venv/bin/activate

# Quick run
python quick_run.py
```

Or run the full script again:
```bash
./setup_and_run.sh
```

---

## ⚡ Quick Run Script (After Setup)

Once you've completed initial setup, use the fast Python script:

```bash
# Basic usage (port 8765)
python quick_run.py

# Custom port
python quick_run.py --port 9000

# Custom host and port
python quick_run.py --host 0.0.0.0 --port 9000

# Don't open browser
python quick_run.py --no-browser
```

**Features:**
- Automatically kills process on port
- Checks if virtual environment is activated
- Verifies dependencies are installed
- Opens browser automatically (unless --no-browser)

---

## 📋 Model Download Options

During setup, you'll be asked which model to download:

### Option 1: CogVideoX-5B (Recommended)
- **Quality**: ★★★★★ Best
- **VRAM**: 16GB+
- **Size**: ~20GB download
- **Time**: 10-20 minutes
- **Best for**: RTX 3090, 4080, 4090, A5000, A6000

### Option 2: CogVideoX-2B (Low-VRAM)
- **Quality**: ★★★★☆ Good
- **VRAM**: 8GB+
- **Size**: ~10GB download
- **Time**: 5-10 minutes
- **Best for**: RTX 3060 Ti, 3070, 4060 Ti

### Option 3: Skip Download
- Models will auto-download on first use
- First generation will take longer (extra 10-20 min)
- Good if you have slow internet or want to try the service first

---

## 🔧 Troubleshooting

### "Python not found" or "Python version too old"

**Windows:**
1. Download Python 3.10+ from https://python.org
2. During installation, check "Add Python to PATH"
3. Restart Command Prompt and try again

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv

# Mac
brew install python@3.10
```

### "Port already in use" - Script can't kill process

**Windows (Run as Administrator):**
1. Right-click Command Prompt
2. Select "Run as Administrator"
3. Run the script again

**Linux/Mac (Use sudo):**
```bash
sudo ./setup_and_run.sh
```

Or manually kill the process:
```bash
# Find process
lsof -ti:8765

# Kill it
kill -9 <PID>
```

### CUDA Not Available

The script installs PyTorch with CUDA 12.1 support. If CUDA still not available:

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```

2. **Install correct CUDA toolkit:**
   - CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - CUDA 12.1: https://developer.nvidia.com/cuda-12-1-0-download-archive

3. **Reinstall PyTorch manually:**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Script Hangs During Model Download

Model downloads are large (~10-20GB). If it seems stuck:

1. **Check internet connection**
2. **Wait longer** - can take 20+ minutes on slow connections
3. **Press Ctrl+C** to cancel and choose Option 3 (skip download)
4. Models will auto-download on first video generation

### Dependencies Installation Fails

If pip install fails:

1. **Update pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Try without quiet flag:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Check disk space** - need ~5GB for dependencies

---

## 🚀 Using a Different Port

The scripts use port **8765** by default. To change:

### Option 1: Edit Configuration (Persistent)

Edit `.env` file:
```env
PORT=9876
```

Then run quick_run.py normally.

### Option 2: Command Line (Temporary)

```bash
python quick_run.py --port 9876
```

### Option 3: Edit Script

**Windows (setup_and_run.bat):**
```batch
SET SERVICE_PORT=9876
```

**Linux/Mac (setup_and_run.sh):**
```bash
SERVICE_PORT=9876
```

---

## 🔐 Security Notes

### Binding to 0.0.0.0 (All Interfaces)

By default, service binds to `127.0.0.1` (localhost only) for security.

**To allow external access:**

Edit `.env`:
```env
HOST=0.0.0.0
```

Or use quick_run.py:
```bash
python quick_run.py --host 0.0.0.0
```

**⚠️ Warning**: Only do this on trusted networks. Anyone on your network can access the service.

### Firewall

If using 0.0.0.0, you may need to allow the port through firewall:

**Windows:**
```cmd
netsh advfirewall firewall add rule name="Video Service" dir=in action=allow protocol=TCP localport=8765
```

**Linux (ufw):**
```bash
sudo ufw allow 8765/tcp
```

---

## 📊 What Runs Where

### First Run (Full Setup)
```
setup_and_run.bat/sh
├── Check Python version
├── Kill process on port 8765
├── Create virtual environment
├── Install PyTorch + CUDA
├── Install requirements.txt
├── (Optional) Download CogVideoX model
└── Start service on port 8765
    └── Open browser automatically
```

### Subsequent Runs (Quick)
```
quick_run.py
├── Check virtual environment
├── Verify dependencies installed
├── Kill process on port
└── Start service
    └── Open browser automatically
```

---

## 💡 Tips

1. **First time?** Use `setup_and_run` script - it does everything
2. **Already set up?** Use `quick_run.py` - it's faster
3. **Want to customize?** Edit `.env` file for persistent config
4. **Need different port?** Use `--port` flag with quick_run.py
5. **Testing changes?** Keep service running, it hot-reloads code changes (if started with --reload)

---

## 📚 Additional Resources

- **Main README**: See `README.md` for full documentation
- **CogVideoX Guide**: See `COGVIDEOX_QUICKSTART.md` for model-specific info
- **API Reference**: See `API_REFERENCE.md` for API usage
- **Manual Setup**: See `README.md` if you prefer manual installation

---

## ❓ FAQ

**Q: Do I need to run the setup script every time?**
A: No! After first setup, use `quick_run.py` for faster startup.

**Q: Can I use a different Python version?**
A: Python 3.10, 3.11, or 3.12 are supported. 3.10+ required.

**Q: What if I don't have a CUDA GPU?**
A: Service will run on CPU but will be extremely slow (not recommended).

**Q: Can I run multiple instances?**
A: Yes, use different ports for each instance:
```bash
python quick_run.py --port 8765  # Instance 1
python quick_run.py --port 8766  # Instance 2
```

**Q: How do I stop the service?**
A: Press `Ctrl+C` in the terminal window.

**Q: How do I update the service?**
A: Pull latest code and run setup script again:
```bash
git pull
./setup_and_run.sh  # or setup_and_run.bat on Windows
```

---

**Ready to generate videos in one click!** 🎬✨

Simply run:
- Windows: `setup_and_run.bat`
- Linux/Mac: `./setup_and_run.sh`
