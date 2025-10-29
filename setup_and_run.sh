#!/bin/bash
# ============================================================================
# Local Video Generation Service - Automated Setup and Run Script (Linux/Mac)
# ============================================================================
# This script handles everything:
# - Kills existing processes on port
# - Creates virtual environment
# - Installs all dependencies
# - Downloads CogVideoX model (optional)
# - Starts the service
# ============================================================================

set -e  # Exit on error

# Configuration
SERVICE_PORT=8765
SERVICE_HOST="127.0.0.1"
VENV_DIR="venv"
PYTHON_MIN_VERSION="3.10"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo ""
echo "============================================================================"
echo "     Local Video Generation Service - Automated Setup"
echo "============================================================================"
echo ""

# Step 0: Check if Python is installed
log_info "Checking Python installation..."

if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed"
    echo "Please install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
log_ok "Found Python $PYTHON_VERSION"

# Check Python version
PYTHON_VERSION_CHECK=$(python3 -c "import sys; print('ok' if sys.version_info >= (3, 10) else 'fail')")
if [ "$PYTHON_VERSION_CHECK" != "ok" ]; then
    log_error "Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

log_ok "Python version compatible"
echo ""

# Step 1: Kill any process using the port
echo "============================================================================"
echo "Step 1: Checking port $SERVICE_PORT"
echo "============================================================================"
echo ""

if command -v lsof &> /dev/null; then
    # Use lsof on Mac/Linux
    PID=$(lsof -ti:$SERVICE_PORT 2>/dev/null || true)
    if [ -n "$PID" ]; then
        log_info "Found process $PID using port $SERVICE_PORT"
        log_info "Killing process $PID..."
        kill -9 $PID 2>/dev/null || log_warning "Could not kill process (may need sudo)"
        log_ok "Process killed successfully"
    else
        log_ok "Port $SERVICE_PORT is available"
    fi
elif command -v fuser &> /dev/null; then
    # Use fuser on Linux if lsof not available
    if fuser $SERVICE_PORT/tcp &> /dev/null; then
        log_info "Killing process on port $SERVICE_PORT..."
        fuser -k $SERVICE_PORT/tcp 2>/dev/null || log_warning "Could not kill process (may need sudo)"
        log_ok "Process killed successfully"
    else
        log_ok "Port $SERVICE_PORT is available"
    fi
else
    log_warning "Cannot check port status (lsof/fuser not found)"
fi

echo ""
sleep 2

# Step 2: Create virtual environment
echo "============================================================================"
echo "Step 2: Setting up Python virtual environment"
echo "============================================================================"
echo ""

if [ -d "$VENV_DIR" ]; then
    log_info "Virtual environment already exists at $VENV_DIR"
    log_info "Using existing environment..."
else
    log_info "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    log_ok "Virtual environment created"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source $VENV_DIR/bin/activate
log_ok "Virtual environment activated"
echo ""

# Step 3: Install dependencies
echo "============================================================================"
echo "Step 3: Installing dependencies"
echo "============================================================================"
echo ""

# Check if PyTorch is installed
if python -c "import torch" &> /dev/null; then
    log_ok "PyTorch already installed"
else
    log_info "PyTorch not found, installing PyTorch with CUDA 12.1..."
    log_info "This may take 5-10 minutes..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet --upgrade || {
        log_warning "PyTorch installation failed, trying CPU version..."
        pip install torch torchvision torchaudio --quiet --upgrade
    }
    log_ok "PyTorch installed"
fi

# Check if diffusers is installed with correct version
if python -c "import diffusers; from packaging import version; exit(0 if version.parse(diffusers.__version__) >= version.parse('0.30.0') else 1)" &> /dev/null; then
    log_ok "Dependencies already installed"
else
    log_info "Installing/upgrading dependencies from requirements.txt..."
    log_info "This may take 5-10 minutes..."
    pip install -r requirements.txt --quiet --upgrade
    log_ok "Dependencies installed"
fi

echo ""

# Check CUDA availability
log_info "Checking CUDA availability..."
python -c "import torch; print('[OK] CUDA Available: ' + str(torch.cuda.is_available())); print('[INFO] Device: ' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"
echo ""

# Step 4: Setup configuration
echo "============================================================================"
echo "Step 4: Configuration"
echo "============================================================================"
echo ""

if [ ! -f ".env" ]; then
    log_info "Creating .env configuration file..."
    cp .env.example .env
    # Update port in .env
    if command -v sed &> /dev/null; then
        sed -i.bak "s/PORT=8000/PORT=$SERVICE_PORT/g" .env
        rm -f .env.bak
    fi
    log_ok "Configuration file created"
else
    log_ok "Configuration file already exists"
    # Update port anyway
    if command -v sed &> /dev/null; then
        sed -i.bak "s/PORT=[0-9]*/PORT=$SERVICE_PORT/g" .env
        rm -f .env.bak
        log_info "Updated port to $SERVICE_PORT in .env"
    fi
fi

echo ""

# Step 5: Download models (optional)
echo "============================================================================"
echo "Step 5: Download CogVideoX Model (Optional)"
echo "============================================================================"
echo ""
echo "CogVideoX-5B is ~20GB and takes 10-20 minutes to download."
echo "The model will auto-download on first use if you skip this step."
echo ""
echo "Download now? (Recommended for better experience)"
echo "[1] Yes - Download CogVideoX-5B (16GB VRAM, best quality)"
echo "[2] Yes - Download CogVideoX-2B (8GB VRAM, low-VRAM mode)"
echo "[3] No - Skip download (will auto-download on first use)"
echo ""

read -p "Enter choice [1/2/3]: " DOWNLOAD_CHOICE

case $DOWNLOAD_CHOICE in
    1)
        echo ""
        log_info "Downloading CogVideoX-5B (~20GB)..."
        log_info "This will take 10-20 minutes depending on your internet speed..."
        pip install huggingface-hub --quiet
        mkdir -p weights/cogvideox
        python -c "from huggingface_hub import snapshot_download; snapshot_download('THUDM/CogVideoX-5b', local_dir='./weights/cogvideox/CogVideoX-5b')" || {
            log_warning "Download failed, model will auto-download on first use"
        }
        log_ok "CogVideoX-5B downloaded successfully"
        ;;
    2)
        echo ""
        log_info "Downloading CogVideoX-2B (~10GB)..."
        log_info "This will take 5-10 minutes depending on your internet speed..."
        pip install huggingface-hub --quiet
        mkdir -p weights/cogvideox
        python -c "from huggingface_hub import snapshot_download; snapshot_download('THUDM/CogVideoX-2b', local_dir='./weights/cogvideox/CogVideoX-2b')" || {
            log_warning "Download failed, model will auto-download on first use"
        }
        log_ok "CogVideoX-2B downloaded successfully"
        # Enable LOW_VRAM mode
        if command -v sed &> /dev/null; then
            sed -i.bak "s/LOW_VRAM=false/LOW_VRAM=true/g" .env
            rm -f .env.bak
            log_info "Enabled LOW_VRAM mode in .env"
        fi
        ;;
    *)
        log_info "Skipping download - model will auto-download on first use"
        ;;
esac

echo ""

# Step 6: Start the service
echo "============================================================================"
echo "Step 6: Starting Video Generation Service"
echo "============================================================================"
echo ""

log_info "Service will start on http://$SERVICE_HOST:$SERVICE_PORT"
log_info "Press Ctrl+C to stop the service"
echo ""
sleep 3

echo "============================================================================"
echo "     Service Starting..."
echo "============================================================================"
echo ""

# Start browser after a delay (in background)
(sleep 5 && open "http://$SERVICE_HOST:$SERVICE_PORT" 2>/dev/null || xdg-open "http://$SERVICE_HOST:$SERVICE_PORT" 2>/dev/null || sensible-browser "http://$SERVICE_HOST:$SERVICE_PORT" 2>/dev/null || true) &

# Start the service
python -m uvicorn app.main:app --host $SERVICE_HOST --port $SERVICE_PORT --log-level info

# If service exits
echo ""
echo "============================================================================"
echo "Service stopped"
echo "============================================================================"
echo ""
