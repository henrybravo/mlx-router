#!/bin/bash
# MLX Router Installation Script for macOS

set -e

echo "🚀 Installing MLX Router..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check system requirements
log_info "Checking system requirements..."

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_error "This script is designed for macOS only"
    exit 1
fi

# Check Python 3.10+
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    log_error "Python $PYTHON_VERSION found, but Python $REQUIRED_VERSION+ is required"
    exit 1
fi

log_success "Python $PYTHON_VERSION found"

# Check for uv availability
USE_UV=false
if command -v uv &> /dev/null; then
    log_success "uv package manager found - using uv for faster installation"
    USE_UV=true
else
    log_warning "uv not found - using standard pip (consider installing uv for faster package installation)"
fi

# Check if service already exists
if sudo launchctl list | grep -q com.henrybravo.mlx-router; then
    log_warning "MLX Router service already exists. Unloading existing service..."
    sudo launchctl unload /Library/LaunchDaemons/com.henrybravo.mlx-router.plist 2>/dev/null || true
fi

# Get current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check required files
log_info "Checking required files..."
REQUIRED_FILES=("mlx_router" "main.py" "requirements.txt" "config.json" "com.henrybravo.mlx-router.plist" "logs")
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -e "$SCRIPT_DIR/$file" ]]; then
        log_error "Required file not found: $file"
        exit 1
    fi
done
log_success "All required files found"

# Create directories
log_info "Creating system directories..."
sudo mkdir -p /usr/local/opt/mlx-router/logs
sudo mkdir -p /usr/local/etc/mlx-router
log_success "System directories created"

# Copy files
log_info "Copying application files..."
sudo cp -r "$SCRIPT_DIR/mlx_router" "$SCRIPT_DIR/main.py" "$SCRIPT_DIR/requirements.txt" "$SCRIPT_DIR/logs" /usr/local/opt/mlx-router/
sudo cp "$SCRIPT_DIR/config.json" /usr/local/etc/mlx-router/
log_success "Application files copied"

# Create virtual environment
log_info "Creating Python virtual environment..."
sudo python3 -m venv /usr/local/opt/mlx-router/venv
sudo chown -R root:wheel /usr/local/opt/mlx-router/venv
log_success "Virtual environment created"

# Install Python dependencies
log_info "Installing Python dependencies..."
if [[ "$USE_UV" == true ]]; then
    # Use uv for installation (faster)
    log_info "Installing uv package manager..."
    if ! sudo /usr/local/opt/mlx-router/venv/bin/python -m pip install uv; then
        log_error "Failed to install uv, falling back to pip"
        USE_UV=false
    fi
fi

if [[ "$USE_UV" == true ]]; then
    log_info "Installing dependencies with uv..."
    # Force uv to use the correct virtual environment
    if ! sudo VIRTUAL_ENV=/usr/local/opt/mlx-router/venv /usr/local/opt/mlx-router/venv/bin/uv pip install --python /usr/local/opt/mlx-router/venv/bin/python -r /usr/local/opt/mlx-router/requirements.txt; then
        log_error "uv installation failed, falling back to pip"
        USE_UV=false
    fi
fi

if [[ "$USE_UV" == false ]]; then
    # Use standard pip with optimizations
    log_info "Upgrading pip..."
    sudo /usr/local/opt/mlx-router/venv/bin/pip install --upgrade pip
    log_info "Installing dependencies with pip..."
    sudo /usr/local/opt/mlx-router/venv/bin/pip install -r /usr/local/opt/mlx-router/requirements.txt
fi

# Verify and fix critical dependencies
log_info "Verifying Python dependencies..."
CRITICAL_DEPS=("mlx" "mlx_lm" "fastapi" "pyfiglet" "psutil" "uvicorn")
FAILED_DEPS=()

for dep in "${CRITICAL_DEPS[@]}"; do
    if ! sudo /usr/local/opt/mlx-router/venv/bin/python -c "import $dep" 2>/dev/null; then
        log_warning "Dependency '$dep' not found, attempting to install individually..."
        FAILED_DEPS+=("$dep")
    fi
done

# Fix failed dependencies individually
if [[ ${#FAILED_DEPS[@]} -gt 0 ]]; then
    log_info "Installing missing dependencies individually..."
    for dep in "${FAILED_DEPS[@]}"; do
        log_info "Installing $dep..."
        if ! sudo /usr/local/opt/mlx-router/venv/bin/pip install "$dep"; then
            log_error "Failed to install critical dependency '$dep'"
            exit 1
        fi
        
        # Verify the fix worked
        if sudo /usr/local/opt/mlx-router/venv/bin/python -c "import $dep" 2>/dev/null; then
            log_success "$dep installed successfully"
        else
            log_error "$dep still not working after installation"
            exit 1
        fi
    done
fi

# Final verification of main.py imports
log_info "Testing main application imports..."
if ! sudo /usr/local/opt/mlx-router/venv/bin/python -c "import sys; sys.path.insert(0, '/usr/local/opt/mlx-router'); import main" 2>/dev/null; then
    log_error "main.py import test failed. Attempting to fix..."
    
    # Try reinstalling all requirements as a last resort
    log_info "Reinstalling all requirements..."
    sudo /usr/local/opt/mlx-router/venv/bin/pip install --force-reinstall -r /usr/local/opt/mlx-router/requirements.txt
    
    # Test again
    if ! sudo /usr/local/opt/mlx-router/venv/bin/python -c "import sys; sys.path.insert(0, '/usr/local/opt/mlx-router'); import main" 2>/dev/null; then
        log_error "main.py import still failing after reinstall. Check error details:"
        sudo /usr/local/opt/mlx-router/venv/bin/python -c "import sys; sys.path.insert(0, '/usr/local/opt/mlx-router'); import main" 2>&1 | head -5
        exit 1
    fi
fi

log_success "All Python dependencies verified and working"

# Install launchd plist
log_info "Installing system service..."
sudo cp "$SCRIPT_DIR/com.henrybravo.mlx-router.plist" /Library/LaunchDaemons/
sudo launchctl load /Library/LaunchDaemons/com.henrybravo.mlx-router.plist
log_success "System service installed and started"

# Wait for service to start
log_info "Waiting for service to start..."
sleep 3

# Verify installation
log_info "Verifying installation..."
if sudo launchctl list | grep -q com.henrybravo.mlx-router; then
    log_success "Service is running"
else
    log_error "Service failed to start - check logs at /usr/local/opt/mlx-router/logs/mlx-router.error.log"
    exit 1
fi

# Test API endpoint
if curl -s --max-time 5 http://localhost:8800/health >/dev/null 2>&1; then
    log_success "API endpoint responding"
else
    log_warning "API endpoint not yet responding (may still be starting up) or you have specified another port than default 8800 in config.json"
fi

echo ""
log_success "MLX Router installed and started successfully!"
echo ""
echo "📍 API: http://localhost:8800"
echo "📄 Interactive docs: http://localhost:8800/docs"
echo "📊 Logs: /usr/local/opt/mlx-router/logs/mlx_router.log"
echo "🚫 Error logs: /usr/local/opt/mlx-router/logs/mlx_router.error.log"
echo ""
echo "🔧 Management commands:"
echo "   Stop:      sudo launchctl unload /Library/LaunchDaemons/com.henrybravo.mlx-router.plist"
echo "   Start:     sudo launchctl load /Library/LaunchDaemons/com.henrybravo.mlx-router.plist"
echo "   Uninstall: ./uninstall-launchd.sh"