#!/bin/bash
# MLX Router Installation Script for macOS

set -e

# Configurable install directory (defaults to $HOME/mlx_router_app)
INSTALL_DIR="${INSTALL_DIR:-$HOME/mlx_router_app}"

# Set the python venv version you want to run mlx_router: 3.11-3.13
VENV_PYTHON_VERSION="3.13"

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

# Check for Python 3.11+ (try 3.13, 3.12, 3.11 in order)
PYTHON_CMD=""
PYTHON_VERSION=""

for version in 13 12 11; do
    if command -v python3.${version} &> /dev/null; then
        candidate_version=$(python3.${version} -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null)
        if [[ "$candidate_version" == "3.${version}" ]]; then
            PYTHON_CMD="python3.${version}"
            PYTHON_VERSION="$candidate_version"
            break
        fi
    fi
done

# Fallback to python3 if specific versions not found
if [[ -z "$PYTHON_CMD" ]]; then
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.11"
        if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" == "$REQUIRED_VERSION" ]]; then
            PYTHON_CMD="python3"
        else
            log_error "Python $PYTHON_VERSION found, but Python 3.11+ is required"
            exit 1
        fi
    else
        log_error "Python 3.11+ is not installed. Please install Python 3.11+ first."
        exit 1
    fi
fi

log_success "Python $PYTHON_VERSION found (using $PYTHON_CMD)"

# Check for uv availability
USE_UV=false
if command -v uv &> /dev/null; then
    log_success "uv package manager found - using uv for faster installation"
    USE_UV=true
else
    log_warning "uv not found - using standard pip (consider installing uv for faster package installation)"
fi

# Check if service already exists
if launchctl list | grep -q com.henrybravo.mlx-router; then
    log_warning "MLX Router service already exists. Unloading existing service..."
    launchctl unload "$HOME/Library/LaunchAgents/com.henrybravo.mlx-router.plist" 2>/dev/null || true
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

log_info "Installing in venv with python $VENV_PYTHON_VERSION version"

# Create directories
log_info "Creating user directories..."
mkdir -p "$INSTALL_DIR/logs"
log_success "User directories created"

# Copy files
log_info "Copying application files..."
cp -r "$SCRIPT_DIR/mlx_router" "$SCRIPT_DIR/main.py" "$SCRIPT_DIR/requirements.txt" "$SCRIPT_DIR/logs" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/config.json" "$INSTALL_DIR/"
log_success "Application files copied"

# Create virtual environment
log_info "Creating Python virtual environment..."
if [[ "$USE_UV" == true ]]; then
    log_info "Creating venv with uv..."
    uv venv --python $VENV_PYTHON_VERSION --seed "$INSTALL_DIR/venv"
else
    $PYTHON_CMD -m venv "$INSTALL_DIR/venv"
fi
log_success "Virtual environment created"

# Install Python dependencies
log_info "Installing Python dependencies..."
if [[ "$USE_UV" == true ]]; then
    log_info "Installing dependencies with uv..."
    # Use global uv with local cache to avoid permission issues
    # Allow pre-releases for mlx-lm 0.30.0 which depends on transformers==5.0.0rc1
    if ! uv --cache-dir "$INSTALL_DIR/.cache" pip install --prerelease=allow --python "$INSTALL_DIR/venv/bin/python" -r "$INSTALL_DIR/requirements.txt"; then
        log_error "uv installation failed, falling back to pip"
        USE_UV=false
    fi
fi

if [[ "$USE_UV" == false ]]; then
    # Use standard pip with optimizations
    log_info "Upgrading pip..."
    "$INSTALL_DIR/venv/bin/pip" install --upgrade pip
    log_info "Installing dependencies with pip..."
    "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"
fi

# Verify and fix critical dependencies
log_info "Verifying Python dependencies..."
CRITICAL_DEPS=("mlx" "mlx_lm" "fastapi" "pyfiglet" "psutil" "uvicorn")
FAILED_DEPS=()

for dep in "${CRITICAL_DEPS[@]}"; do
    if ! "$INSTALL_DIR/venv/bin/python" -c "import $dep" 2>/dev/null; then
        log_warning "Dependency '$dep' not found, attempting to install individually..."
        FAILED_DEPS+=("$dep")
    fi
done

# Fix failed dependencies individually
if [[ ${#FAILED_DEPS[@]} -gt 0 ]]; then
    log_info "Installing missing dependencies individually..."
    for dep in "${FAILED_DEPS[@]}"; do
        log_info "Installing $dep..."
        if ! "$INSTALL_DIR/venv/bin/pip" install "$dep"; then
            log_error "Failed to install critical dependency '$dep'"
            exit 1
        fi

        # Verify the fix worked
        if "$INSTALL_DIR/venv/bin/python" -c "import $dep" 2>/dev/null; then
            log_success "$dep installed successfully"
        else
            log_error "$dep still not working after installation"
            exit 1
        fi
    done
fi

# Final verification of main.py imports
log_info "Testing main application imports..."
if ! "$INSTALL_DIR/venv/bin/python" -c "import sys; sys.path.insert(0, '$INSTALL_DIR'); import main" 2>/dev/null; then
    log_error "main.py import test failed. Attempting to fix..."

    # Try reinstalling all requirements as a last resort
    log_info "Reinstalling all requirements..."
    "$INSTALL_DIR/venv/bin/pip" install --force-reinstall -r "$INSTALL_DIR/requirements.txt"

    # Test again
    if ! "$INSTALL_DIR/venv/bin/python" -c "import sys; sys.path.insert(0, '$INSTALL_DIR'); import main" 2>/dev/null; then
        log_error "main.py import still failing after reinstall. Check error details:"
        "$INSTALL_DIR/venv/bin/python" -c "import sys; sys.path.insert(0, '$INSTALL_DIR'); import main" 2>&1 | head -5
        exit 1
    fi
fi

log_success "All Python dependencies verified and working"

# Install launchd plist
log_info "Installing user service..."
mkdir -p "$HOME/Library/LaunchAgents"
sed "s|__INSTALL_DIR__|$INSTALL_DIR|g; s|__HOME__|$HOME|g" "$SCRIPT_DIR/com.henrybravo.mlx-router.plist" > "$HOME/Library/LaunchAgents/com.henrybravo.mlx-router.plist"
launchctl load "$HOME/Library/LaunchAgents/com.henrybravo.mlx-router.plist"
log_success "User service installed and started"

# Wait for service to start
log_info "Waiting for service to start..."
sleep 3

# Verify installation
log_info "Verifying installation..."
if launchctl list | grep -q com.henrybravo.mlx-router; then
    log_success "Service is running"
else
    log_error "Service failed to start - check logs at $INSTALL_DIR/logs/mlx_router.error.log"
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
echo "📊 Logs: $INSTALL_DIR/logs/mlx_router.log"
echo "🚫 Error logs: $INSTALL_DIR/logs/mlx_router.error.log"
echo ""
echo "🔧 Management commands:"
echo "   Stop:      launchctl unload ~/Library/LaunchAgents/com.henrybravo.mlx-router.plist"
echo "   Start:     launchctl load ~/Library/LaunchAgents/com.henrybravo.mlx-router.plist"
echo "   Uninstall: ./uninstall-launchd.sh"