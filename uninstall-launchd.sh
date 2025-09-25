#!/bin/bash
# MLX Router Uninstallation Script for macOS

set -e

# Configurable install directory (defaults to $HOME/mlx_router_app)
INSTALL_DIR="${INSTALL_DIR:-$HOME/mlx_router_app}"

echo "Uninstalling MLX Router..."

# Stop and unload the service
if launchctl list | grep -q com.henrybravo.mlx-router; then
    echo "Stopping MLX Router service..."
    launchctl unload "$HOME/Library/LaunchAgents/com.henrybravo.mlx-router.plist"
fi

# Remove launchd plist
if [ -f "$HOME/Library/LaunchAgents/com.henrybravo.mlx-router.plist" ]; then
    echo "Removing launchd configuration..."
    rm "$HOME/Library/LaunchAgents/com.henrybravo.mlx-router.plist"
fi

# Remove installation directory
if [ -d "$INSTALL_DIR" ]; then
    echo "Removing installation directory..."
    rm -rf "$INSTALL_DIR"
fi

# Remove log files (optional - ask user)
if [ -f "$INSTALL_DIR/logs/mlx_router.log" ] || [ -f "$INSTALL_DIR/logs/mlx_router.error.log" ]; then
    read -p "Remove log files? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing log files..."
        rm -f "$INSTALL_DIR/logs/mlx_router.log"
        rm -f "$INSTALL_DIR/logs/mlx_router.error.log"
    fi
fi

echo "MLX Router has been uninstalled!"