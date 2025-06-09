#!/bin/bash
# MLX Router Uninstallation Script for macOS

set -e

echo "Uninstalling MLX Router..."

# Stop and unload the service
if sudo launchctl list | grep -q com.henrybravo.mlx-router; then
    echo "Stopping MLX Router service..."
    sudo launchctl unload /Library/LaunchDaemons/com.henrybravo.mlx-router.plist
fi

# Remove launchd plist
if [ -f "/Library/LaunchDaemons/com.henrybravo.mlx-router.plist" ]; then
    echo "Removing launchd configuration..."
    sudo rm /Library/LaunchDaemons/com.henrybravo.mlx-router.plist
fi

# Remove installation directory
if [ -d "/usr/local/opt/mlx-router" ]; then
    echo "Removing installation directory..."
    sudo rm -rf /usr/local/opt/mlx-router
fi

# Remove configuration directory
if [ -d "/usr/local/etc/mlx-router" ]; then
    echo "Removing configuration directory..."
    sudo rm -rf /usr/local/etc/mlx-router
fi

# Remove log files (optional - ask user)
if [ -f "/usr/local/var/log/mlx-router.log" ] || [ -f "/usr/local/var/log/mlx-router.error.log" ]; then
    read -p "Remove log files? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing log files..."
        sudo rm -f /usr/local/var/log/mlx-router.log
        sudo rm -f /usr/local/var/log/mlx-router.error.log
    fi
fi

echo "MLX Router has been uninstalled!"