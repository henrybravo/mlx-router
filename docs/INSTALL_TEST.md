# MLX Router Installation Testing Guide

## Pre-Installation Checks

1. **Set installation directory:**
```bash
INSTALL_DIR="${INSTALL_DIR:-$HOME/mlx_router_app}"
echo "Installing to: $INSTALL_DIR"
```

2. **Verify all files are present:**
```bash
ls -la main.py config.json com.henrybravo.mlx-router.plist mlx_router/ requirements.txt
```

3. **Check for existing installations:**
```bash
launchctl list | grep mlx-router
ls -la ~/Library/LaunchAgents/com.henrybravo.mlx-router.plist
```

## Installation Process

1. **Run the installation script:**
```bash
./install-launchd.sh
```

2. **Verify installation directories:**
```bash
ls -la "$INSTALL_DIR/"
ls -la "$INSTALL_DIR/logs/"
```

3. **Check service status:**
```bash
launchctl list | grep mlx-router
```

## Testing the Service

1. **Wait for startup (30 seconds), then test endpoints:**
```bash
# Health check
curl -s http://localhost:8800/health | jq

# Models endpoint
curl -s http://localhost:8800/v1/models | jq

# FastAPI docs
open http://localhost:8800/docs
```

2. **Check log files:**
```bash
tail -f "$INSTALL_DIR/logs/mlx-router.log"
tail -f "$INSTALL_DIR/logs/mlx-router.error.log"
```

3. **Test chat completion:**
```bash
curl -s -X POST http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq
```

## Service Management

1. **Stop service:**
```bash
launchctl unload ~/Library/LaunchAgents/com.henrybravo.mlx-router.plist
```

2. **Start service:**
```bash
launchctl load ~/Library/LaunchAgents/com.henrybravo.mlx-router.plist
```

3. **Restart service:**
```bash
launchctl unload ~/Library/LaunchAgents/com.henrybravo.mlx-router.plist
launchctl load ~/Library/LaunchAgents/com.henrybravo.mlx-router.plist
```

## Troubleshooting

1. **Check service status:**
```bash
launchctl print gui/$(id -u)/com.henrybravo.mlx-router
```

2. **View recent logs:**
```bash
tail -50 "$INSTALL_DIR/logs/mlx-router.log"
tail -50 "$INSTALL_DIR/logs/mlx-router.error.log"
```

3. **Test Python environment:**
```bash
"$INSTALL_DIR/venv/bin/python" --version
"$INSTALL_DIR/venv/bin/pip" list | grep mlx
```

4. **Manual service start (for debugging):**
```bash
"$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/main.py" --config "$INSTALL_DIR/config.json"
```

## Uninstallation

```bash
./uninstall-launchd.sh
```

## Expected Behavior

- ✅ Service loads successfully
- ✅ API responds on http://localhost:8800
- ✅ Health endpoint returns status
- ✅ Models endpoint lists available models
- ✅ Chat completions work with available models
- ✅ Logs are written to $INSTALL_DIR/logs/
- ✅ Service restarts automatically if it crashes