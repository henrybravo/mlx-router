# MLX Router Installation Testing Guide

## Pre-Installation Checks

1. **Verify all files are present:**
```bash
ls -la main.py config.json com.henrybravo.mlx-router.plist mlx_router/ requirements.txt
```

2. **Check for existing installations:**
```bash
sudo launchctl list | grep mlx-router
ls -la /Library/LaunchDaemons/com.henrybravo.mlx-router.plist
```

## Installation Process

1. **Run the installation script:**
```bash
./install-launchd.sh
```

2. **Verify installation directories:**
```bash
ls -la /usr/local/opt/mlx-router/
ls -la /usr/local/etc/mlx-router/
ls -la /usr/local/opt/mlx-router/logs/
```

3. **Check service status:**
```bash
sudo launchctl list | grep mlx-router
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
tail -f /usr/local/opt/mlx-router/logs/mlx-router.log
tail -f /usr/local/opt/mlx-router/logs/mlx-router.error.log
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
sudo launchctl unload /Library/LaunchDaemons/com.henrybravo.mlx-router.plist
```

2. **Start service:**
```bash
sudo launchctl load /Library/LaunchDaemons/com.henrybravo.mlx-router.plist
```

3. **Restart service:**
```bash
sudo launchctl unload /Library/LaunchDaemons/com.henrybravo.mlx-router.plist
sudo launchctl load /Library/LaunchDaemons/com.henrybravo.mlx-router.plist
```

## Troubleshooting

1. **Check service status:**
```bash
sudo launchctl print system/com.henrybravo.mlx-router
```

2. **View recent logs:**
```bash
tail -50 /usr/local/opt/mlx-router/logs/mlx-router.log
tail -50 /usr/local/opt/mlx-router/logs/mlx-router.error.log
```

3. **Test Python environment:**
```bash
sudo /usr/local/opt/mlx-router/venv/bin/python --version
sudo /usr/local/opt/mlx-router/venv/bin/pip list | grep mlx
```

4. **Manual service start (for debugging):**
```bash
/usr/local/opt/mlx-router/venv/bin/python /usr/local/opt/mlx-router/main.py --config /usr/local/etc/mlx-router/config.json
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
- ✅ Logs are written to /usr/local/var/log/
- ✅ Service restarts automatically if it crashes