#!/usr/bin/env python3
"""
Production MLX Model Router
Optimized for Apple Silicon with robust error handling and resource management
Now powered by FastAPI for a robust, modern API.
"""

import argparse
import pyfiglet
import json
import logging
from logging.handlers import RotatingFileHandler

import mlx.core as mx
import uvicorn

from mlx_router.core.model_manager import MLXModelManager
from mlx_router.core.resource_monitor import ResourceMonitor
from mlx_router.config.model_config import ModelConfig
from mlx_router.api.app import app, set_model_manager

# Version information
VERSION = "2.0.0"
RELEASE_DATE = "20250607"
AUTHOR = "Henry Bravo - info@henrybravo.nl"

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler('logs/mlx_router.log', maxBytes=10*1024*1024, backupCount=3)
    ]
)
logger = logging.getLogger(__name__)

def log_and_print(message, level="info"):
    print(message)
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)
    elif level == "critical":
        logger.critical(message)
    elif level == "fatal":
        logger.fatal(message)
    elif level == "trace":
        logger.trace(message)

def print_banner():
    """Print the MLX Router banner with ASCII art"""
    banner = pyfiglet.figlet_format("mlx-router", font="slant")
    print("\033[1;36m" + banner + "\033[0m")
    print("\033[1;35mPowered by FastAPI\033[0m")
    print(f"\033[1;33mVersion {VERSION} ({RELEASE_DATE})\033[0m")
    print(f"\033[1;32m{AUTHOR}\033[0m\n")

def parse_args():
    parser = argparse.ArgumentParser(description="MLX Model Router")
    parser.add_argument("-v", "--version", action="store_true", help="Display version and exit")
    parser.add_argument("--ip", default="0.0.0.0", help="IP address (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8800, help="Port (default: 8800)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens (default: 4096)")
    parser.add_argument("--timeout", type=int, default=120, help="Generation timeout (default: 120s)")
    parser.add_argument("--config", help="Config file path (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.version:
        print_banner()
        return
    
    logger.info(f"Starting MLX Router v{VERSION} (Release Date: {RELEASE_DATE}) by {AUTHOR} (https://github.com/henrybravo/mlx-router)")

    try:
        mx.set_default_device(mx.gpu)
        logger.info("MLX default device set to GPU.")
    except Exception as e:
        log_and_print(f"Could not set MLX default device to GPU: {e}. MLX will use its default.", level="warning")
    
    config_data = {}
    if args.config:
        try:
            with open(args.config, 'r') as f: config_data = json.load(f)
            logger.info(f"Loaded configuration from {args.config}")
            ModelConfig.load_from_dict(config_data.get('models', {}))
            defaults = config_data.get('defaults', {})
            args.max_tokens = defaults.get('max_tokens', args.max_tokens)
            args.timeout = defaults.get('timeout', args.timeout)
        except Exception as e:
            log_and_print(f"Failed to load config file {args.config}: {e}", level="error")
            exit(1)
    
    model_manager = MLXModelManager(max_tokens=args.max_tokens, timeout=args.timeout)
    set_model_manager(model_manager)
    
    default_model_to_preload = config_data.get('default_model')
    if not default_model_to_preload and model_manager.available_models:
        default_model_to_preload = model_manager.available_models[0]
    
    if default_model_to_preload:
        logger.info(f"Attempting to preload default model: {default_model_to_preload}")
        try: model_manager.load_model(default_model_to_preload)
        except (ValueError, RuntimeError) as e: log_and_print(f"Failed to preload default model: {e}", level="warning")
    
    print_banner()
    mem_info = ResourceMonitor.get_memory_info()
    print(f"🚀 MLX Router running on http://{args.ip}:{args.port}")
    print(f"📄 View interactive API docs at http://{args.ip}:{args.port}/docs")
    print(f"💾 System Memory: Total={mem_info['total_gb']:.1f}GB, Available={mem_info['available_gb']:.1f}GB")
    
    if model_manager.available_models:
        print("📋 Available models:")
        for model_id in model_manager.available_models: print(f"  - {model_id}")
    else:
        log_and_print("⚠️ No models loaded or configured.", level="warning")
        
    uvicorn.run(app, host=args.ip, port=args.port)

if __name__ == "__main__":
    main()