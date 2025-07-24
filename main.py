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
import sys
from logging.handlers import RotatingFileHandler

import mlx.core as mx
import uvicorn

from mlx_router.core.model_manager import MLXModelManager
from mlx_router.core.resource_monitor import ResourceMonitor
from mlx_router.config.model_config import ModelConfig
from mlx_router.api.app import app, set_model_manager

from mlx_router.__version__ import VERSION, RELEASE_DATE, AUTHOR

# Setup structured logging with separate handlers for stdout and stderr
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.WARNING

class ErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.WARNING

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create stdout handler for INFO and DEBUG messages
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
stdout_handler.addFilter(InfoFilter())

# Create stderr handler for WARNING, ERROR, and CRITICAL messages
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)
stderr_handler.setFormatter(formatter)
stderr_handler.addFilter(ErrorFilter())

# Create file handler for all messages
file_handler = RotatingFileHandler('logs/mlx_router.log', maxBytes=10*1024*1024, backupCount=3)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Add handlers to root logger
root_logger.addHandler(stdout_handler)
root_logger.addHandler(stderr_handler)
root_logger.addHandler(file_handler)

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
    class BannerHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def __init__(self, prog, indent_increment=2, max_help_position=40, width=None):
            # Get terminal width or use a reasonable default
            if width is None:
                try:
                    import shutil
                    width = min(shutil.get_terminal_size().columns, 120)
                except:
                    width = 100
            super().__init__(prog, indent_increment, max_help_position, width)
        
        def format_help(self):
            print_banner()
            return super().format_help()
        
    parser = argparse.ArgumentParser(
            description="""
MLX Model Router - A powerful server for managing and serving multiple MLX models.
This server provides a unified FastAPI interface for various MLX models, allowing hot-swapping
between different models without restarting the server.
            """,
            formatter_class=BannerHelpFormatter,
            epilog="""
Examples:
  # Start server with default settings
  python main.py

  # Start server on specific IP and port
  python main.py --ip 127.0.0.1 -p 8080

  # Start server with config file and specific port
  python main.py --config config.json --port 8888
  
  # Start server with debug logging enabled
  python main.py --debug
            """
    )

    parser.add_argument("-v", "--version", action="store_true", help="Display version and exit")
    parser.add_argument("--ip", default="0.0.0.0", help="IP address (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8800, help="Port (default: 8800)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens (default: 4096)")
    parser.add_argument("--timeout", type=int, default=120, help="Generation timeout (default: 120s)")
    parser.add_argument("--config", help="Config file path (optional)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.version:
        print_banner()
        return

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
            server_config = config_data.get('server', {})
            
            # Apply config values only if CLI args weren't explicitly provided
            args.max_tokens = defaults.get('max_tokens', args.max_tokens)
            args.timeout = defaults.get('timeout', args.timeout)
            
            # For server config, use config file values as defaults unless overridden by CLI
            if args.ip == "0.0.0.0":  # Default value, not explicitly set
                args.ip = server_config.get('ip', args.ip)
            if args.port == 8800:  # Default value, not explicitly set
                args.port = server_config.get('port', args.port)
            if not args.debug:  # Debug flag not set via CLI
                args.debug = server_config.get('debug', args.debug)
        except Exception as e:
            log_and_print(f"Failed to load config file {args.config}: {e}", level="error")
            exit(1)
    
    # Set logging level based on debug flag (after config is loaded)
    if args.debug:
        root_logger.setLevel(logging.DEBUG)
        stdout_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
    
    logger.info(f"Starting MLX Router v{VERSION} (Release Date: {RELEASE_DATE}) by {AUTHOR} (https://github.com/henrybravo/mlx-router)")
    
    model_manager = MLXModelManager(max_tokens=args.max_tokens, timeout=args.timeout)
    
    # Refresh available models after config is loaded to ensure we use config-based model definitions
    if args.config and config_data.get('models'):
        model_manager.refresh_available_models()
    
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