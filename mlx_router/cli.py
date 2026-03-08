#!/usr/bin/env python3
"""
CLI entry point for mlx-router.

This module is the installed package entry point (`mlx-router` command).
The root `main.py` is a thin wrapper around this for development convenience.
"""

import argparse
import atexit
import json
import logging
import os
import signal
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import mlx.core as mx
import pyfiglet
import uvicorn

from mlx_router.__version__ import AUTHOR, RELEASE_DATE, VERSION
from mlx_router.api.app import app, set_model_manager
from mlx_router.config.model_config import ModelConfig
from mlx_router.core.manager import MLXModelManager
from mlx_router.core.resource_monitor import ResourceMonitor

# ---------------------------------------------------------------------------
# Log filters
# ---------------------------------------------------------------------------


class _InfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


class _ErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


def _setup_logging(debug: bool = False, log_dir: Path | None = None) -> None:
    """Configure root logger. Called once at startup inside main()."""
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    root = logging.getLogger()
    root.setLevel(level)

    # stdout — INFO/DEBUG only (when running in a terminal)
    if sys.stdout.isatty():
        stdout_h = logging.StreamHandler(sys.stdout)
        stdout_h.setLevel(logging.DEBUG)
        stdout_h.setFormatter(formatter)
        stdout_h.addFilter(_InfoFilter())
        root.addHandler(stdout_h)

    # stderr — WARNING and above
    stderr_h = logging.StreamHandler(sys.stderr)
    stderr_h.setLevel(logging.WARNING)
    stderr_h.setFormatter(formatter)
    stderr_h.addFilter(_ErrorFilter())
    root.addHandler(stderr_h)

    # Rotating file handler — only when log dir exists / can be created
    resolved_log_dir = log_dir or Path("logs")
    try:
        resolved_log_dir.mkdir(parents=True, exist_ok=True)
        file_h = RotatingFileHandler(
            resolved_log_dir / "mlx_router.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
        )
        file_h.setLevel(level)
        file_h.setFormatter(formatter)
        root.addHandler(file_h)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not create log file in %s — logging to console only.", resolved_log_dir
        )


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def print_banner() -> None:
    """Print the MLX Router ASCII-art banner."""
    banner = pyfiglet.figlet_format("mlx-router", font="slant")
    print("\033[1;36m" + banner + "\033[0m")
    print("\033[1;35mPowered by FastAPI\033[0m")
    print(f"\033[1;33mVersion {VERSION} ({RELEASE_DATE})\033[0m")
    print(f"\033[1;32m{AUTHOR}\033[0m\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    class _BannerFormatter(argparse.RawDescriptionHelpFormatter):
        def __init__(self, prog: str, **kwargs: object) -> None:
            try:
                import shutil

                width = min(shutil.get_terminal_size().columns, 120)
            except Exception:
                width = 100
            super().__init__(prog, max_help_position=40, width=width)  # type: ignore[arg-type]

        def format_help(self) -> str:
            print_banner()
            return super().format_help()

    parser = argparse.ArgumentParser(
        description=(
            "MLX Model Router — OpenAI-compatible inference server for Apple Silicon.\n"
            "Hot-swap MLX models, stream responses, and process images/PDFs."
        ),
        formatter_class=_BannerFormatter,
        epilog=(
            "Examples:\n"
            "  mlx-router                              # start with defaults\n"
            "  mlx-router --config config.json         # load model config\n"
            "  mlx-router --ip 127.0.0.1 --port 8080  # custom bind\n"
            "  mlx-router --debug                      # verbose logging\n"
        ),
    )
    parser.add_argument("-v", "--version", action="store_true", help="Print version and exit")
    parser.add_argument("--ip", default="0.0.0.0", help="Bind IP (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8800, help="Port (default: 8800)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens (default: 4096)")
    parser.add_argument("--timeout", type=int, default=120, help="Generation timeout in seconds (default: 120)")
    parser.add_argument("--config", help="Path to config.json")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """mlx-router entry point — invoked by the `mlx-router` CLI command."""
    args = _parse_args()

    if args.version:
        print_banner()
        return

    _setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)

    # MLX device setup
    try:
        mx.set_default_device(mx.gpu)
        logger.info("MLX default device set to GPU.")
    except Exception as exc:
        logger.warning("Could not set MLX device to GPU: %s — using MLX default.", exc)

    # Config loading
    config_data: dict = {}
    if args.config:
        try:
            with open(args.config) as f:
                config_data = json.load(f)
            logger.info("Loaded configuration from %s", args.config)

            if not isinstance(config_data, dict):
                raise ValueError("Config must be a JSON object")

            models_cfg = config_data.get("models", {})
            if models_cfg and not isinstance(models_cfg, dict):
                raise ValueError("'models' must be a dictionary")

            defaults = config_data.get("defaults", {})
            server_cfg = config_data.get("server", {})

            if defaults and not isinstance(defaults, dict):
                raise ValueError("'defaults' must be a dictionary")
            if server_cfg and not isinstance(server_cfg, dict):
                raise ValueError("'server' must be a dictionary")

            # Model directory
            model_dir = defaults.get("model_directory")
            model_dir = os.environ.get("MLX_MODEL_DIR", model_dir)
            if model_dir:
                ModelConfig.set_model_directory(model_dir)
                logger.info("Model directory: %s", model_dir)
            elif os.environ.get("MLX_MODEL_DIR"):
                ModelConfig.set_model_directory(os.environ["MLX_MODEL_DIR"])
                logger.info("Model directory (env): %s", os.environ["MLX_MODEL_DIR"])

            # Memory thresholds
            mem_threshold = defaults.get("memory_threshold_gb")
            swap_critical = defaults.get("swap_critical_percent", 90.0)
            swap_high = defaults.get("swap_high_percent", 75.0)
            if mem_threshold is not None:
                ResourceMonitor.set_memory_threshold_gb(mem_threshold)
            ResourceMonitor.set_swap_thresholds(swap_critical, swap_high)

            safety_margin = defaults.get("safety_margin", 1.2)
            ResourceMonitor.set_safety_margin(safety_margin)

            ModelConfig.load_from_dict(models_cfg)

            args.max_tokens = defaults.get("max_tokens", args.max_tokens)
            args.timeout = defaults.get("timeout", args.timeout)
            if args.ip == "0.0.0.0":
                args.ip = server_cfg.get("ip", args.ip)
            if args.port == 8800:
                args.port = server_cfg.get("port", args.port)
            if not args.debug:
                args.debug = server_cfg.get("debug", args.debug)

        except FileNotFoundError:
            logger.error("Config file not found: %s", args.config)
            sys.exit(1)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in %s: %s", args.config, exc)
            sys.exit(1)
        except ValueError as exc:
            logger.error("Invalid config structure: %s", exc)
            sys.exit(1)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(
        "Starting MLX Router v%s (%s) by %s — https://github.com/henrybravo/mlx-router",
        VERSION,
        RELEASE_DATE,
        AUTHOR,
    )

    model_manager = MLXModelManager(max_tokens=args.max_tokens, timeout=args.timeout)

    def _shutdown(signum: object = None, frame: object = None) -> None:
        logger.info("Shutting down MLX Router…")
        try:
            model_manager.shutdown()
        except Exception as exc:
            logger.error("Error during shutdown: %s", exc)
        finally:
            sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    atexit.register(_shutdown)

    if args.config and (config_data.get("models") or ModelConfig.get_model_directory()):
        model_manager.refresh_available_models()

    set_model_manager(model_manager)

    from mlx_router.api.app import set_global_config

    set_global_config(config_data)

    # Preload default model
    default_model = config_data.get("default_model") or config_data.get("defaults", {}).get("model")
    if default_model:
        logger.info("Preloading default model: %s", default_model)
        try:
            model_manager.load_model(default_model)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Failed to preload default model: %s", exc)
    elif model_manager.available_models:
        logger.info(
            "Discovered %s available models. No default model configured; waiting for explicit selection.",
            len(model_manager.available_models),
        )

    print_banner()
    mem = ResourceMonitor.get_memory_info()
    print(f"🚀 MLX Router running on http://{args.ip}:{args.port}")
    print(f"📄 Interactive docs: http://{args.ip}:{args.port}/docs")
    print(f"💾 Memory: {mem['total_gb']:.1f} GB total, {mem['available_gb']:.1f} GB available")

    if model_manager.available_models:
        print("📋 Available models:")
        for m in model_manager.available_models:
            print(f"  - {m}")
    else:
        logger.warning("No models loaded or configured.")

    uvicorn.run(app, host=args.ip, port=args.port)
