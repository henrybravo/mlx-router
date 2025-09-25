#!/usr/bin/env python3
"""
Model configuration management for MLX Router
Supports loading models from local directories and HuggingFace Hub
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ModelConfig:
    """Model-specific configurations with support for local model directories"""

    # No more hard-coded models - configurations loaded dynamically
    MODELS = {}
    _model_directory = None

    @classmethod
    def set_model_directory(cls, model_directory: str):
        """Set the base directory for local model storage"""
        if model_directory:
            # Expand environment variables and user home
            expanded_path = os.path.expandvars(os.path.expanduser(model_directory))
            cls._model_directory = Path(expanded_path).resolve()
            logger.info(f"Model directory set to: {cls._model_directory}")
        else:
            cls._model_directory = None

    @classmethod
    def get_model_directory(cls) -> Optional[Path]:
        """Get the configured model directory"""
        return cls._model_directory

    @classmethod
    def resolve_model_path(cls, model_name: str) -> str:
        """Resolve model name to full path, checking local directory first"""
        # If it's already an absolute path, use it directly
        if os.path.isabs(model_name):
            return model_name

        # Check if model exists in configured model directory
        if cls._model_directory and cls._model_directory.exists():
            # First try direct directory name
            model_path = cls._model_directory / model_name
            if model_path.exists():
                # Look for the actual model files (could be in snapshots subdirectory)
                if (model_path / "snapshots").exists():
                    # HuggingFace-style structure
                    snapshots = list((model_path / "snapshots").iterdir())
                    if snapshots:
                        return str(snapshots[0])  # Use latest snapshot
                else:
                    # Direct model directory
                    return str(model_path)

            # Try HuggingFace cache directory naming convention
            hf_cache_name = f"models--{model_name.replace('/', '--')}"
            hf_model_path = cls._model_directory / hf_cache_name
            if hf_model_path.exists():
                # HuggingFace cache structure
                if (hf_model_path / "snapshots").exists():
                    snapshots = list((hf_model_path / "snapshots").iterdir())
                    if snapshots:
                        # Sort by modification time to get latest
                        latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                        return str(latest_snapshot)

        # Fall back to HuggingFace identifier
        return model_name

    @classmethod
    def discover_local_models(cls) -> Dict[str, Dict[str, Any]]:
        """Discover models in the configured local directory"""
        discovered_models = {}

        if not cls._model_directory or not cls._model_directory.exists():
            return discovered_models

        logger.info(f"Scanning model directory: {cls._model_directory}")

        for item in cls._model_directory.iterdir():
            if not item.is_dir():
                continue

            model_name = item.name
            actual_model_name = None

            # Check if this is a HuggingFace cache directory
            if model_name.startswith("models--") and "--" in model_name:
                # Convert from cache format: models--org--model -> org/model
                parts = model_name[8:].split("--")  # Remove "models--" prefix
                if len(parts) >= 2:
                    org = parts[0]
                    model_parts = parts[1:]
                    actual_model_name = f"{org}/{'-'.join(model_parts)}"
            else:
                # Direct directory name
                actual_model_name = model_name

            if actual_model_name:
                model_config = cls._extract_model_config(item)
                if model_config:
                    discovered_models[actual_model_name] = model_config
                    logger.debug(f"Discovered local model: {actual_model_name} (from {model_name})")

        return discovered_models

    @classmethod
    def _extract_model_config(cls, model_path: Path) -> Optional[Dict[str, Any]]:
        """Extract configuration from a local model directory"""
        config_file = None

        # Check for HuggingFace-style snapshots
        if (model_path / "snapshots").exists():
            snapshots = list((model_path / "snapshots").iterdir())
            if snapshots:
                snapshot_path = snapshots[0]  # Use latest
                config_file = snapshot_path / "config.json"
        else:
            # Direct model directory
            config_file = model_path / "config.json"

        if config_file and config_file.exists():
            try:
                import json
                with open(config_file, 'r') as f:
                    hf_config = json.load(f)

                # Extract relevant parameters
                config = {
                    "max_tokens": min(hf_config.get("max_position_embeddings", 4096), 4096),
                    "temp": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "min_p": 0.05,
                    "chat_template": cls._detect_chat_template(hf_config),
                    "required_memory_gb": cls._estimate_memory_requirement(hf_config),
                    "path": str(model_path)
                }

                # Add memory pressure settings
                config["memory_pressure_max_tokens"] = {
                    "normal": config["max_tokens"],
                    "moderate": min(config["max_tokens"] // 2, 2048),
                    "high": min(config["max_tokens"] // 4, 1024),
                    "critical": min(config["max_tokens"] // 8, 512)
                }

                return config

            except Exception as e:
                logger.warning(f"Failed to parse config for {model_path.name}: {e}")

        # Fallback: basic config for directories with model files
        if cls._has_model_files(model_path):
            return {
                "max_tokens": 4096,
                "temp": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "min_p": 0.05,
                "chat_template": "generic",
                "required_memory_gb": 8,
                "path": str(model_path),
                "memory_pressure_max_tokens": {
                    "normal": 4096,
                    "moderate": 2048,
                    "high": 1024,
                    "critical": 512
                }
            }

        return None

    @classmethod
    def _has_model_files(cls, model_path: Path) -> bool:
        """Check if directory contains model files"""
        model_extensions = {'.safetensors', '.bin', '.pth', '.pt'}
        return any(
            file_path.is_file() and file_path.suffix in model_extensions
            for file_path in model_path.rglob('*')
        )

    @classmethod
    def _detect_chat_template(cls, hf_config: Dict[str, Any]) -> str:
        """Detect chat template from HuggingFace config"""
        model_name = hf_config.get("_name_or_path", "").lower()

        # Check model name patterns
        if "llama-3" in model_name or "llama3" in model_name:
            return "llama3"
        elif "qwen" in model_name:
            return "qwen"
        elif "deepseek" in model_name:
            return "deepseek"
        elif "phi-4" in model_name or "phi4" in model_name:
            return "phi4"
        elif "chatml" in model_name:
            return "chatml"

        return "generic"

    @classmethod
    def _estimate_memory_requirement(cls, hf_config: Dict[str, Any]) -> int:
        """Estimate memory requirement based on model config"""
        model_name = hf_config.get("_name_or_path", "").lower()

        # Check for size indicators in model name
        if "70b" in model_name:
            return 45
        elif "30b" in model_name or "34b" in model_name:
            return 20
        elif "13b" in model_name or "14b" in model_name:
            return 10
        elif "7b" in model_name or "8b" in model_name:
            return 8
        elif "3b" in model_name or "4b" in model_name:
            return 4
        elif "1b" in model_name or "2b" in model_name:
            return 2

        # Default estimate
        return 8

    @classmethod
    def get_config(cls, model_name):
        """Get configuration for a model, checking local configs first"""
        # Check configured models first
        if model_name in cls.MODELS:
            return cls.MODELS[model_name]

        # Check if it's a local model that needs discovery
        if cls._model_directory:
            local_models = cls.discover_local_models()
            if model_name in local_models:
                return local_models[model_name]

        # Fallback to defaults
        return {
            "max_tokens": 4096, "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
            "chat_template": "generic", "required_memory_gb": 8
        }

    @classmethod
    def get_available_models(cls):
        """Get all available models from config and local discovery"""
        available = list(cls.MODELS.keys())

        # Add locally discovered models
        if cls._model_directory:
            local_models = cls.discover_local_models()
            available.extend(local_models.keys())

        return list(set(available))  # Remove duplicates

    @classmethod
    def get_models_by_memory_requirement(cls, max_memory_gb=None):
        """Get models filtered by memory requirement"""
        if max_memory_gb is None:
            return cls.get_available_models()

        suitable_models = []
        for model_name in cls.get_available_models():
            config = cls.get_config(model_name)
            if config.get("required_memory_gb", 8) <= max_memory_gb:
                suitable_models.append((model_name, config.get("required_memory_gb", 8)))

        suitable_models.sort(key=lambda x: x[1])
        return [model[0] for model in suitable_models]

    @classmethod
    def suggest_best_model_for_memory(cls, available_memory_gb, prefer_performance=True):
        """Suggest best model for available memory"""
        suitable_models = cls.get_models_by_memory_requirement(available_memory_gb * 0.8)
        if not suitable_models:
            return None
        return suitable_models[-1] if prefer_performance else suitable_models[0]

    @classmethod
    def get_chat_template(cls, model_name):
        """Get chat template for a model"""
        return cls.get_config(model_name).get("chat_template", "generic")

    @classmethod
    def load_from_dict(cls, config_dict):
        """Load model configurations from a dictionary (typically from config.json)"""
        if isinstance(config_dict, dict) and config_dict:
            # Replace the entire MODELS dictionary with the config file contents
            cls.MODELS = config_dict.copy()
            logger.info(f"Loaded ModelConfig with {len(config_dict)} model configurations from config file")
        elif config_dict is not None and isinstance(config_dict, dict):
            # Empty dict provided - clear models
            cls.MODELS = {}
            logger.warning("Empty model configuration provided, clearing models")
        else:
            # None or invalid config - keep current models
            logger.warning("Invalid model configuration provided, keeping current models")