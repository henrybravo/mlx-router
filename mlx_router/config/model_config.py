#!/usr/bin/env python3
"""
Model configuration management for MLX Router
"""

import logging

logger = logging.getLogger(__name__)

class ModelConfig:
    """Model-specific configurations optimized for Apple Silicon"""
    MODELS = {
        "mlx-community/Qwen3-30B-A3B-8bit": {
            "max_tokens": 8192, "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
            "chat_template": "qwen", "required_memory_gb": 10
        },
        "mlx-community/Llama-3.3-70B-Instruct-8bit": { # max_tokens can be defined here if needed
            "max_tokens": 4096, "temp": 0.6, "top_p": 0.9, "top_k": 50, "min_p": 0.05, # Default 4096, can be less
            "chat_template": "llama3", "required_memory_gb": 12
        },
        "mlx-community/Llama-3.2-3B-Instruct-4bit": {
            "max_tokens": 4096, "temp": 0.7, "top_p": 0.95, "top_k": 50, "min_p": 0.03,
            "chat_template": "llama3", "required_memory_gb": 4
        },
        "mlx-community/DeepSeek-R1-0528-Qwen3-8B-8bit": {
            "max_tokens": 8192, "temp": 0.6, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
            "chat_template": "deepseek", "required_memory_gb": 8
        },
        "mlx-community/DeepSeek-R1-0528-Qwen3-8B-bf16": {
            "max_tokens": 8192, "temp": 0.6, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
            "chat_template": "deepseek", "required_memory_gb": 10
        },
        "deepseek-ai/deepseek-coder-6.7b-instruct": {
            "max_tokens": 4096, "temp": 0.1, "top_p": 0.95, "top_k": 20, "min_p": 0.1,
            "chat_template": "deepseek", "required_memory_gb": 6
        },
        "mlx-community/Phi-4-reasoning-plus-6bit": {
            "max_tokens": 4096, "temp": 0.3, "top_p": 0.9, "top_k": 25, "min_p": 0.08,
            "chat_template": "phi4", "required_memory_gb": 6
        }
    }

    @classmethod
    def get_config(cls, model_name):
        return cls.MODELS.get(model_name, {
            "max_tokens": 4096, "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
            "chat_template": "generic", "required_memory_gb": 8
        })

    @classmethod
    def get_available_models(cls):
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_models_by_memory_requirement(cls, max_memory_gb=None):
        if max_memory_gb is None: return cls.get_available_models()
        suitable_models = [(name, config.get("required_memory_gb", 8)) 
                           for name, config in cls.MODELS.items() 
                           if config.get("required_memory_gb", 8) <= max_memory_gb]
        suitable_models.sort(key=lambda x: x[1])
        return [model[0] for model in suitable_models]
    
    @classmethod
    def suggest_best_model_for_memory(cls, available_memory_gb, prefer_performance=True):
        suitable_models = cls.get_models_by_memory_requirement(available_memory_gb * 0.8)
        if not suitable_models: return None
        return suitable_models[-1] if prefer_performance else suitable_models[0]

    @classmethod
    def get_chat_template(cls, model_name):
        return cls.get_config(model_name).get("chat_template", "generic")
    
    @classmethod
    def load_from_dict(cls, config_dict):
        """Load model configurations from a dictionary (typically from config.json)"""
        if isinstance(config_dict, dict) and config_dict:
            # Replace the entire MODELS dictionary with the config file contents
            cls.MODELS = config_dict.copy()
            logger.info(f"Loaded ModelConfig with {len(config_dict)} model configurations from config file")
        elif not config_dict:
            logger.warning("Empty model configuration provided, keeping default models")