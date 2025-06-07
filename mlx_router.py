#!/usr/bin/env python3
"""
Production MLX Model Router
Optimized for Apple Silicon with robust error handling and resource management
"""

import argparse
import pyfiglet
import gc
import json
import logging
import time
import threading # for model_lock, though ThreadPoolExecutor also uses threading
import psutil
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from http.server import HTTPServer, BaseHTTPRequestHandler
from logging.handlers import RotatingFileHandler
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler # for streaming
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer

# Version information
VERSION = "1.0.0"
RELEASE_DATE = "20250606"
AUTHOR = "Henry Bravo - info@henrybravo.nl"

# Setup structured logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler('mlx_router.log', maxBytes=10*1024*1024, backupCount=3)
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print the MLX Router banner with ASCII art"""
    banner = pyfiglet.figlet_format("mlx-router", font="slant")
    print("\033[1;36m" + banner + "\033[0m")
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
This server provides a unified API interface for various MLX models, allowing hot-swapping
between different models without restarting the server.
            """,
            formatter_class=BannerHelpFormatter,
            epilog="""
Examples:
  # Start server with default settings
  python mlx_router.py

  # Start server on specific IP and port
  python mlx_router.py --ip 127.0.0.1 -p 8080

  # Start server with config file and specific port
  python mlx_router.py --config config.json --port 8888
            """
    )
    
    parser.add_argument("-v", "--version", action="store_true", help="Display version and exit")
    parser.add_argument("--ip", default="0.0.0.0", help="IP address (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=8800, help="Port (default: 8800)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens (default: 4096)")
    parser.add_argument("--timeout", type=int, default=120, help="Generation timeout (default: 120s)")
    parser.add_argument("--config", help="Config file path (optional)")
    
    return parser.parse_args()

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
        config = cls.MODELS.get(model_name, {
            "max_tokens": 4096, "temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05,
            "chat_template": "generic", "required_memory_gb": 8
        })
        return config

    @classmethod
    def get_available_models(cls):
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_models_by_memory_requirement(cls, max_memory_gb=None):
        """Get models that fit within memory constraints"""
        if max_memory_gb is None:
            return cls.get_available_models()
        
        suitable_models = []
        for model_name, config in cls.MODELS.items():
            required_memory = config.get("required_memory_gb", 8)
            if required_memory <= max_memory_gb:
                suitable_models.append((model_name, required_memory))
        
        # Sort by memory requirement (ascending)
        suitable_models.sort(key=lambda x: x[1])
        return [model[0] for model in suitable_models]
    
    @classmethod
    def suggest_best_model_for_memory(cls, available_memory_gb, prefer_performance=True):
        """Suggest the best model that fits in available memory"""
        suitable_models = cls.get_models_by_memory_requirement(available_memory_gb * 0.8)  # 20% safety margin
        
        if not suitable_models:
            return None
        
        if prefer_performance:
            # Return the largest model that fits (typically better performance)
            return suitable_models[-1]
        else:
            # Return the smallest model (most memory efficient)
            return suitable_models[0]

    @classmethod
    def get_chat_template(cls, model_name):
        config = cls.get_config(model_name)
        return config.get("chat_template", "generic")

class ResourceMonitor:
    """Monitor system resources for Apple Silicon optimization"""
    
    _last_memory_check = 0
    _cached_memory_info = None
    _memory_cache_duration = 2.0  # Cache memory info for 2 seconds to reduce overhead
    
    @staticmethod
    def get_memory_info(use_cache=True):
        """Get current memory usage with optional caching"""
        current_time = time.time()
        
        # Use cached result if within cache duration
        if (use_cache and ResourceMonitor._cached_memory_info and 
            current_time - ResourceMonitor._last_memory_check < ResourceMonitor._memory_cache_duration):
            return ResourceMonitor._cached_memory_info
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "used_percent": memory.percent,
            "free_gb": memory.free / (1024**3),
            "buffers_gb": getattr(memory, 'buffers', 0) / (1024**3),
            "cached_gb": getattr(memory, 'cached', 0) / (1024**3),
            "swap_total_gb": swap.total / (1024**3),
            "swap_used_gb": swap.used / (1024**3),
            "swap_percent": swap.percent,
            "fragmentation_score": ResourceMonitor._calculate_fragmentation_score(memory)
        }
        
        # Update cache
        ResourceMonitor._cached_memory_info = memory_info
        ResourceMonitor._last_memory_check = current_time
        
        return memory_info
    
    @staticmethod
    def _calculate_fragmentation_score(memory):
        """Calculate a simple fragmentation score (0-100, lower is better)"""
        # Simple heuristic: high fragmentation when available memory is much less than free memory
        if hasattr(memory, 'free') and memory.available > 0:
            # If available is much less than free, suggests fragmentation
            fragmentation_ratio = 1.0 - (memory.available / max(memory.free + getattr(memory, 'buffers', 0) + getattr(memory, 'cached', 0), memory.available))
            return min(100, max(0, fragmentation_ratio * 100))
        return 0  # Unable to calculate
    
    @staticmethod
    def check_memory_available(model_name, safety_margin=1.5):
        """Check if sufficient memory is available for the specified model"""
        required_gb = ModelConfig.get_config(model_name).get("required_memory_gb", 8)
        info = ResourceMonitor.get_memory_info()
        
        # Apply safety margin and consider fragmentation
        effective_required = required_gb * safety_margin
        
        # Reduce fragmentation penalty if we have abundant memory
        fragmentation_penalty = 1.0
        if info["fragmentation_score"] > 50:
            if info["available_gb"] > required_gb * 4:  # Abundant memory
                fragmentation_penalty = 1.1  # Minimal penalty
            elif info["available_gb"] > required_gb * 2:  # Sufficient memory
                fragmentation_penalty = 1.15  # Small penalty
            else:
                fragmentation_penalty = 1.2  # Standard penalty
            logger.debug(f"Memory fragmentation detected ({info['fragmentation_score']:.1f}), applying {fragmentation_penalty}x penalty")
        
        effective_required *= fragmentation_penalty
        
        return info["available_gb"] >= effective_required
    
    @staticmethod
    def get_memory_pressure():
        """Get memory pressure level for Apple Silicon optimization"""
        info = ResourceMonitor.get_memory_info()
        if info["used_percent"] > 90:
            return "critical"
        elif info["used_percent"] > 80:
            return "high"
        elif info["used_percent"] > 70:
            return "moderate"
        return "normal"
    
    @staticmethod
    def get_memory_pressure_max_tokens(model_name, pressure_level):
        """Get max tokens for current memory pressure level"""
        model_config = ModelConfig.get_config(model_name)
        pressure_tokens = model_config.get("memory_pressure_max_tokens", {})
        
        if pressure_level in pressure_tokens:
            return pressure_tokens[pressure_level]
        
        # Fallback to standard max_tokens if no pressure-specific config
        return model_config.get("max_tokens", 4096)
    
    @staticmethod
    def should_defer_model_load(model_name):
        """Check if model loading should be deferred due to memory pressure"""
        pressure = ResourceMonitor.get_memory_pressure()
        info = ResourceMonitor.get_memory_info(use_cache=False)  # Fresh info for loading decisions
        required_gb = ModelConfig.get_config(model_name).get("required_memory_gb", 8)
        
        # Dynamic swap threshold based on available memory
        # If we have abundant RAM (>50GB available), be more lenient with swap usage
        swap_threshold = 95 if info["available_gb"] > 50 else 90 if info["available_gb"] > 20 else 50
        
        # Consider swap usage as additional pressure indicator, but with dynamic thresholds
        if info["swap_percent"] > swap_threshold:
            return True, f"High swap usage ({info['swap_percent']:.1f}% > {swap_threshold}%) indicates memory pressure"
        
        # Bypass memory pressure checks if we have abundant available RAM for the model
        if info["available_gb"] > required_gb * 3:  # 3x safety margin means we can bypass some restrictions
            logger.debug(f"Abundant memory available ({info['available_gb']:.1f}GB > {required_gb * 3}GB), bypassing pressure checks")
            return False, f"Abundant memory available ({info['available_gb']:.1f}GB)"
        
        # More aggressive memory checking with fragmentation consideration
        if pressure == "critical":
            # Even in critical pressure, allow loading if we have enough available memory
            if info["available_gb"] > required_gb * 2:
                logger.warning(f"Critical pressure but sufficient memory available ({info['available_gb']:.1f}GB > {required_gb * 2}GB), allowing load")
                return False, f"Critical pressure bypassed due to sufficient available memory"
            return True, f"Critical memory pressure ({info['used_percent']:.1f}%)"
        elif pressure == "high":
            if not ResourceMonitor.check_memory_available(model_name, safety_margin=1.2):  # Reduced safety margin
                return True, f"High memory pressure with insufficient memory (need {required_gb}GB, available {info['available_gb']:.1f}GB)"
        elif pressure == "moderate":
            # Only defer if both fragmentation is high AND memory is tight
            if info["fragmentation_score"] > 70 and not ResourceMonitor.check_memory_available(model_name, safety_margin=1.5):
                return True, f"Moderate memory pressure with fragmentation concerns (frag score: {info['fragmentation_score']:.1f})"
        
        return False, "Memory sufficient for loading"

class MLXModelManager:
    def __init__(self, max_tokens=4096, timeout=120):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.default_max_tokens = max_tokens
        self.default_timeout = timeout
        self.model_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1) # Serializes MLX operations
        self.start_time = time.time()
        self.request_count = 0
        
        self.available_models = self._validate_models()
        logger.info(f"Validated {len(self.available_models)} available models")
    
    def increment_request_count(self):
        with self.model_lock: # Though simple, locking ensures it's safe if other counters are added
            self.request_count += 1
    
    def get_health_metrics(self):
        mem_info = ResourceMonitor.get_memory_info()
        return {
            "uptime_seconds": int(time.time() - self.start_time),
            "request_count": self.request_count,
            "current_model": self.current_model_name,
            "memory": mem_info,
            "memory_pressure": ResourceMonitor.get_memory_pressure(),
            "memory_health": {
                "fragmentation_score": mem_info["fragmentation_score"],
                "swap_pressure": "high" if mem_info["swap_percent"] > 20 else "low",
                "can_load_models": [model for model in ModelConfig.get_available_models() 
                                   if not ResourceMonitor.should_defer_model_load(model)[0]]
            }
        }
    
    def _validate_models(self):
        """Validate which models are actually available with memory awareness"""
        available = []
        memory_info = ResourceMonitor.get_memory_info()
        
        for model_name in ModelConfig.get_available_models():
            try:
                if ".." in model_name or model_name.startswith("/") or model_name.endswith("/"):
                    logger.warning(f"Skipping potentially unsafe model name: {model_name}")
                    continue
                
                if model_name.count("/") != 1:
                    logger.warning(f"Skipping invalid model format: {model_name}")
                    continue
                
                # Check memory compatibility
                required_memory = ModelConfig.get_config(model_name).get("required_memory_gb", 8)
                if required_memory > memory_info["total_gb"]:
                    logger.warning(f"Model {model_name} requires {required_memory}GB but system only has {memory_info['total_gb']:.1f}GB total memory")
                    continue
                
                available.append(model_name)
                logger.debug(f"Model available: {model_name} (requires {required_memory}GB)")
                
            except Exception as e:
                logger.warning(f"Model {model_name} validation failed: {e}")
        
        # Sort by memory requirement for better loading order
        available.sort(key=lambda m: ModelConfig.get_config(m).get("required_memory_gb", 8))
        return available
    
    def get_recommended_model(self):
        """Get recommended model based on current memory state"""
        if not self.available_models:
            return None
        
        memory_info = ResourceMonitor.get_memory_info()
        pressure = ResourceMonitor.get_memory_pressure()
        
        if pressure in ["critical", "high"]:
            # Suggest the smallest available model
            return ModelConfig.suggest_best_model_for_memory(
                memory_info["available_gb"], prefer_performance=False
            )
        else:
            # Suggest the best performing model that fits
            return ModelConfig.suggest_best_model_for_memory(
                memory_info["available_gb"], prefer_performance=True
            )
    
    def load_model(self, model_name):
        """Thread-safe model loading without caching"""
        if not model_name or model_name not in self.available_models:
            logger.error(f"Invalid model name: {model_name}")
            return False
            
        with self.model_lock:
            if self.current_model_name == model_name:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            # Enhanced memory pressure checking
            should_defer, defer_reason = ResourceMonitor.should_defer_model_load(model_name)
            if should_defer:
                logger.warning(f"Deferring model load for {model_name}: {defer_reason}")
                logger.info("Attempting aggressive memory cleanup before retry...")
                self._aggressive_cleanup_memory()
                
                # Re-check after aggressive cleanup
                should_defer_retry, defer_reason_retry = ResourceMonitor.should_defer_model_load(model_name)
                if should_defer_retry:
                    logger.error(f"Cannot load {model_name} even after cleanup: {defer_reason_retry}")
                    return False
                else:
                    logger.info(f"Memory cleanup successful, proceeding with {model_name} load")

            logger.info(f"🔄 Loading model: {model_name}")
            
            if self.current_model is not None:
                self._unload_current_model()
            
            try:
                start_time = time.time()
                self.current_model, self.current_tokenizer = load(model_name)
                
                self._warmup_model()
                
                load_time = time.time() - start_time
                self.current_model_name = model_name
                
                mem_info = ResourceMonitor.get_memory_info()
                logger.info(f"✅ Loaded {model_name} in {load_time:.2f}s (Memory: {mem_info['used_percent']:.1f}%)")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}")
                self._cleanup_memory() # Cleanup after failed load
                return False

    def _unload_current_model(self):
        """Safely unload current model and free memory"""
        if self.current_model_name: # Check if a model is actually loaded
            logger.info(f"Unloading model: {self.current_model_name}")
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            self._cleanup_memory()
        else:
            logger.debug("No current model to unload.")
    
    def _cleanup_memory(self):
        """Standard memory cleanup for Apple Silicon"""
        gc.collect()
        try:
            mx.clear_cache() # MLX specific cache clearing
        except AttributeError: # In case mx.clear_cache() isn't available in some versions
            logger.debug("mx.clear_cache() not available in this MLX version.")
        except Exception as e:
            logger.warning(f"Error during mx.clear_cache(): {e}")
    
    def _aggressive_cleanup_memory(self):
        """Aggressive memory cleanup with fragmentation management"""
        logger.info("Starting aggressive memory cleanup...")
        
        # Get baseline memory info
        mem_before = ResourceMonitor.get_memory_info()
        
        # Multiple rounds of garbage collection
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)  # Brief pause between collections
        
        # MLX-specific cleanup
        try:
            mx.clear_cache()
            # Additional MLX cleanup if available
            if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()
        except AttributeError:
            logger.debug("Advanced MLX cache clearing not available in this version.")
        except Exception as e:
            logger.warning(f"Error during advanced MLX cleanup: {e}")
        
        # Force Python memory compaction (if available)
        try:
            import ctypes
            libc = ctypes.CDLL("libc.dylib")
            libc.malloc_zone_pressure_relief(None, 0)
            logger.debug("Applied macOS memory pressure relief")
        except (ImportError, AttributeError, OSError):
            logger.debug("macOS memory pressure relief not available")
        
        # Final cleanup round
        gc.collect()
        
        # Report cleanup effectiveness
        mem_after = ResourceMonitor.get_memory_info()
        freed_gb = mem_before["available_gb"] - mem_after["available_gb"]
        logger.info(f"Aggressive cleanup completed. Freed: {abs(freed_gb):.2f}GB, Available: {mem_after['available_gb']:.2f}GB ({mem_after['used_percent']:.1f}% used)")
    
    def _warmup_model(self):
        """Warm up model with a quick generation"""
        if not self.current_model or not self.current_tokenizer:
            logger.warning("Cannot warmup model: model or tokenizer not loaded.")
            return
        try:
            test_prompt = self._get_default_prompt()
            _ = generate( # Assign to _ to indicate result is not used
                self.current_model,
                self.current_tokenizer,
                prompt=test_prompt,
                max_tokens=5, # Minimal tokens for warmup
                verbose=False
            )
            logger.debug(f"Warmup successful for {self.current_model_name}")
        except Exception as e:
            logger.debug(f"Warmup failed for {self.current_model_name} (non-critical): {e}")
    
    def _get_default_prompt(self):
        """Model-specific default prompts for warmup or empty inputs"""
        if not self.current_model_name: # Generic prompt if no model context
            return "<|user|>\nHello\n<|assistant|>\n"
            
        # These are examples; ensure they match actual model templating if critical for warmup
        if "Llama-3" in self.current_model_name: # Covers Llama-3.3 and Llama-3.2
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "Qwen" in self.current_model_name:
            return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        elif "DeepSeek" in self.current_model_name: # Covers DeepSeek Coder and other DeepSeek
            return "User: Hello\n\nAssistant: "
        elif "Phi-4" in self.current_model_name:
            return "<|user|>\nHello<|end|>\n<|assistant|>\n"
        else: # Fallback generic prompt
            return "<|user|>\nHello\n<|assistant|>\n"
    
    def _format_messages(self, messages):
        """Format messages with model-specific chat templates"""
        if not messages:
            return self._get_default_prompt() # Use default if messages are empty
        
        chat_template_name = ModelConfig.get_chat_template(self.current_model_name)
        
        # Apply model-specific formatting logic
        if chat_template_name == "llama3":
            return self._format_llama3_messages(messages)
        elif chat_template_name == "deepseek":
            return self._format_deepseek_messages(messages)
        elif chat_template_name == "qwen" or chat_template_name == "chatml":
            return self._format_qwen_messages(messages)
        elif chat_template_name == "phi4":
            return self._format_phi4_messages(messages)
        else: # Generic fallback
            return self._format_generic_messages(messages)
    
    def _format_llama3_messages(self, messages):
        """
        Format messages using the official Llama 3 chat template.
        
        Based on Meta's official Llama 3 instruction format:
        - Messages start with <|begin_of_text|> token
        - Each message has <|start_header_id|>{role}<|end_header_id|> header
        - Content follows after double newline
        - Each message ends with <|eot_id|> token
        - Assistant's turn starts with its header ready for generation
        """
        prompt = "<|begin_of_text|>"
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "").strip()
            
            # Skip empty messages except system messages (which can be empty placeholders)
            if not content and role != "system":
                continue
                
            # Format: <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        
        # End with assistant header ready for response generation
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    def _format_deepseek_messages(self, messages):
        """
        Format messages using the official DeepSeek chat template.
        
        Based on the official template from DeepSeek-Coder documentation:
        - System prompt for instruction-following
        - ### Instruction: for user messages
        - ### Response: for assistant messages
        """
        prompt = ""
        
        # Check if user provided a system prompt
        has_system_prompt = any(msg.get("role", "").lower() == "system" for msg in messages)
        
        # Add default system prompt if none provided
        if not has_system_prompt:
            # Official DeepSeek system prompt (English only)
            prompt += "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n"
        
        # Process messages
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "").strip()
            
            if role == "system":
                # Add user's system prompt
                prompt += f"{content}\n"
            elif role == "user":
                # Add user instruction
                prompt += f"### Instruction:\n{content}\n"
            elif role == "assistant":
                # Add assistant response
                prompt += f"### Response:\n{content}\n"
        
        # End with response marker for the model to continue
        prompt += "### Response:\n"
        
        return prompt

    def _format_qwen_messages(self, messages):
        """
        Format messages using the ChatML template for Qwen models.
        
        Qwen uses ChatML format:
        - <|im_start|> marks the beginning of a message
        - Role (system/user/assistant) follows immediately
        - Content appears on the next line
        - <|im_end|> marks the end of a message
        - No default system prompt in newer Qwen versions (Qwen2.5+)
        """
        prompt = ""
        
        # Track if a system message was provided
        has_system_message = False
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "").strip()
            
            # Skip empty messages
            if not content:
                continue
                
            # Format: <|im_start|>{role}\n{content}<|im_end|>\n
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
            if role == "system":
                has_system_message = True
        
        # Note: Qwen2.5+ models don't use a default system prompt
        # Earlier versions used "You are a helpful assistant."
        # We'll omit the default to follow the latest convention
        
        # End with assistant marker ready for generation
        prompt += "<|im_start|>assistant\n"
        
        return prompt

    def _format_phi4_messages(self, messages):
        """
        Format messages using the Phi-4 chat template.
        
        Phi-4 uses a modified ChatML-style format:
        - <|im_start|>{role}<|im_sep|> marks the beginning with role
        - Content follows immediately
        - <|im_end|> marks the end of each message
        - System messages are supported in Phi-4 (unlike Phi-3)
        
        Note: Phi-3 used <|user|>, <|assistant|>, <|system|>, <|end|> tokens
        but Phi-4 switched to ChatML-style with <|im_sep|> separator
        """
        prompt = ""
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "").strip()
            
            # Skip empty messages
            if not content:
                continue
            
            # Phi-4 format: <|im_start|>{role}<|im_sep|>\n{content}<|im_end|>\n
            prompt += f"<|im_start|>{role}<|im_sep|>\n{content}<|im_end|>\n"
        
        # End with assistant marker ready for generation
        prompt += "<|im_start|>assistant<|im_sep|>\n"
        
        return prompt

    def _format_generic_messages(self, messages):
        """
        Format messages using a generic template for unknown/unsupported models.
        
        This is a fallback formatter that attempts to create a reasonable
        prompt structure when the specific model format is unknown.
        Uses a simple but clear format that most models can understand.
        """
        prompt = ""
        
        # Track if we've seen a system message
        has_system = False
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "").strip()
            
            # Skip empty messages
            if not content:
                continue
            
            # Handle different roles with clear separation
            if role == "system":
                # System messages often work better at the beginning
                # Use a clear indicator that this is system context
                prompt += f"System: {content}\n\n"
                has_system = True
                
            elif role == "user":
                # Clear user designation with proper spacing
                if prompt and not prompt.endswith("\n\n"):
                    prompt += "\n"
                prompt += f"User: {content}\n"
                
            elif role == "assistant":
                # Assistant responses
                prompt += f"Assistant: {content}\n"
                
            else:
                # Handle any custom roles gracefully
                # Capitalize first letter for consistency
                role_display = role.capitalize()
                if prompt and not prompt.endswith("\n\n"):
                    prompt += "\n"
                prompt += f"{role_display}: {content}\n"
        
        # If no system message was provided, optionally add a minimal one
        if not has_system and messages:
            # Prepend a basic system context
            prompt = "You are a helpful AI assistant.\n\n" + prompt
        
        # Ensure proper spacing before assistant prompt
        if prompt and not prompt.endswith("\n"):
            prompt += "\n"
        
        # Use a clear, universal format for the assistant's turn
        prompt += "Assistant:"
        
        # Add a space after the colon for better generation
        prompt += " "
        
        return prompt
    
    def _sanitize_response(self, response_text):
        """Sanitize model response for proper rendering, preserving meaningful newlines."""
        if not response_text:
            return ""
        
        # Define cleanup patterns - adjust as needed, less aggressive now
        cleanup_patterns = [
            # Specific template artifacts (examples, expand as needed)
            (r'<\|start_header_id\|>.*?<\|end_header_id\|>', ''),
            (r'<\|eot_id\|>', ''),
            (r'<\|begin_of_text\|>', ''),
            (r'<\|end_of_text\|>', ''),
            (r'<\|im_start\|>system\n.*?\n<\|im_end\|>', ''), # Remove system blocks if they appear in response
            (r'<\|im_start\|>user\n.*?\n<\|im_end\|>', ''),   # Remove user blocks
            (r'<\|im_start\|>assistant\n', ''),             # Remove assistant start if it's just the tag
            (r'<\|im_end\|>', ''),
            (r'<\|user\|>', ''), (r'<\|end\|>', ''), # For Phi style
            (r'<\|assistant\|>', ''),
             # More general tags - be cautious not to remove legitimate content
            (r'^\s*<\|.*?\|>\s*\n?', ''), # Tags at the beginning of the string or lines

            # Common prefixes if they are redundant
            (r'^(Assistant|User|System):\s*', ''),
        ]
        
        cleaned = response_text
        for pattern, replacement in cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.DOTALL | re.MULTILINE)
        
        # Normalize whitespace while preserving newlines
        lines = cleaned.split('\n')
        processed_lines = []
        for line in lines:
            stripped_line = " ".join(line.split()) # Normalize spaces within a line
            if stripped_line: # Keep line if it has content after normalization
                processed_lines.append(stripped_line)
            elif processed_lines and processed_lines[-1]: # Keep a single empty line if it follows a content line (for paragraph breaks)
                 processed_lines.append("")


        # Reconstruct, then remove leading/trailing whitespace from the whole block
        # and ensure not too many consecutive newlines
        result = '\n'.join(processed_lines).strip()
        result = re.sub(r'\n{3,}', '\n\n', result) # Collapse 3+ newlines to 2

        if not result or len(result.strip()) < 1: # Allow single character responses
            logger.warning("Response was empty or very short after sanitization.")
            # Consider if a default apology is always best or if empty is sometimes valid
            return "I apologize, but I encountered an issue generating a response. Please try again."
        
        return result

    def generate_response(self, messages, max_tokens=None, temperature=None, top_p=None, top_k=None, min_p=None):
        """Generate response with timeout protection"""
        if not self.current_model:
            logger.error("Generation attempted without a loaded model.")
            return "ERROR: No model loaded. Please select a model first."
        
        model_config = ModelConfig.get_config(self.current_model_name)
        
        # Use provided parameters or fall back to model_config, then general defaults
        max_tokens_val = max_tokens if max_tokens is not None else model_config.get("max_tokens", self.default_max_tokens)
        temperature_val = temperature if temperature is not None else model_config.get("temp", 0.7)
        top_p_val = top_p if top_p is not None else model_config.get("top_p", 0.9)
        top_k_val = top_k if top_k is not None else model_config.get("top_k", 40)
        min_p_val = min_p if min_p is not None else model_config.get("min_p", 0.05)

        # Validate parameters (ensure they are within reasonable bounds)
        temperature_val = max(0.01, min(2.0, temperature_val))
        top_p_val = max(0.01, min(1.0, top_p_val))
        max_tokens_val = max(1, min(max_tokens_val, model_config.get("max_tokens", self.default_max_tokens))) # Ensure max_tokens doesn't exceed model's capability or a safe upper bound

        # Apply memory pressure-aware token adjustment
        pressure_level = ResourceMonitor.get_memory_pressure()
        if pressure_level != "normal":
            pressure_max_tokens = ResourceMonitor.get_memory_pressure_max_tokens(self.current_model_name, pressure_level)
            if pressure_max_tokens < max_tokens_val:
                logger.warning(f"Memory pressure '{pressure_level}' detected. Reducing max_tokens from {max_tokens_val} to {pressure_max_tokens}")
                max_tokens_val = pressure_max_tokens
        
        prompt = self._format_messages(messages)
        
        try:
            future = self.executor.submit(
                self._generate_with_mlx, 
                prompt, max_tokens_val, temperature_val, top_p_val, top_k_val, min_p_val
            )
            raw_response = future.result(timeout=self.default_timeout) # Configurable timeout
            
            return self._sanitize_response(raw_response)
            
        except FutureTimeoutError:
            logger.error(f"Generation timeout after {self.default_timeout}s for model {self.current_model_name}")
            return "ERROR: Generation timed out. Try a shorter prompt or reduce max_tokens."
        except Exception as e:
            logger.error(f"Error during generation with {self.current_model_name}: {e}", exc_info=True)
            return f"ERROR: Generation failed due to an internal error: {str(e)}"

    def _generate_with_mlx(self, prompt, max_tokens, temperature, top_p, top_k, min_p):
        """Core MLX generation logic"""
        try:
            sampler_args = {"temp": temperature, "top_p": top_p, "min_tokens_to_keep": 1}
            if min_p is not None and min_p > 0.0: sampler_args["min_p"] = min_p
            if top_k is not None and top_k > 0: sampler_args["top_k"] = top_k
            
            # Ensure sampler creation itself is robust
            try:
                sampler = make_sampler(**sampler_args)
            except TypeError as te: # Handle cases where some args might not be supported by current make_sampler
                logger.warning(f"Sampler creation TypeError: {te}. Trying with basic args.")
                # Fallback to more basic sampler if specific args cause issues
                basic_sampler_args = {"temp": temperature, "top_p": top_p, "min_tokens_to_keep": 1}
                if top_k is not None and top_k > 0: # top_k is often problematic if not supported
                    try: make_sampler(**{"top_k":top_k}) # test if top_k alone is the problem
                    except TypeError: 
                        logger.warning("top_k sampler arg not supported by make_sampler, removing.")
                    else: basic_sampler_args["top_k"]=top_k
                sampler = make_sampler(**basic_sampler_args)
            except Exception as e_sampler: # Catch other sampler creation errors
                logger.error(f"Unexpected error creating sampler: {e_sampler}. Using default sampler.")
                sampler = make_sampler(temp=0.7, top_p=0.9, min_tokens_to_keep=1) # Last resort default

            start_time = time.time()
            response = generate(
                self.current_model, self.current_tokenizer,
                prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False
            )
            gen_time = time.time() - start_time
            # Simple token estimation for logging; actual tokens calculated later
            num_tokens_generated = len(self.current_tokenizer.encode(response)) if self.current_tokenizer else len(response.split())
            logger.info(f"Generated ~{num_tokens_generated} tokens in {gen_time:.2f}s using {self.current_model_name}")
            return response
        except Exception as e:
            logger.error(f"MLX generation core error with {self.current_model_name}: {e}", exc_info=True)
            raise # Re-raise to be caught by the calling function's error handler

    def get_proper_token_count(self, text):
        """Get accurate token count using the loaded tokenizer."""
        if not self.current_tokenizer or not text:
            return 0
        try:
            # Standard way to get token IDs from mlx_lm Tokenizer
            if hasattr(self.current_tokenizer, 'encode'):
                tokens = self.current_tokenizer.encode(text)
                return len(tokens) # encode usually returns a list of ints
            else: # Fallback if no 'encode' method (should not happen with mlx_lm tokenizers)
                logger.warning(f"Tokenizer for {self.current_model_name} lacks 'encode' method. Falling back to word count for text: '{text[:50]}...'")
                return len(text.split())
        except Exception as e:
            logger.warning(f"Token counting failed for {self.current_model_name} on text '{text[:50]}...': {e}. Falling back to word count.", exc_info=True)
            return len(text.split())

    def _calculate_usage_tokens(self, messages, generated_content):
        """Calculate token usage for API response."""
        try:
            # Format messages to get the actual prompt sent to the model
            formatted_prompt = self._format_messages(messages)
            prompt_tokens = self.get_proper_token_count(formatted_prompt)
            completion_tokens = self.get_proper_token_count(generated_content)
            
            # Ensure tokens are at least 1 if content exists, or 0 if not.
            prompt_tokens = max(0, prompt_tokens) if formatted_prompt else 0
            completion_tokens = max(0, completion_tokens) if generated_content else 0
            
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        except Exception as e:
            logger.error(f"Token calculation error: {e}", exc_info=True)
            # Fallback to rough estimate
            prompt_len = sum(len(msg.get('content', '').split()) for msg in messages)
            completion_len = len(generated_content.split())
            return {
                "prompt_tokens": max(1, prompt_len) if prompt_len > 0 else 0,
                "completion_tokens": max(1, completion_len) if completion_len > 0 else 0,
                "total_tokens": max(1, prompt_len) + max(1, completion_len) if (prompt_len > 0 or completion_len > 0) else 0
            }

class APIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, model_manager=None, **kwargs):
        self.model_manager = model_manager
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")

    def _send_cors_headers_only(self):
        """Helper that ONLY sends CORS headers. Assumes send_response was called first."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '3600')

    def do_OPTIONS(self):
        """Correctly handle CORS preflight requests."""
        self.send_response(204) # 204 No Content
        self._send_cors_headers_only()
        self.end_headers()
    
    def _send_json_response(self, response_data, status_code=200):
        """Send a JSON response with consistent CORS and Content-Length headers."""
        try:
            body = json.dumps(response_data).encode('utf-8')
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers_only()
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            logger.error(f"Failed to send JSON response: {e}", exc_info=True)
            if not self.headers_sent:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self._send_cors_headers_only()
                self.end_headers()
            error_response = {"error": {"message": "Server error while sending response.", "type": "server_error"}}
            try:
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
            except Exception as final_e:
                logger.error(f"Truly failed to send any error response: {final_e}")

    def _validate_json_input(self, raw_data, max_size_mb=10):
        """Validate and parse JSON input from request body."""
        if len(raw_data) > max_size_mb * 1024 * 1024:
            raise ValueError(f"Request payload too large (>{max_size_mb}MB).")
        try:
            return json.loads(raw_data.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON payload: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid UTF-8 encoding in payload: {e}")

    def _validate_api_response(self, content, model_name):
        """Validate API response content (after sanitization) for potential issues."""
        if not isinstance(content, str): # Should always be string after sanitization
            logger.error(f"Invalid content type for validation: {type(content)} for model {model_name}")
            return False, "Invalid content type (not a string)."
        
        issues = []
        # These checks assume _sanitize_response has already run.
        # They are looking for things that either sanitization missed,
        # or patterns that are still problematic despite general cleaning.
        
        # Example: check for overly long words which might indicate garbage
        if any(len(word) > 80 for word in content.split()): # 80 is arbitrary
            issues.append("Contains unusually long words.")

        # Example: check for excessive repetition (simple version)
        if len(content) > 100: # Only for longer content
            substrings = [content[i:i+20] for i in range(0, len(content)-20, 10)] # Sample substrings
            if len(substrings) != len(set(substrings)): # Basic check for repeated substrings
                # This is a very naive check and can have false positives.
                # issues.append("Potential excessive repetition detected.")
                pass # Commented out as it's noisy. A better check would be needed.

        # Check JSON compatibility of the content string itself (e.g. unescaped controls)
        try:
            json.dumps({"test_content": content})
        except (TypeError, ValueError) as e:
            issues.append(f"Content not JSON serializable: {str(e)}")
        
        if issues:
            logger.warning(f"Post-sanitization validation issues for {model_name}: {issues}. Content snippet: '{content[:100]}...'")
            return False, "; ".join(issues)
        
        return True, "Valid"

    def do_GET(self):
        if self.path == '/v1/models':
            self._handle_models_list()
        elif self.path == '/health' or self.path == '/v1/health': # More specific health paths
            self._handle_health_check_api()
        else:
            self._send_json_response({"error": {"message": "Endpoint not found", "type": "invalid_request_error"}}, 404)

    def _handle_models_list(self):
        memory_info = ResourceMonitor.get_memory_info()
        
        models_data = []
        for model_id in self.model_manager.available_models:
            model_config = ModelConfig.get_config(model_id)
            can_load, load_reason = ResourceMonitor.should_defer_model_load(model_id)
            
            model_data = {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-router",
                "memory_requirements": {
                    "required_gb": model_config.get("required_memory_gb", 8),
                    "can_load_now": not can_load,
                    "load_status_reason": load_reason if can_load else "Available for loading"
                },
                "parameters": {
                    "max_tokens": model_config.get("max_tokens", 4096),
                    "chat_template": model_config.get("chat_template", "generic")
                }
            }
            models_data.append(model_data)
        
        recommended_model = self.model_manager.get_recommended_model()
        
        response = {
            "object": "list",
            "data": models_data,
            "memory_status": {
                "available_gb": round(memory_info["available_gb"], 2),
                "pressure": ResourceMonitor.get_memory_pressure(),
                "recommended_model": recommended_model
            }
        }
        self._send_json_response(response)

    def _handle_health_check_api(self):
        metrics = self.model_manager.get_health_metrics()
        
        # Determine overall health status
        memory_pressure = metrics["memory_pressure"]
        health_status = "healthy"
        if memory_pressure == "critical":
            health_status = "critical"
        elif memory_pressure == "high":
            health_status = "degraded"
        elif memory_pressure == "moderate":
            health_status = "warning"
        
        status_response = {
            "status": health_status,
            "uptime_seconds": metrics["uptime_seconds"],
            "request_count": metrics["request_count"],
            "current_model": metrics["current_model"],
            "memory": {
                "total_gb": round(metrics["memory"]["total_gb"], 2),
                "available_gb": round(metrics["memory"]["available_gb"], 2),
                "used_gb": round(metrics["memory"]["used_gb"], 2),
                "used_percent": round(metrics["memory"]["used_percent"], 1),
                "pressure": memory_pressure,
                "fragmentation_score": round(metrics["memory"]["fragmentation_score"], 1),
                "swap_used_percent": round(metrics["memory"]["swap_percent"], 1)
            },
            "memory_health": metrics["memory_health"]
        }
        self._send_json_response(status_response)

    def do_POST(self):
        if self.path == '/v1/chat/completions':
            self._handle_chat_completions()
        else:
            self._send_json_response({"error": {"message": "Endpoint not found", "type": "invalid_request_error"}}, 404)
    
    def _handle_streaming_response(self, messages, model_name, max_tokens, temperature, top_p, top_k, min_p, request_id):
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self._send_cors_headers() # Apply CORS for streaming
            self.end_headers()

            # Model loading should have happened in _handle_chat_completions before calling this
            if not self.model_manager.current_model_name == model_name or not self.model_manager.current_model:
                 raise ValueError(f"Model {model_name} not correctly loaded for streaming.")

            prompt = self.model_manager._format_messages(messages)
            
            # Sampler setup (similar to non-streaming, ensure robustness)
            sampler_args = {"temp": temperature, "top_p": top_p, "min_tokens_to_keep": 1}
            if min_p is not None and min_p > 0.0: sampler_args["min_p"] = min_p
            if top_k is not None and top_k > 0: sampler_args["top_k"] = top_k
            
            try:
                sampler = make_sampler(**sampler_args)
            except Exception as e_sampler:
                logger.warning(f"Stream sampler creation error: {e_sampler}, using basic. Request ID: {request_id}")
                sampler = make_sampler(temp=temperature, top_p=top_p, min_tokens_to_keep=1)


            full_response_content = []
            for token_chunk in generate(self.model_manager.current_model, self.model_manager.current_tokenizer, prompt, max_tokens, sampler=sampler, stream=True):
                # Accumulate for final token count
                full_response_content.append(token_chunk)

                # Sanitize each token chunk - light sanitization for streaming
                # More complex regex might be too slow per token.
                # Consider if sanitization is needed per token or only at the end for streaming.
                # For now, sending raw token, assuming client handles display.
                # Or a very light common artifact removal:
                # token_chunk = token_chunk.replace("<|eot_id|>", "").replace("<|endoftext|>", "")

                sse_event_id = f"chatcmpl-{request_id}-{int(time.time()*1000)}"
                chunk_data = {
                    "id": sse_event_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": token_chunk}, "finish_reason": None}]
                }
                self.wfile.write(f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8'))
                self.wfile.flush()
            
            # Send final chunk with finish_reason
            final_event_id = f"chatcmpl-{request_id}-{int(time.time()*1000)}-final"
            final_chunk_data = {
                "id": final_event_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 # Calculate usage for the final (empty) chunk for some clients.
                "usage": self.model_manager._calculate_usage_tokens(messages, "".join(full_response_content))
            }
            self.wfile.write(f"data: {json.dumps(final_chunk_data)}\n\n".encode('utf-8'))
            self.wfile.flush()
            
            # SSE standard: end with a [DONE] message
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        except BrokenPipeError:
            logger.info(f"Client disconnected during streaming. Request ID: {request_id}")
        except Exception as e:
            logger.error(f"Streaming error for request ID {request_id}: {e}", exc_info=True)
            if not self.headers_sent or self.wfile.writable(): # Check if we can still write to the stream
                try:
                    error_payload = {"error": {"message": str(e), "type": "streaming_error", "code": "stream_failed"}}
                    self.wfile.write(f"data: {json.dumps(error_payload)}\n\n".encode('utf-8'))
                    self.wfile.write(b"data: [DONE]\n\n") # Still send DONE after error for robust clients
                    self.wfile.flush()
                except Exception as write_err:
                     logger.error(f"Failed to write error to stream for request ID {request_id}: {write_err}")
        finally:
            logger.info(f"Streaming finished for request ID: {request_id}")

    def _handle_chat_completions(self):
        request_id = hashlib.md5(f"{time.time()}{self.client_address}".encode()).hexdigest()[:12]
        # Create a logger specific to this request for better traceability
        req_logger = logging.getLogger(f"{__name__}.ReqID.{request_id}")
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                raise ValueError("Empty request body.")
            if content_length > 10 * 1024 * 1024: # 10MB limit
                raise ValueError(f"Request payload too large: {content_length} bytes.")

            post_data = self.rfile.read(content_length)
            data = self._validate_json_input(post_data)
            
            start_time = time.time()
            req_logger.debug(f"Received chat completion request: {json.dumps(data, indent=2)[:500]}...")
            
            model_name = data.get('model')
            messages = data.get('messages', [])
            stream = data.get('stream', False)
            
            if not model_name or not messages:
                raise ValueError("Missing required fields: 'model' and/or 'messages'.")
            
            # Get parameters from request or use defaults from model config / general defaults
            model_specific_config = ModelConfig.get_config(model_name)
            max_tokens_req = data.get('max_tokens', model_specific_config.get('max_tokens', self.model_manager.default_max_tokens))
            temperature_req = data.get('temperature', model_specific_config.get('temp', 0.7))
            top_p_req = data.get('top_p', model_specific_config.get('top_p', 0.9))
            # top_k and min_p often default to "not set" if not provided by user
            top_k_req = data.get('top_k', model_specific_config.get('top_k', -1)) # -1 can mean "not used" by some samplers
            min_p_req = data.get('min_p', model_specific_config.get('min_p', 0.0)) # 0.0 can mean "not used"

            req_logger.info(f"Processing chat request for model '{model_name}'. Stream: {stream}, MaxTokens: {max_tokens_req}, Temp: {temperature_req:.2f}.")
            
            self.model_manager.increment_request_count()
            
            # Check if model loading should be deferred due to memory pressure
            should_defer, defer_reason = ResourceMonitor.should_defer_model_load(model_name)
            if should_defer:
                raise ValueError(f"Cannot load model {model_name} due to memory constraints: {defer_reason}")
            
            if not self.model_manager.load_model(model_name): # Load model if not already loaded or different
                raise ValueError(f"Failed to load or switch to model: {model_name}. It might be unavailable or system is out of resources.")
            
            if stream:
                self._handle_streaming_response(messages, model_name, max_tokens_req, temperature_req, top_p_req, top_k_req, min_p_req, request_id)
                return # Streaming handles its own response
            
            # Non-streaming part
            generated_content = self.model_manager.generate_response(
                messages, max_tokens_req, temperature_req, top_p_req, top_k_req, min_p_req
            )
            
            req_logger.debug(f"Raw generated content (first 100 chars): '{generated_content[:100]}...'")
            
            if generated_content.startswith("ERROR:"): # Check for errors from generate_response
                raise ValueError(generated_content[7:]) # Propagate error
            
            # Validate the already sanitized content (mostly for logging/awareness)
            is_valid, validation_msg = self._validate_api_response(generated_content, model_name)
            if not is_valid:
                req_logger.warning(f"Post-sanitization validation for model '{model_name}' reported: '{validation_msg}'. Proceeding with the sanitized content.")
                # No re-sanitization here, content is already from _sanitize_response
                if not generated_content or generated_content.strip() == "I apologize, but I encountered an issue generating a response. Please try again.":
                     req_logger.warning(f"Validation failed and content is the default apology. This may indicate a deeper generation problem for model '{model_name}'.")


            usage_stats = self.model_manager._calculate_usage_tokens(messages, generated_content)
            
            response_payload = {
                "id": f"chatcmpl-{request_id}-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_content},
                    "finish_reason": "stop" # Assuming "stop" for non-streamed full responses
                }],
                "usage": usage_stats
            }
            
            elapsed_time = time.time() - start_time
            req_logger.info(f"Successfully delivered non-streaming response for model '{model_name}' in {elapsed_time:.2f}s. Prompt tokens: {usage_stats['prompt_tokens']}, Completion tokens: {usage_stats['completion_tokens']}.")
            req_logger.debug(f"Final JSON response (non-streaming, snippet): {json.dumps(response_payload, indent=2)[:500]}...")
            
            self._send_json_response(response_payload)
            
        except ValueError as ve: # Specific for input errors, config errors etc.
            req_logger.warning(f"ValueError in chat completion: {ve}", exc_info=True)
            error_payload = {"error": {"message": str(ve), "type": "invalid_request_error", "code": "bad_request"}}
            self._send_json_response(error_payload, status_code=400)
        except Exception as e: # Catch-all for other unexpected errors
            req_logger.error(f"Unexpected error in chat completion: {e}", exc_info=True)
            error_payload = {"error": {"message": "An unexpected internal server error occurred.", "type": "server_error", "code": "internal_error"}}
            self._send_json_response(error_payload, status_code=500)

def main():
    args = parse_args()
    
    if args.version:
        print_banner()
        return

    # Initialize default device for MLX (Apple Silicon GPU)
    try:
        mx.set_default_device(mx.gpu)
        logger.info("MLX default device set to GPU.")
    except Exception as e:
        logger.warning(f"Could not set MLX default device to GPU: {e}. MLX will use its default.", exc_info=True)
        # Script can continue, MLX will pick a device (likely CPU if GPU failed)
    
    config_data = {}
    if args.config: # Load external config if path provided
        config_path = Path(args.config)
        if config_path.exists() and config_path.is_file():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                logger.info(f"Successfully loaded configuration from {args.config}")
                
                # Allow overriding ModelConfig.MODELS and default operational parameters
                if 'models' in config_data and isinstance(config_data['models'], dict):
                    ModelConfig.MODELS.update(config_data['models'])
                    logger.info(f"Updated {len(config_data['models'])} model configurations from file.")
                
                if 'defaults' in config_data and isinstance(config_data['defaults'], dict):
                    defaults = config_data['defaults']
                    args.max_tokens = defaults.get('max_tokens', args.max_tokens)
                    args.timeout = defaults.get('timeout', args.timeout)
                    logger.info(f"Applied operational default overrides from config: max_tokens={args.max_tokens}, timeout={args.timeout}s")
                    
            except json.JSONDecodeError as e_json:
                logger.error(f"Failed to parse JSON from config file {args.config}: {e_json}")
                print(f"⚠️ Error: Config file {args.config} contains invalid JSON. Using defaults.")
            except Exception as e_conf:
                logger.error(f"Failed to load or process config file {args.config}: {e_conf}", exc_info=True)
                print(f"⚠️ Error: Could not load config file {args.config}. Using defaults.")
        else:
            logger.warning(f"Config file {args.config} not found or is not a file. Using defaults.")
            print(f"⚠️ Warning: Config file {args.config} not found. Using defaults.")
    
    model_manager = MLXModelManager(max_tokens=args.max_tokens, timeout=args.timeout)
    
    # Preload a default model if specified in config or use the first available one
    # Ensure available_models is not empty before trying to access its first element
    default_model_to_preload = config_data.get('default_model')
    if not default_model_to_preload and model_manager.available_models:
        default_model_to_preload = model_manager.available_models[0] 
        logger.info(f"No default_model in config, attempting to preload first available: {default_model_to_preload}")

    if default_model_to_preload:
        if default_model_to_preload in model_manager.available_models:
            logger.info(f"Attempting to preload default model: {default_model_to_preload}")
            if not model_manager.load_model(default_model_to_preload):
                logger.warning(f"Failed to preload default model {default_model_to_preload}. Server will start without a preloaded model.")
        else:
            logger.warning(f"Default model {default_model_to_preload} specified in config is not in the list of validated available models. Skipping preload.")
    elif not model_manager.available_models:
        logger.warning("No models available or configured. The router will start but may not be able to serve requests.")


    # Factory function for the request handler, passing the model_manager instance
    def handler_factory(*args_handler, **kwargs_handler):
        return APIHandler(*args_handler, model_manager=model_manager, **kwargs_handler)
    
    mem_info = ResourceMonitor.get_memory_info()
    
    print_banner()
    print(f"🚀 MLX Router running on http://{args.ip}:{args.port}")
    print(f"💾 System Memory: Total={mem_info['total_gb']:.1f}GB, Available={mem_info['available_gb']:.1f}GB, Used={mem_info['used_percent']:.1f}%")
    print(f"⚙️ Settings: Default Max Tokens (model can override)={args.max_tokens}, Generation Timeout={args.timeout}s")
    # Removed workers from this print statement
    
    if model_manager.available_models:
        print("📋 Available models (see ModelConfig or your JSON config for full details):")
        for model_id in model_manager.available_models:
            cfg = ModelConfig.get_config(model_id)
            print(f"  - {model_id} (Template: {cfg.get('chat_template', 'generic')}, MaxTokens: {cfg.get('max_tokens')})")
    else:
        print("⚠️ No models loaded or configured. Please check your configuration.")

    http_server = ThreadingHTTPServer((args.ip, args.port), handler_factory)
    http_server.timeout = 300  # Socket timeout for blocking operations
    
    logger.info(f"Starting Threading HTTP server on {args.ip}:{args.port}")

    try:
        http_server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down MLX Router gracefully...")
        logger.info("KeyboardInterrupt received, shutting down server.")
    except Exception as e_server: # Catch other server-level exceptions
        logger.critical(f"HTTP server failed critically: {e_server}", exc_info=True)
        print(f"\n❌ Server failed: {e_server}")
    finally:
        if hasattr(http_server, 'server_close'):
            http_server.server_close()
        if hasattr(model_manager, 'executor') and model_manager.executor:
            model_manager.executor.shutdown(wait=True) # Ensure background threads finish
        logger.info("MLX Router server stopped.")
        print("✅ Server stopped.")

if __name__ == "__main__":
    main()