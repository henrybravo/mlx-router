#!/usr/bin/env python3
"""
MLX Model Manager for handling model loading, generation, and message formatting
"""

import gc
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from mlx_router.config.model_config import ModelConfig
from mlx_router.core.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)

class MLXModelManager:
    """Core logic for managing MLX models, with detailed logic"""
    def __init__(self, max_tokens=4096, timeout=120):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.default_max_tokens = max_tokens
        self.default_timeout = timeout
        self.model_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.start_time = time.time()
        self.request_count = 0
        self.available_models = self._validate_models()
        logger.info(f"Validated {len(self.available_models)} available models")
        
    def refresh_available_models(self):
        """Refresh available models after configuration changes"""
        self.available_models = self._validate_models()
        logger.info(f"Refreshed model list: validated {len(self.available_models)} available models")

    def increment_request_count(self):
        self.request_count += 1

    def get_health_metrics(self):
        mem_info = ResourceMonitor.get_memory_info()
        pressure = ResourceMonitor.get_memory_pressure()
        status = "healthy"
        if pressure == "critical": status = "critical"
        elif pressure == "high": status = "degraded"
        elif pressure == "moderate": status = "warning"
        
        return {
            "status": status,
            "uptime_seconds": int(time.time() - self.start_time),
            "request_count": self.request_count,
            "current_model": self.current_model_name,
            "memory": mem_info,
            "memory_pressure": ResourceMonitor.get_memory_pressure(),
            "memory_health": {
                "fragmentation_score": mem_info["fragmentation_score"],
                "swap_pressure": "high" if mem_info["swap_percent"] > 20 else "low",
                "can_load_models": [model for model in ModelConfig.get_available_models() 
                                   if not self.should_defer_model_load_for_health(model)]
            }
        }
    
    def should_defer_model_load_for_health(self, model_name):
        # A simplified check just for the health endpoint status
        required_gb = ModelConfig.get_config(model_name).get("required_memory_gb", 8)
        info = ResourceMonitor.get_memory_info()
        return info["available_gb"] < required_gb * 1.2

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
        if not self.available_models: return None
        mem_info = ResourceMonitor.get_memory_info()
        pressure = ResourceMonitor.get_memory_pressure()
        prefer_perf = pressure in ["normal", "moderate"]
        return ModelConfig.suggest_best_model_for_memory(mem_info["available_gb"], prefer_performance=prefer_perf)

    def load_model(self, model_name):
        if not model_name or model_name not in self.available_models:
            raise ValueError(f"Invalid or unavailable model name: {model_name}")
            
        with self.model_lock:
            if self.current_model_name == model_name: return True
            
            should_defer, reason = ResourceMonitor.should_defer_model_load(model_name)
            if should_defer:
                raise RuntimeError(f"Cannot load {model_name}: {reason}")

            logger.info(f"🔄 Loading model: {model_name}")
            self._unload_current_model()
            
            try:
                start_time = time.time()
                self.current_model, self.current_tokenizer = load(model_name)
                self._warmup_model()
                self.current_model_name = model_name
                mem_info = ResourceMonitor.get_memory_info()
                logger.info(f"✅ Loaded {model_name} in {time.time() - start_time:.2f}s (Memory: {mem_info['used_percent']:.1f}%)")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load {model_name}: {e}", exc_info=True)
                self._cleanup_memory()
                raise RuntimeError(f"Failed to load model {model_name}")

    def _unload_current_model(self):
        if self.current_model_name:
            logger.info(f"Unloading model: {self.current_model_name}")
            del self.current_model
            del self.current_tokenizer
            self.current_model = self.current_tokenizer = self.current_model_name = None
            self._cleanup_memory()
        else:
            logger.debug("No current model to unload.")

    def _cleanup_memory(self):
        gc.collect()
        try: mx.clear_cache()
        except AttributeError: # In case mx.clear_cache() isn't available in some versions
            logger.debug("mx.clear_cache() not available in this MLX version.")
        except Exception as e:
            logger.warning(f"Error during mx.clear_cache(): {e}")

    def _warmup_model(self):
        if not self.current_model: return
        try:
            generate(self.current_model, self.current_tokenizer, prompt="Hello", max_tokens=5, verbose=False)
            logger.debug(f"Warmup successful for {self.current_model_name}")
        except Exception as e:
            logger.warning(f"Warmup failed for {self.current_model_name} (non-critical): {e}")

    def _get_default_prompt(self):
        """Model-specific default prompts for warmup or empty inputs"""
        if not self.current_model_name:
            return "<|user|>\nHello\n<|assistant|>\n"
            
        if "Llama-3" in self.current_model_name:
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "Qwen" in self.current_model_name:
            return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        elif "DeepSeek" in self.current_model_name:
            return "User: Hello\n\nAssistant: "
        elif "Phi-4" in self.current_model_name:
            return "<|user|>\nHello<|end|>\n<|assistant|>\n"
        else:
            return "<|user|>\nHello\n<|assistant|>\n"

    def _get_msg_attr(self, msg, attr, default=""):
        """Safely get attribute from message (works with both Pydantic models and dicts)"""
        if hasattr(msg, attr):
            return getattr(msg, attr, default)
        elif hasattr(msg, 'get'):
            return msg.get(attr, default)
        else:
            return default

    def _format_messages(self, messages):
        """Format messages with model-specific chat templates"""
        if not messages:
            return self._get_default_prompt()
        
        chat_template_name = ModelConfig.get_chat_template(self.current_model_name)
        
        if chat_template_name == "llama3":
            return self._format_llama3_messages(messages)
        elif chat_template_name == "deepseek":
            return self._format_deepseek_messages(messages)
        elif chat_template_name == "qwen" or chat_template_name == "chatml":
            return self._format_qwen_messages(messages)
        elif chat_template_name == "phi4":
            return self._format_phi4_messages(messages)
        else:
            return self._format_generic_messages(messages)
    
    def _format_llama3_messages(self, messages):
        """Format messages using the official Llama 3 chat template"""
        prompt = "<|begin_of_text|>"
        
        for msg in messages:
            role = self._get_msg_attr(msg, "role", "user").lower()
            content = self._get_msg_attr(msg, "content", "").strip()
            
            if not content and role != "system":
                continue
                
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
    
    def _format_deepseek_messages(self, messages):
        """Format messages using the official DeepSeek chat template"""
        prompt = ""
        
        has_system_prompt = any(self._get_msg_attr(msg, "role", "").lower() == "system" for msg in messages)
        
        if not has_system_prompt:
            prompt += "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n"
        
        for msg in messages:
            role = self._get_msg_attr(msg, "role", "user").lower()
            content = self._get_msg_attr(msg, "content", "").strip()
            
            if role == "system":
                prompt += f"{content}\n"
            elif role == "user":
                prompt += f"### Instruction:\n{content}\n"
            elif role == "assistant":
                prompt += f"### Response:\n{content}\n"
        
        prompt += "### Response:\n"
        return prompt
    
    def _format_qwen_messages(self, messages):
        """Format messages using Qwen/ChatML template"""
        prompt = ""
        
        for msg in messages:
            role = self._get_msg_attr(msg, "role", "user")
            content = self._get_msg_attr(msg, "content", "").strip()
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    def _format_phi4_messages(self, messages):
        """Format messages using Phi-4 template"""
        prompt = ""
        
        for msg in messages:
            role = self._get_msg_attr(msg, "role", "user")
            content = self._get_msg_attr(msg, "content", "").strip()
            
            if role == "user":
                prompt += f"<|user|>\n{content}<|end|>\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}<|end|>\n"
            elif role == "system":
                prompt += f"<|system|>\n{content}<|end|>\n"
        
        prompt += "<|assistant|>\n"
        return prompt
    
    def _format_generic_messages(self, messages):
        """Generic fallback message formatting"""
        prompt = ""
        for msg in messages:
            role = self._get_msg_attr(msg, "role", "user")
            content = self._get_msg_attr(msg, "content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _sanitize_response(self, response_text):
        """Sanitize model response for proper rendering, preserving meaningful newlines."""
        if not response_text:
            return ""
        
        cleanup_patterns = [
            (r'<\|start_header_id\|>.*?<\|end_header_id\|>', ''),
            (r'<\|eot_id\|>', ''),
            (r'<\|begin_of_text\|>', ''),
            (r'<\|end_of_text\|>', ''),
            (r'<\|im_start\|>system\n.*?\n<\|im_end\|>', ''),
            (r'<\|im_start\|>user\n.*?\n<\|im_end\|>', ''),
            (r'<\|im_start\|>assistant\n', ''),
            (r'<\|im_end\|>', ''),
            (r'<\|user\|>', ''), (r'<\|end\|>', ''),
            (r'<\|assistant\|>', ''),
            (r'^\s*<\|.*?\|>\s*\n?', ''),
            (r'^(Assistant|User|System):\s*', ''),
        ]
        
        cleaned = response_text
        for pattern, replacement in cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.DOTALL | re.MULTILINE)
        
        lines = cleaned.split('\n')
        processed_lines = []
        for line in lines:
            stripped_line = " ".join(line.split())
            if stripped_line:
                processed_lines.append(stripped_line)
            elif processed_lines and processed_lines[-1]:
                processed_lines.append("")
        
        result = '\n'.join(processed_lines).strip()
        result = re.sub(r'\n{3,}', '\n\n', result)

        if not result or len(result.strip()) < 1:
            logger.warning("Response was empty or very short after sanitization.")
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
        max_tokens_val = max(1, min(max_tokens_val, model_config.get("max_tokens", self.default_max_tokens)))

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
            raw_response = future.result(timeout=self.default_timeout)
            
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
            except TypeError as te:
                logger.warning(f"Sampler creation TypeError: {te}. Trying with basic args.")
                basic_sampler_args = {"temp": temperature, "top_p": top_p, "min_tokens_to_keep": 1}
                if top_k is not None and top_k > 0:
                    try: 
                        make_sampler(**{"top_k":top_k})
                    except TypeError: 
                        logger.warning("top_k sampler arg not supported by make_sampler, removing.")
                    else: 
                        basic_sampler_args["top_k"]=top_k
                sampler = make_sampler(**basic_sampler_args)
            except Exception as e_sampler:
                logger.error(f"Unexpected error creating sampler: {e_sampler}. Using default sampler.")
                sampler = make_sampler(temp=0.7, top_p=0.9, min_tokens_to_keep=1)

            start_time = time.time()
            response = generate(
                self.current_model, self.current_tokenizer,
                prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False
            )
            gen_time = time.time() - start_time
            num_tokens_generated = len(self.current_tokenizer.encode(response)) if self.current_tokenizer else len(response.split())
            logger.info(f"Generated ~{num_tokens_generated} tokens in {gen_time:.2f}s using {self.current_model_name}")
            return response
        except Exception as e:
            logger.error(f"MLX generation core error with {self.current_model_name}: {e}", exc_info=True)
            raise