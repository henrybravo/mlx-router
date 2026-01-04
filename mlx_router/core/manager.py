#!/usr/bin/env python3
"""
Model Manager for handling MLX model loading, generation, and message formatting
"""

import gc
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Optional, List, Dict, Any

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

try:
    from jsonschema import validate, ValidationError
    _jsonschema_available = True
except ImportError:
    _jsonschema_available = False
    validate = None
    ValidationError = Exception

from mlx_router.config.model_config import ModelConfig
from mlx_router.core.resource_monitor import ResourceMonitor
from mlx_router.core.patterns import CLEANUP_PATTERNS, GPT_OSS_CLEANUP_PATTERNS, REASONING_PATTERNS, GPT_OSS_FINAL_PATTERN, GPT_OSS_CHANNEL_PATTERN, NEWLINE_PATTERN
from mlx_router.core.templates import CHAT_TEMPLATES
from mlx_router.core.content import decode_base64_to_images, is_pdf_data

logger = logging.getLogger(__name__)

try:
    from mlx_vlm import load as load_vision_model
    from mlx_vlm import generate as generate_vision
    from mlx_vlm import apply_chat_template as apply_vision_chat_template
    from mlx_vlm.utils import load_config as load_vision_config
    _vision_available = True
except ImportError:
    _vision_available = False
    logger.warning("mlx-vlm not installed. Vision model support will be disabled.")
    logger.warning("Install with: pip install mlx-vlm>=0.3.9")

class MLXModelManager:
    """Core logic for managing MLX models, with detailed logic"""
    def __init__(self, max_tokens=4096, timeout=120):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.is_vision_model = False
        self.vision_processor = None
        self.vision_config = None
        self.default_max_tokens = max_tokens
        self.default_timeout = timeout
        self.model_lock = threading.Lock()
        self.request_count_lock = threading.Lock()  # New lock for thread-safe request counting
        # Increase max_workers for better concurrency (CPU cores / 2, minimum 1)
        max_workers = max(1, os.cpu_count() // 2) if os.cpu_count() else 1
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.start_time = time.time()
        self.request_count = 0
        self.available_models = self._validate_models()
        logger.info(f"Validated {len(self.available_models)} available models")
        
    def refresh_available_models(self):
        """Refresh available models after configuration changes"""
        with self.model_lock:  # Thread-safe update of available_models
            self.available_models = self._validate_models()
            logger.info(f"Refreshed model list: validated {len(self.available_models)} available models")

    def increment_request_count(self):
        """Thread-safe increment of request count"""
        with self.request_count_lock:
            self.request_count += 1
    
    def shutdown(self):
        """Properly shutdown the executor and clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            logger.info("MLXModelManager executor shut down")

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
        # A simplified check just for the health endpoint status using consolidated method
        can_load, _ = ResourceMonitor.check_memory_available(model_name, safety_margin=1.2)
        return not can_load

    def _validate_models(self):
        """Validate which models are actually available with memory awareness"""
        available = []
        memory_info = ResourceMonitor.get_memory_info()

        for model_name in ModelConfig.get_available_models():
            try:
                # Skip potentially unsafe model names
                if ".." in model_name:
                    logger.warning(f"Skipping potentially unsafe model name: {model_name}")
                    continue

                # For absolute paths, validate they exist
                if os.path.isabs(model_name):
                    if not os.path.exists(model_name):
                        logger.warning(f"Local model path does not exist: {model_name}")
                        continue
                else:
                    # For relative names, check if they can be resolved to a local path or are valid HF identifiers
                    resolved_path = ModelConfig.resolve_model_path(model_name)
                    if resolved_path != model_name and not os.path.exists(resolved_path):
                        # If it's a local model that doesn't exist, skip it
                        if ModelConfig.get_model_directory():
                            logger.warning(f"Local model {model_name} not found in {ModelConfig.get_model_directory()}")
                            continue
                        # If no local directory configured, assume it's a HF model and continue

                # Check memory compatibility using consolidated method
                can_load, _ = ResourceMonitor.can_load_model(model_name, safety_margin=1.0, use_cache=True)
                if not can_load:
                    logger.warning(f"Model {model_name} cannot be loaded due to memory constraints")
                    continue

                available.append(model_name)
                logger.debug(f"Model available: {model_name}")

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
        if not model_name:
            raise ValueError(f"Invalid model name: {model_name}")

        # Check config first for explicit supports_vision setting, then fall back to keyword detection
        model_config = ModelConfig.get_config(model_name)
        is_vision = model_config.get("supports_vision", False) or \
            any(keyword in model_name.lower() for keyword in ['vl', 'vision', 'llava', 'qwen-vl', 'nvl'])

        if is_vision and not _vision_available:
            raise RuntimeError(f"Vision model support not available. Install mlx-vlm>=0.3.9")

        if model_name not in self.available_models:
            if '/' not in model_name or len(model_name.split('/')) != 2:
                raise ValueError(f"Invalid or unavailable model name: {model_name}")
            logger.info(f"Attempting to load non-configured model: {model_name}")

        with self.model_lock:
            if self.current_model_name == model_name: return True

            should_defer, reason = ResourceMonitor.should_defer_model_load(model_name)
            if should_defer:
                raise RuntimeError(f"Cannot load {model_name}: {reason}")

            logger.info(f"🔄 Loading model: {model_name}")
            self._unload_current_model()

            try:
                start_time = time.time()
                resolved_model_path = ModelConfig.resolve_model_path(model_name)
                logger.debug(f"Resolved model path: {resolved_model_path}")

                if is_vision:
                    self.current_model, self.vision_processor = load_vision_model(resolved_model_path)
                    self.vision_config = load_vision_config(resolved_model_path)
                    self.is_vision_model = True
                    self.current_tokenizer = None
                    logger.info(f"Loaded as vision model")
                else:
                    self.current_model, self.current_tokenizer = load(resolved_model_path)
                    self.is_vision_model = False

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
            self.is_vision_model = False
            self.vision_processor = None
            self.vision_config = None
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

    def _format_messages(self, messages, tools=None):
        """Format messages with model-specific chat templates using data-driven approach"""
        if not messages:
            return self._get_default_prompt()

        chat_template_name = ModelConfig.get_chat_template(self.current_model_name)
        template = CHAT_TEMPLATES.get(chat_template_name, CHAT_TEMPLATES['generic'])

        # For models with native tokenizer chat templates (like gpt-oss), use them if available
        if template.get('use_tokenizer_template') and self.current_tokenizer and hasattr(self.current_tokenizer, 'apply_chat_template'):
            try:
                # Convert messages to dict format for tokenizer
                formatted_messages = []
                for msg in messages:
                    role = self._get_msg_attr(msg, "role", "user")
                    content = self._get_msg_attr(msg, "content", "")
                    formatted_messages.append({"role": role, "content": content})

                # Use tokenizer's native chat template
                prompt = self.current_tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                logger.debug(f"Using native tokenizer chat template for {chat_template_name}")
                return prompt
            except Exception as e:
                logger.warning(f"Failed to use native tokenizer chat template: {e}. Falling back to manual formatting.")

        # Start with prefix
        prompt = template.get('prefix', '')
        
        # Handle tools injection into system message
        tools_prompt = ""
        if tools and template.get('tools_system_prompt'):
            tools_json = json.dumps(tools, indent=2)
            tools_prompt = template['tools_system_prompt'].format(tools_json=tools_json)
        
        # Add system default if needed and no system message exists
        has_system = any(self._get_msg_attr(msg, "role", "").lower() == "system" for msg in messages)
        if not has_system:
            system_content = ""
            if template.get('system_default'):
                system_content = template['system_default']
            if tools_prompt:
                system_content = tools_prompt if not system_content else f"{system_content}\n\n{tools_prompt}"
            
            if system_content:
                if template.get('system_format'):
                    prompt += template['system_format'].format(content=system_content)
                elif 'role_format' in template:
                    prompt += template['role_format'].format(role='system', content=system_content)
                elif chat_template_name == 'llama3':
                    prompt += template['role_start'].format(role='system') + system_content + template['role_end']
                else:
                    prompt += system_content
        
        # Process messages
        for msg in messages:
            role = self._get_msg_attr(msg, "role", "user").lower()
            content = self._get_msg_attr(msg, "content", "").strip()
            
            if not content and role != "system":
                continue
            
            # If this is a system message and we have tools, inject tools prompt
            if role == "system" and tools_prompt:
                content = f"{content}\n\n{tools_prompt}" if content else tools_prompt
            
            # Use role-specific format or generic role format
            role_format_key = f"{role}_format"
            if role_format_key in template:
                prompt += template[role_format_key].format(content=content)
            elif 'role_format' in template:
                prompt += template['role_format'].format(role=role, content=content)
            elif chat_template_name == 'llama3':
                # Special handling for llama3 format
                prompt += template['role_start'].format(role=role) + content + template['role_end']
        
        # Add assistant start
        prompt += template.get('assistant_start', '')
        
        return prompt
    
    # Removed individual format functions - now using data-driven approach in _format_messages

    def _sanitize_response(self, response_text):
        """Sanitize model response for proper rendering, preserving meaningful newlines."""
        if not response_text:
            return ""

        cleaned = response_text

        # Get model config to check reasoning_response setting
        model_config = ModelConfig.get_config(self.current_model_name) if self.current_model_name else {}
        reasoning_enabled = model_config.get("reasoning_response", "disable") == "enable"

        # Special handling for GPT-OSS Harmony format
        if GPT_OSS_CHANNEL_PATTERN.search(cleaned):
            if not reasoning_enabled:
                # Extract only final answer channel (default behavior)
                final_match = GPT_OSS_FINAL_PATTERN.search(cleaned)
                if final_match:
                    cleaned = final_match.group(1).strip()
                    logger.debug("Extracted GPT-OSS final answer channel content (reasoning disabled)")
                else:
                    logger.warning("GPT-OSS response contains channels but no 'final' channel found")
            else:
                # Keep full response with reasoning, just clean special tokens
                logger.debug("Keeping full GPT-OSS response with reasoning (reasoning enabled)")

        # Remove Qwen3 and other reasoning patterns (only if reasoning is disabled)
        if not reasoning_enabled:
            for pattern, replacement in REASONING_PATTERNS:
                cleaned = pattern.sub(replacement, cleaned)

        # Apply general cleanup patterns (always apply)
        for pattern, replacement in CLEANUP_PATTERNS:
            cleaned = pattern.sub(replacement, cleaned)

        # Apply GPT-OSS specific patterns only when reasoning is disabled
        if not reasoning_enabled:
            for pattern, replacement in GPT_OSS_CLEANUP_PATTERNS:
                cleaned = pattern.sub(replacement, cleaned)

        # Process lines to normalize whitespace
        lines = cleaned.split('\n')
        processed_lines = []
        for line in lines:
            stripped_line = " ".join(line.split())
            if stripped_line:
                processed_lines.append(stripped_line)
            elif processed_lines and processed_lines[-1]:
                processed_lines.append("")

        result = '\n'.join(processed_lines).strip()
        result = NEWLINE_PATTERN.sub('\n\n', result)

        if not result or len(result.strip()) < 1:
            logger.warning("Response was empty or very short after sanitization.")
            return "I apologize, but I encountered an issue generating a response. Please try again."

        return result

    def _filter_special_token(self, token: str) -> str:
        """Filter individual token for streaming - removes special tokens in real-time"""
        # Quick check: if token doesn't contain special characters, return as-is
        if '<' not in token and '|' not in token:
            return token

        # Apply token-level cleanup patterns (only the simple ones that don't need context)
        filtered = token
        for pattern, replacement in CLEANUP_PATTERNS:
            filtered = pattern.sub(replacement, filtered)

        return filtered

    def _parse_tool_calls(self, text: str, tools: List[dict]) -> Optional[Dict[str, Any]]:
        """Parse model output for tool calls and validate against tool schemas."""
        
        # Extract tool call content
        match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if not match:
            return None
            
        tool_call_str = match.group(1).strip()
        
        try:
            # Parse JSON - handle both single object and array
            parsed = json.loads(tool_call_str)
            if isinstance(parsed, dict):
                parsed = [parsed]
                
            # Validate each tool call against available tools
            validated_calls = []
            for call in parsed:
                tool_name = call.get('name')
                tool_args = call.get('arguments', {})
                
                # Find tool definition
                tool_def = next((t for t in tools if t.get('function', {}).get('name') == tool_name), None)
                if not tool_def:
                    logger.warning(f"Unknown tool called: {tool_name}")
                    continue
                    
                # Validate arguments against schema if jsonschema is available
                try:
                    if 'parameters' in tool_def.get('function', {}):
                        validate(instance=tool_args, schema=tool_def['function']['parameters'])
                    
                    validated_calls.append({
                        'id': f"call_{hashlib.md5(f'{time.time()}{tool_name}'.encode()).hexdigest()[:8]}",
                        'type': 'function',
                        'function': {
                            'name': tool_name,
                            'arguments': json.dumps(tool_args)
                        }
                    })
                except ValidationError as e:
                    logger.warning(f"Invalid arguments for {tool_name}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Tool validation error for {tool_name}: {e}")
                    # Still include the tool call even if validation fails
                    validated_calls.append({
                        'id': f"call_{hashlib.md5(f'{time.time()}{tool_name}'.encode()).hexdigest()[:8]}",
                        'type': 'function',
                        'function': {
                            'name': tool_name,
                            'arguments': json.dumps(tool_args)
                        }
                    })
                    
            return {'tool_calls': validated_calls} if validated_calls else None
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {e}")
            return None

    def generate_response(self, messages, tools=None, max_tokens=None, temperature=None, top_p=None, top_k=None, min_p=None, images=None):
        """Generate response with timeout protection and optional tool support"""
        if not self.current_model:
            logger.error("Generation attempted without a loaded model.")
            return {
                'type': 'error',
                'content': "ERROR: No model loaded. Please select a model first.",
                'finish_reason': 'error'
            }
        
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
        
        # Format prompt with tools if provided
        prompt = self._format_messages(messages, tools)

        try:
            # Generate response based on model type
            if self.is_vision_model:
                future = self.executor.submit(
                    self._generate_with_vlm,
                    prompt, max_tokens_val, temperature_val, top_p_val, top_k_val, min_p_val, images, False
                )
            else:
                future = self.executor.submit(
                    self._generate_with_mlx,
                    prompt, max_tokens_val, temperature_val, top_p_val, top_k_val, min_p_val, False
                )
            raw_response = future.result(timeout=self.default_timeout)

            # Check for tool calls if tools were provided (not applicable for vision models)
            if tools and not self.is_vision_model:
                tool_calls = self._parse_tool_calls(raw_response, tools)
                if tool_calls:
                    return {
                        'type': 'tool_calls',
                        'content': None,
                        'tool_calls': tool_calls['tool_calls'],
                        'finish_reason': 'tool_calls'
                    }
            
            return {
                'type': 'text',
                'content': self._sanitize_response(raw_response),
                'finish_reason': 'stop'
            }
            
        except FutureTimeoutError:
            logger.error(f"Generation timeout after {self.default_timeout}s for model {self.current_model_name}")
            return {
                'type': 'error',
                'content': "ERROR: Generation timed out. Try a shorter prompt or reduce max_tokens.",
                'finish_reason': 'timeout'
            }
        except Exception as e:
            logger.error(f"Error during generation with {self.current_model_name}: {e}", exc_info=True)
            return {
                'type': 'error',
                'content': f"ERROR: Generation failed due to an internal error: {str(e)}",
                'finish_reason': 'error'
            }

    async def generate_stream_response(self, messages, tools=None, max_tokens=None, temperature=None, top_p=None, top_k=None, min_p=None, images=None, stream_chunk_size=None):
        """Generate streaming response for real-time token delivery
        
        Args:
            stream_chunk_size: Buffer size for streaming tokens (default: 8). Higher values may improve
                             throughput but increase latency. Lower values give more real-time delivery.
        """
        if not self.current_model:
            logger.error("Streaming generation attempted without a loaded model.")
            yield "ERROR: No model loaded. Please select a model first."
            return

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

        prompt = self._format_messages(messages, tools)

        try:
            # Run the streaming generator directly in thread pool executor
            import asyncio
            loop = asyncio.get_running_loop()

            def get_token_generator():
                """Worker function to create and return the token generator"""
                try:
                    if self.is_vision_model:
                        # Vision models don't support streaming yet, return full response
                        result = self._generate_with_vlm(
                            prompt, max_tokens_val, temperature_val, top_p_val, top_k_val, min_p_val, images=images, stream=False
                        )
                        # Wrap in a generator to yield the full response as a single token
                        def single_result_gen():
                            yield result
                        return single_result_gen()
                    else:
                        return self._generate_with_mlx(
                            prompt, max_tokens_val, temperature_val, top_p_val, top_k_val, min_p_val, stream=True, stream_chunk_size=stream_chunk_size
                        )
                except Exception as e:
                    logger.error(f"Error in stream worker: {e}", exc_info=True)
                    raise

            # Get the generator from executor
            future = loop.run_in_executor(self.executor, get_token_generator)
            token_generator = await future

            # Yield tokens directly as they become available
            for token in token_generator:
                yield token

        except Exception as e:
            logger.error(f"Error during streaming generation with {self.current_model_name}: {e}", exc_info=True)
            yield f"ERROR: Streaming generation failed: {str(e)}"

    def _generate_with_mlx(self, prompt, max_tokens, temperature, top_p, top_k, min_p, stream=False, stream_chunk_size=None):
        """Core MLX generation logic - can return generator for streaming or full text"""
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
            response_generator = generate(
                self.current_model, self.current_tokenizer,
                prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False
            )
            
            if stream:
                # For streaming, return a generator that yields each token
                return self._stream_tokens(response_generator, start_time, stream_chunk_size)
            else:
                # For non-streaming, collect all tokens and return as string (backward compatibility)
                response = "".join(response_generator)
                gen_time = time.time() - start_time
                num_tokens_generated = len(self.current_tokenizer.encode(response)) if self.current_tokenizer else len(response.split())
                logger.info(f"Generated ~{num_tokens_generated} tokens in {gen_time:.2f}s using {self.current_model_name}")
                return response
                
        except Exception as e:
            logger.error(f"MLX generation core error with {self.current_model_name}: {e}", exc_info=True)
            raise

    def _preprocess_images_for_vlm(self, images: List[str]) -> List[str]:
        """
        Preprocess images for vision model, converting PDFs to images if needed.

        mlx-vlm expects file paths, not base64 data URIs. This method:
        1. Decodes base64 data (handles both images and PDFs)
        2. Saves PIL images to temporary files
        3. Returns list of file paths

        Args:
            images: List of image URLs or base64 data URIs

        Returns:
            List of file paths to temporary image files
        """
        processed_images = []

        for img_data in images:
            # Check if it's a base64 data URI
            if img_data.startswith('data:'):
                try:
                    # Use decode_base64_to_images which handles both images and PDFs
                    pil_images = decode_base64_to_images(img_data)

                    # Save PIL images to temporary files (mlx-vlm expects file paths)
                    for pil_img in pil_images:
                        # Convert to RGB if necessary (PDFs might be RGBA)
                        if pil_img.mode in ('RGBA', 'LA', 'P'):
                            pil_img = pil_img.convert('RGB')

                        # Create a temporary file that won't be deleted automatically
                        # (we'll let the OS clean up temp files)
                        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                        pil_img.save(temp_file.name, format='PNG')
                        temp_file.close()
                        processed_images.append(temp_file.name)
                        logger.debug(f"Saved image to temp file: {temp_file.name}")

                except ImportError as e:
                    # PDF support not available
                    raise RuntimeError(f"PDF processing requires pdf2image. Install with: pip install pdf2image. Error: {e}")
                except Exception as e:
                    logger.error(f"Failed to preprocess image: {e}")
                    raise
            else:
                # URL or file path - pass through as-is
                processed_images.append(img_data)

        return processed_images

    def _cleanup_temp_files(self, file_paths: List[str]):
        """Clean up temporary image files created during preprocessing."""
        for path in file_paths:
            try:
                if path.startswith(tempfile.gettempdir()) and os.path.exists(path):
                    os.unlink(path)
                    logger.debug(f"Cleaned up temp file: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {path}: {e}")

    def _generate_with_vlm(self, prompt, max_tokens, temperature, top_p, top_k, min_p, images=None, stream=False):
        """Core MLX-VLM generation logic - can return generator for streaming or full text"""
        if not _vision_available:
            logger.error("Vision model support not available. Install mlx-vlm>=0.3.9")
            raise RuntimeError("Vision model support requires mlx-vlm to be installed")

        if not self.current_model:
            logger.error("Vision generation attempted without a loaded model.")
            raise RuntimeError("No model loaded.")

        processed_images = []
        try:
            start_time = time.time()

            # Preprocess images - converts PDFs to images and saves to temp files
            processed_images = self._preprocess_images_for_vlm(images) if images else []

            # Prepare images - can be a single image or list of images
            # mlx_vlm expects image as keyword arg, can be str or List[str]
            image_arg = processed_images[0] if len(processed_images) == 1 else (processed_images if processed_images else None)
            num_images = len(processed_images)
            logger.debug(f"Vision generation with {num_images} images")

            # Use mlx-vlm's apply_chat_template to format prompt with image placeholders
            # This is required for vision models to properly handle image tokens
            formatted_prompt = apply_vision_chat_template(
                self.vision_processor,
                self.vision_config,
                prompt,
                num_images=num_images
            )
            logger.debug(f"Formatted vision prompt: {formatted_prompt[:200]}...")

            # Generate using mlx-vlm
            response = generate_vision(
                model=self.current_model,
                processor=self.vision_processor,
                prompt=formatted_prompt,
                image=image_arg,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                verbose=False
            )

            if stream:
                return response
            else:
                gen_time = time.time() - start_time
                # mlx_vlm.generate returns GenerationResult object with .text attribute
                result_text = response.text if hasattr(response, 'text') else str(response)
                logger.info(f"Generated vision response in {gen_time:.2f}s using {self.current_model_name} ({response.generation_tokens} tokens, {response.generation_tps:.1f} tokens/s)")
                return result_text

        except Exception as e:
            logger.error(f"MLX-VLM generation error with {self.current_model_name}: {e}", exc_info=True)
            raise
        finally:
            # Clean up temporary files
            self._cleanup_temp_files(processed_images)

    def _stream_tokens(self, response_generator, start_time, stream_chunk_size=None):
        """Generator helper for streaming tokens with minimal buffering for real-time delivery
        
        Args:
            stream_chunk_size: Buffer size limit. Default 8 for real-time delivery.
                             Higher values batch more tokens before yielding.
        """
        token_count = 0
        buffer = ""
        # Use config value or default to 8 for real-time delivery
        buffer_size_limit = stream_chunk_size if stream_chunk_size is not None else 8

        for token in response_generator:
            token_count += 1
            buffer += token

            # Yield tokens aggressively for real-time streaming
            # Only buffer when we detect potential special token starts
            if len(buffer) >= buffer_size_limit:
                # Check for special token patterns that need buffering
                has_partial_special = (
                    buffer.endswith('<') or 
                    buffer.endswith('<|') or 
                    buffer.endswith('<|e') or
                    buffer.endswith('<|en') or
                    buffer.endswith('<|end') or
                    '<|' in buffer[-10:] and '|>' not in buffer[-10:]  # Incomplete special token
                )
                
                if has_partial_special:
                    # Keep buffering to see full special token
                    continue
                
                # Check if buffer contains complete special tokens to filter
                if '<|' in buffer and '|>' in buffer:
                    # Apply cleanup patterns to complete tokens
                    cleaned = buffer
                    for pattern, replacement in CLEANUP_PATTERNS:
                        cleaned = pattern.sub(replacement, cleaned)
                    if cleaned:
                        yield cleaned
                    buffer = ""
                else:
                    # No special tokens - yield immediately
                    yield buffer
                    buffer = ""
            elif token and '<' not in buffer and '|' not in buffer:
                # No special chars at all - yield token immediately for real-time delivery
                yield buffer
                buffer = ""

        # Process remaining buffer at the end
        if buffer:
            # Apply full sanitization to remaining buffer
            cleaned = buffer

            # Get model config to check reasoning_response setting
            model_config = ModelConfig.get_config(self.current_model_name) if self.current_model_name else {}
            reasoning_enabled = model_config.get("reasoning_response", "disable") == "enable"

            # Special handling for GPT-OSS Harmony format
            if GPT_OSS_CHANNEL_PATTERN.search(cleaned):
                if not reasoning_enabled:
                    # Extract only final answer channel (default behavior)
                    final_match = GPT_OSS_FINAL_PATTERN.search(cleaned)
                    if final_match:
                        cleaned = final_match.group(1).strip()
                        logger.debug("Extracted GPT-OSS final answer channel in streaming (reasoning disabled)")
                else:
                    # Keep full response with reasoning
                    logger.debug("Keeping full GPT-OSS response in streaming (reasoning enabled)")

            # Apply reasoning and cleanup patterns
            if not reasoning_enabled:
                for pattern, replacement in REASONING_PATTERNS:
                    cleaned = pattern.sub(replacement, cleaned)

            # Apply general cleanup patterns (always)
            for pattern, replacement in CLEANUP_PATTERNS:
                cleaned = pattern.sub(replacement, cleaned)

            # Apply GPT-OSS specific patterns only when reasoning is disabled
            if not reasoning_enabled:
                for pattern, replacement in GPT_OSS_CLEANUP_PATTERNS:
                    cleaned = pattern.sub(replacement, cleaned)

            if cleaned.strip():
                yield cleaned

        # Log performance after streaming completes
        gen_time = time.time() - start_time
        logger.info(f"Streamed ~{token_count} tokens in {gen_time:.2f}s using {self.current_model_name}")