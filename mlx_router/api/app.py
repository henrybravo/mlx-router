#!/usr/bin/env python3
"""
FastAPI application and endpoints for MLX Router
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from mlx_router.config.model_config import ModelConfig
from mlx_router.core.resource_monitor import ResourceMonitor
from mlx_router.core.content import MessageContent, normalize_message_content, extract_images_from_content
from mlx_router.core.manager import MLXModelManager
from mlx_router.__version__ import VERSION

VISION_ENABLED_MODELS = [
    'vl', 'vision', 'llava', 'qwen-vl', 'nvl'
]

logger = logging.getLogger(__name__)

# Global config store for defaults
_global_config = {}

app = FastAPI(title="MLX Model Router", version=VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Pydantic Models for Request/Response Validation
class ChatMessage(BaseModel):
    role: str
    content: MessageContent

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(None, ge=0.01, le=2.0, description="Temperature must be between 0.01 and 2.0")
    top_p: Optional[float] = Field(None, ge=0.01, le=1.0, description="top_p must be between 0.01 and 1.0")
    top_k: Optional[int] = Field(None, ge=1, le=200, description="top_k must be between 1 and 200")
    max_tokens: Optional[int] = Field(None, ge=1, le=131072, description="max_tokens must be between 1 and 131072")
    stream: Optional[bool] = None  # Will use config default if None
    tools: Optional[List[Dict[str, Any]]] = None  # Function calling tools
    tool_choice: Optional[str] = None  # "none", "auto", or specific tool name
    min_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="min_p must be between 0.0 and 1.0")
    
    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages cannot be empty")
        for msg in v:
            if not msg.role or not msg.content:
                raise ValueError("each message must have both role and content")
            if msg.role not in ['system', 'user', 'assistant']:
                raise ValueError("message role must be 'system', 'user', or 'assistant'")
        return v

def normalize_request_messages(messages: List[ChatMessage], model_id: str) -> tuple[List[Dict[str, str]], List[str]]:
    """
    Normalize all message contents from array/string format to string format.

    This ensures backward compatibility with existing code while supporting new
    OpenAI multimodal array format.

    Args:
        messages: List of ChatMessage objects with content as str or list[ContentPart]
        model_id: Model ID for capability detection

    Returns:
        Tuple of (normalized messages list, list of image URLs/base64 data)
    """
    normalized = []
    all_images = []
    # Check config first for explicit supports_vision setting, then fall back to keyword detection
    model_config = ModelConfig.get_config(model_id)
    support_vision = model_config.get("supports_vision", False) or \
        any(keyword.lower() in model_id.lower() for keyword in VISION_ENABLED_MODELS)

    for msg in messages:
        try:
            content_str = normalize_message_content(msg.content, support_vision=support_vision)
            normalized.append({"role": msg.role, "content": content_str})
            # Extract images from this message
            if support_vision:
                images = extract_images_from_content(msg.content)
                all_images.extend(images)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid message content: {str(e)}"
            )
    return normalized, all_images

model_manager = None

# API Endpoints
@app.get("/v1/models", summary="List Available Models")
async def list_models():
    """Lists all models with full details"""
    models_data = []
    for model_id in model_manager.available_models:
        config = ModelConfig.get_config(model_id)
        can_load, reason = ResourceMonitor.should_defer_model_load(model_id)
        models_data.append({
            "id": model_id, "object": "model", "created": int(time.time()), "owned_by": "mlx-router",
            "memory_requirements": {
                "required_gb": config.get("required_memory_gb", 8),
                "can_load_now": not can_load,
                "load_status_reason": "Available for loading" if not can_load else reason,
            },
            "parameters": {
                "max_tokens": config.get("max_tokens", 4096),
                "chat_template": config.get("chat_template", "generic"),
            }
        })
    mem_info = ResourceMonitor.get_memory_info()
    return {
        "object": "list", "data": models_data,
        "memory_status": {
            "available_gb": round(mem_info['available_gb'], 2),
            "pressure": ResourceMonitor.get_memory_pressure(),
            "recommended_model": model_manager.get_recommended_model()
        }
    }

@app.get("/health", summary="Get Server Health", tags=["Health"])
@app.get("/v1/health", summary="Get Server Health", tags=["Health"])
async def health_check():
    """Provides a detailed health check matching"""
    return model_manager.get_health_metrics()

@app.post("/v1/chat/completions", summary="Create a Chat Completion", tags=["Chat"])
async def create_chat_completion(request: ChatCompletionRequest):
    """Generates a model response for the given chat conversation."""
    request_id = hashlib.md5(f"{time.time()}{request.model}".encode()).hexdigest()[:12]
    logger.info(f"ReqID-{request_id}: Received chat request for model '{request.model}'")

    normalized_messages, images = normalize_request_messages(request.messages, request.model)

    # Determine streaming mode early for error handling
    if request.stream is not None:
        stream_mode = request.stream
    else:
        # Use config default when stream is not specified
        defaults = _global_config.get('defaults', {})
        stream_mode = defaults.get('stream', False)

    logger.debug(f"ReqID-{request_id}: Stream mode: {stream_mode} (request={request.stream}, config_default={_global_config.get('defaults', {}).get('stream', False)})")

    try:
        await asyncio.to_thread(model_manager.load_model, request.model)
    except (ValueError, RuntimeError) as e:
        logger.error(f"ReqID-{request_id}: Model loading error: {e}")
        error_message = str(e)
        if stream_mode:
            # Get streaming format from config for error response
            streaming_format = _global_config.get('defaults', {}).get('streaming_format', 'sse')

            # For streaming, return error in stream format
            async def error_stream():
                error_chunk = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"ERROR: {error_message}"},
                        "finish_reason": "stop"
                    }]
                }
                if streaming_format == "json_array":
                    yield json.dumps([error_chunk])
                elif streaming_format == "json_lines":
                    yield f"{json.dumps(error_chunk)}\n"
                    yield '{"done": true}\n'
                else:
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            if streaming_format == "json_array":
                media_type = "application/json"
            elif streaming_format == "json_lines":
                media_type = "application/json"
            else:
                media_type = "text/plain"
            return StreamingResponse(error_stream(), media_type=media_type)
        else:
            # For non-streaming, return JSON error
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": error_message,
                        "type": "model_load_error",
                        "code": "model_unavailable"
                    }
                }
            )

    model_manager.increment_request_count()

    if stream_mode:
        # Get streaming format from config
        streaming_format = _global_config.get('defaults', {}).get('streaming_format', 'sse')
        logger.debug(f"ReqID-{request_id}: Using streaming format: {streaming_format}")

        # Handle streaming request
        async def stream_generator():
            """Generate streaming response in configured format"""
            try:
                if streaming_format == "json_array":
                    # For json_array, collect all content and send as complete response
                    full_content = []
                    current_content = ""

                    # Stream tokens from model manager and collect them
                    async for token in model_manager.generate_stream_response(
                        normalized_messages,
                        request.tools,
                        request.max_tokens,
                        request.temperature,
                        request.top_p,
                        request.top_k,
                        getattr(request, 'min_p', None),
                        images=images
                    ):
                        if token.startswith("ERROR:"):
                            # Handle error
                            full_content = f"\n\n{token}"
                            break
                        else:
                            current_content += token

                    # Send the complete response as a single JSON object (like non-streaming)
                    response_obj = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": current_content
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                    yield json.dumps(response_obj)

                else:
                    # Original streaming logic for sse and json_lines
                    # Initial chunk with role
                    first_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None
                        }]
                    }

                    if streaming_format == "json_lines":
                        yield f"{json.dumps(first_chunk)}\n"
                    else:  # sse
                        yield f"data: {json.dumps(first_chunk)}\n\n"

                    # Stream tokens from model manager
                    async for token in model_manager.generate_stream_response(
                        normalized_messages,
                        request.tools,
                        request.max_tokens,
                        request.temperature,
                        request.top_p,
                        request.top_k,
                        getattr(request, 'min_p', None),
                        images=images
                    ):
                        if token.startswith("ERROR:"):
                            # Handle error in stream
                            error_chunk = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": f"\n\n{token}"},
                                    "finish_reason": "stop"
                                }]
                            }
                            if streaming_format == "json_lines":
                                yield f"{json.dumps(error_chunk)}\n"
                            else:
                                yield f"data: {json.dumps(error_chunk)}\n\n"
                            break
                        else:
                            # Normal token chunk
                            chunk = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": token},
                                    "finish_reason": None
                                }]
                            }
                            if streaming_format == "json_lines":
                                yield f"{json.dumps(chunk)}\n"
                            else:
                                yield f"data: {json.dumps(chunk)}\n\n"

                    # Final chunk
                    final_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    if streaming_format == "json_lines":
                        yield f"{json.dumps(final_chunk)}\n"
                        yield '{"done": true}\n'  # JSON lines format done marker
                    else:
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"

                logger.info(f"ReqID-{request_id}: Successfully completed streaming request.")

            except Exception as e:
                logger.error(f"ReqID-{request_id}: Streaming error: {e}", exc_info=True)
                # Send error response
                if streaming_format == "json_array":
                    error_response = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"ERROR: {str(e)}"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                    yield json.dumps(error_response)
                else:
                    # Send error in stream format
                    error_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"\n\nERROR: {str(e)}"},
                            "finish_reason": "stop"
                        }]
                    }
                    if streaming_format == "json_lines":
                        yield f"{json.dumps(error_chunk)}\n"
                        yield '{"done": true}\n'
                    else:
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"

        # Set media type based on format
        if streaming_format == "json_array":
            media_type = "application/json"
        elif streaming_format == "json_lines":
            media_type = "application/json"
        else:  # sse
            media_type = "text/plain"
        return StreamingResponse(stream_generator(), media_type=media_type)

    try:
        response = await asyncio.to_thread(
            model_manager.generate_response,
            normalized_messages,
            request.tools,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
            getattr(request, 'min_p', None),
            images
        )
        
        # Handle different response types
        if response.get('type') == 'error':
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": response['content'],
                        "type": "generation_error",
                        "code": "internal_error"
                    }
                }
            )
        elif response.get('type') == 'tool_calls':
            # Tool calls response
            message = {
                "role": "assistant",
                "content": response['content'],
                "tool_calls": response['tool_calls']
            }
            finish_reason = "tool_calls"
        else:
            # Standard text response
            message = {
                "role": "assistant",
                "content": response['content']
            }
            finish_reason = response.get('finish_reason', 'stop')

        response_payload = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        logger.info(f"ReqID-{request_id}: Successfully processed non-streaming request.")
        return JSONResponse(content=response_payload)
    except Exception as e:
        logger.error(f"ReqID-{request_id}: Generation error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "generation_error",
                    "code": "internal_error"
                }
            }
        )

def set_model_manager(manager):
    """Set the global model manager instance"""
    global model_manager
    model_manager = manager

def set_global_config(config_data):
    """Set the global configuration data"""
    global _global_config
    _global_config = config_data