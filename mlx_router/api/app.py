#!/usr/bin/env python3
"""
FastAPI application and endpoints for MLX Router
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from mlx_router.config.model_config import ModelConfig
from mlx_router.core.resource_monitor import ResourceMonitor
from mlx_router.__version__ import VERSION

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
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(None, ge=0.01, le=2.0, description="Temperature must be between 0.01 and 2.0")
    top_p: Optional[float] = Field(None, ge=0.01, le=1.0, description="top_p must be between 0.01 and 1.0")
    top_k: Optional[int] = Field(None, ge=1, le=200, description="top_k must be between 1 and 200")
    max_tokens: Optional[int] = Field(None, ge=1, le=131072, description="max_tokens must be between 1 and 131072")
    stream: Optional[bool] = None  # Will use config default if None
    
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

    try:
        await asyncio.to_thread(model_manager.load_model, request.model)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_manager.increment_request_count()

    # Determine streaming mode: 
    # - If client explicitly sets stream=true/false, use that
    # - If client doesn't specify stream (None), use config default
    if request.stream is not None:
        stream_mode = request.stream
    else:
        # Use config default when stream is not specified
        defaults = _global_config.get('defaults', {})
        stream_mode = defaults.get('stream', False)
    
    logger.debug(f"ReqID-{request_id}: Stream mode: {stream_mode} (request={request.stream}, config_default={_global_config.get('defaults', {}).get('stream', False)})")

    if stream_mode:
        # Handle streaming request
        async def stream_generator():
            """Generate Server-Sent Events for streaming response"""
            try:
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
                yield f"data: {json.dumps(first_chunk)}\n\n"
                
                # Stream tokens from model manager
                async for token in model_manager.generate_stream_response(
                    request.messages,
                    request.max_tokens,
                    request.temperature,
                    request.top_p,
                    request.top_k,
                    getattr(request, 'min_p', None)
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
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                
                logger.info(f"ReqID-{request_id}: Successfully completed streaming request.")
                
            except Exception as e:
                logger.error(f"ReqID-{request_id}: Streaming error: {e}", exc_info=True)
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
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    try:
        content = await asyncio.to_thread(
            model_manager.generate_response,
            request.messages,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.top_k,
            getattr(request, 'min_p', None)
        )
        
        if content.startswith("ERROR:"):
            raise HTTPException(status_code=500, detail=content)
            
        response_payload = {
            "id": f"chatcmpl-{request_id}", 
            "object": "chat.completion", 
            "model": request.model, 
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }
        logger.info(f"ReqID-{request_id}: Successfully processed non-streaming request.")
        return JSONResponse(content=response_payload)
    except Exception as e:
        logger.error(f"ReqID-{request_id}: Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def set_model_manager(manager):
    """Set the global model manager instance"""
    global model_manager
    model_manager = manager

def set_global_config(config_data):
    """Set the global configuration data"""
    global _global_config
    _global_config = config_data