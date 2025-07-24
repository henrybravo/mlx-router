#!/usr/bin/env python3
"""
FastAPI application and endpoints for MLX Router
"""

import asyncio
import hashlib
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from mlx_router.config.model_config import ModelConfig
from mlx_router.core.resource_monitor import ResourceMonitor
from mlx_router.__version__ import VERSION

logger = logging.getLogger(__name__)

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
    max_tokens: Optional[int] = Field(None, ge=1, le=32768, description="max_tokens must be between 1 and 32768")
    stream: Optional[bool] = False
    
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

    if request.stream:
        raise HTTPException(status_code=501, detail="Streaming not yet implemented in 2.0.0")

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