# mlx_model.py
import json
import logging
import httpx
from typing import Any, Optional, TypedDict, AsyncIterable
from typing_extensions import Unpack

from strands.models import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

logger = logging.getLogger(__name__)

class MLXModel(Model):
    """Custom model provider for MLX server with OpenAI-compatible API."""
    
    class ModelConfig(TypedDict):
        """Configuration for MLX model.
        
        Attributes:
            model_id: ID of the MLX model
            base_url: Base URL of the MLX server
            api_key: API key for authentication
            params: Model parameters (e.g., max_tokens, temperature)
        """
        model_id: str
        base_url: str
        api_key: str
        params: Optional[dict[str, Any]]

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str = "dummy",
        **model_config: Unpack[ModelConfig]
    ) -> None:
        """Initialize MLX model provider.
        
        Args:
            base_url: The base URL of your MLX server
            model_id: The model identifier
            api_key: API key (can be dummy for local servers)
            **model_config: Additional model configuration
        """
        self.config = MLXModel.ModelConfig(
            model_id=model_id,
            base_url=base_url.rstrip('/'),
            api_key=api_key,
            params=model_config.get('params', {})
        )
        logger.debug("config=<%s> | initializing", self.config)

    def update_config(self, **model_config: Unpack[ModelConfig]) -> None:
        """Update the MLX model configuration.
        
        Args:
            **model_config: Configuration overrides
        """
        self.config.update(model_config)

    def get_config(self) -> ModelConfig:
        """Get the MLX model configuration.
        
        Returns:
            The current model configuration
        """
        return self.config

    def _format_messages(self, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, str]]:
        """Convert Strands messages to OpenAI format.
        
        Args:
            messages: Strands messages format
            system_prompt: Optional system prompt
            
        Returns:
            List of messages in OpenAI format
        """
        openai_messages = []
        
        # Add system prompt if provided
        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})
        
        # Convert each message
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            
            # Handle content format - convert array to string
            if isinstance(content, list):
                text_content = ""
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif "text" in item:
                            text_content += item.get("text", "")
                    elif isinstance(item, str):
                        text_content += item
                if text_content.strip():  # Only add if there's actual content
                    openai_messages.append({"role": role, "content": text_content})
            elif isinstance(content, str) and content.strip():
                openai_messages.append({"role": role, "content": content})
        
        return openai_messages

    def _format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format request for MLX server API.
        
        Args:
            messages: Conversation messages
            tool_specs: Tool specifications (not supported by MLX server)
            system_prompt: System prompt
            
        Returns:
            Formatted request payload
        """
        openai_messages = self._format_messages(messages, system_prompt)
        
        request = {
            "model": self.config["model_id"],
            "messages": openai_messages,
            "stream": True,  # Use real streaming for optimal performance
            **self.config.get("params", {})
        }
        
        # Note: MLX server doesn't support tools, so we ignore tool_specs
        if tool_specs:
            logger.warning("Tools are not supported by MLX server, ignoring tool_specs")
        
        return request

    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterable[StreamEvent]:
        """Stream responses from MLX model using real Server-Sent Events.
        
        This method makes a streaming request to MLX Router and translates the
        Server-Sent Events into Strands StreamEvent objects in real-time,
        providing optimal performance and immediate token delivery.
        
        Args:
            messages: List of conversation messages
            tool_specs: Optional list of available tools (ignored for MLX)
            system_prompt: Optional system prompt
            **kwargs: Additional keyword arguments
            
        Yields:
            StreamEvent objects translated from MLX Router's SSE stream
        """
        logger.debug("messages=<%s> tool_specs=<%s> system_prompt=<%s> | formatting streaming request",
                     messages, tool_specs, system_prompt)
        
        # Format the request (now with stream=True)
        request = self._format_request(messages, tool_specs, system_prompt)
        logger.debug("request=<%s> | starting SSE stream", request)
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config['api_key']}"
        }
        
        try:
            # Make streaming API call
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.config['base_url']}/chat/completions",
                    json=request,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    
                    logger.debug("SSE stream established | processing events")
                    
                    # Track if we've sent messageStart
                    message_started = False
                    content_chunks = []
                    
                    # Process Server-Sent Events line by line
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                            
                        # Parse SSE format: "data: {json}" or "data: [DONE]"
                        if line.startswith("data: "):
                            data_content = line[6:]  # Remove "data: " prefix
                            
                            # Check for stream termination
                            if data_content == "[DONE]":
                                logger.debug("SSE stream completed with [DONE]")
                                break
                            
                            try:
                                # Parse the JSON chunk
                                chunk = json.loads(data_content)
                                
                                # Extract delta information
                                choices = chunk.get("choices", [])
                                if not choices:
                                    continue
                                    
                                choice = choices[0]
                                delta = choice.get("delta", {})
                                finish_reason = choice.get("finish_reason")
                                
                                # Handle role (messageStart)
                                if "role" in delta and not message_started:
                                    yield {
                                        "messageStart": {
                                            "role": delta["role"]
                                        }
                                    }
                                    message_started = True
                                    logger.debug("messageStart event sent")
                                
                                # Handle content chunks
                                if "content" in delta:
                                    content = delta["content"]
                                    content_chunks.append(content)
                                    
                                    yield {
                                        "contentBlockDelta": {
                                            "delta": {
                                                "text": content
                                            }
                                        }
                                    }
                                    logger.debug(f"content delta sent: {repr(content[:20])}...")
                                
                                # Handle completion
                                if finish_reason:
                                    yield {
                                        "messageStop": {
                                            "stopReason": "end_turn" if finish_reason == "stop" else finish_reason
                                        }
                                    }
                                    logger.debug(f"messageStop event sent: {finish_reason}")
                                    
                                    # Add metadata (token usage not available in streaming, but we can estimate)
                                    full_content = ''.join(content_chunks)
                                    estimated_output_tokens = len(full_content.split())
                                    
                                    yield {
                                        "metadata": {
                                            "usage": {
                                                "inputTokens": 0,  # Not available in streaming
                                                "outputTokens": estimated_output_tokens,
                                                "totalTokens": estimated_output_tokens
                                            },
                                            "metrics": {
                                                "latencyMs": 0  # Real-time streaming
                                            }
                                        }
                                    }
                                    
                                    break
                                    
                            except json.JSONDecodeError as je:
                                logger.warning(f"Failed to parse SSE JSON chunk: {data_content[:100]}... | {je}")
                                continue
                    
                    logger.debug("SSE stream processing complete")
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 413:  # Payload too large
                logger.error("Context window overflow: %s", e.response.text)
                raise RuntimeError("Context window overflow - request too large") from e
            else:
                logger.error("HTTP error: %s - %s", e.response.status_code, e.response.text)
                raise
        except Exception as e:
            logger.error("Error in SSE streaming from MLX server: %s", str(e))
            raise

    async def structured_output(
        self,
        output_model,
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ):
        """Get structured output (not supported by MLX server).
        
        Args:
            output_model: The Pydantic model for structured output
            prompt: The prompt messages
            system_prompt: Optional system prompt
            **kwargs: Additional arguments
            
        Raises:
            NotImplementedError: MLX server doesn't support structured output
        """
        raise NotImplementedError(
            "Structured output is not supported by the MLX server. "
            "MLX server doesn't support function calling or tool use required for structured output."
        )


# Usage example
def create_mlx_agent():
    """Create an agent using the MLX model provider."""
    from agents.strands.mlx_model import Agent
    from strands_tools import calculator
    
    # Initialize MLX model
    mlx_model = MLXModel(
        base_url="http://host.docker.internal:8888/v1",
        model_id="mlx-community/Qwen3-30B-A3B-8bit",
        api_key="strands-key",
        params={
            "max_tokens": 1000,
            "temperature": 0.7,
        }
    )
    
    # Create agent (without tools initially since MLX server doesn't support them)
    agent = Agent(model=mlx_model)
    
    return agent

# Test the implementation
async def test_mlx_agent():
    """Test the MLX agent."""
    agent = create_mlx_agent()
    
    # Test basic conversation
    response = await agent.invoke_async("What is 2+2? Please explain your reasoning.")
    print("Response:", response)
    
    return response

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_mlx_agent())