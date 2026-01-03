#!/usr/bin/env python3
"""
Content normalization for OpenAI multimodal content format support
Handles both string and array content formats for message content
"""

import base64
import io
import logging
import re
from typing import Union, List, Literal, Optional, Tuple

import requests
from PIL import Image

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ImageUrlDetail(BaseModel):
    """Image URL with optional detail level for vision models"""
    url: str = Field(description="URL or base64 data URI for image")
    detail: Literal["auto", "low", "high"] = Field(
        default="auto",
        description="Detail level: auto (default), low (speed), or high (quality)"
    )


class TextContentPart(BaseModel):
    """Text content part in multimodal messages"""
    type: Literal["text"] = Field(default="text", description="Content type identifier")
    text: str = Field(description="Text content")


class ImageUrlContentPart(BaseModel):
    """Image URL content part in multimodal messages (Phase 2: vision models)"""
    type: Literal["image_url"] = Field(default="image_url", description="Content type identifier")
    image_url: ImageUrlDetail = Field(description="Image URL and detail settings")


ContentPart = Union[TextContentPart, ImageUrlContentPart]
MessageContent = Union[str, List[ContentPart]]


def decode_base64_image(base64_data: str) -> Optional[Image.Image]:
    """
    Decode base64 image data to PIL Image.

    Args:
        base64_data: Base64 encoded image data (with or without data URI prefix)

    Returns:
        PIL Image object, or None if decoding fails

    Raises:
        ValueError: If base64 data is invalid
    """
    try:
        data_uri_match = re.match(r'^data:image/([a-zA-Z]+);base64,(.+)$', base64_data)
        if data_uri_match:
            base64_str = data_uri_match.group(2)
        else:
            base64_str = base64_data

        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        logger.debug("Successfully decoded base64 image")
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise ValueError(f"Invalid base64 image data: {str(e)}")


def fetch_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """
    Fetch image from URL and return as PIL Image.

    Args:
        url: Image URL (http/https)
        timeout: Request timeout in seconds

    Returns:
        PIL Image object

    Raises:
        ValueError: If URL is invalid or image cannot be fetched
    """
    try:
        logger.debug(f"Fetching image from URL: {url}")
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        logger.debug("Successfully fetched image from URL")
        return image
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch image from URL: {e}")
        raise ValueError(f"Failed to fetch image from URL: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to process image from URL: {e}")
        raise ValueError(f"Failed to process image: {str(e)}")


def preprocess_image_for_vision(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Preprocess image for vision model input.

    Args:
        image: PIL Image object
        target_size: Target size as (height, width)

    Returns:
        Preprocessed PIL Image object
    """
    try:
        image = image.convert('RGB')
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        logger.debug(f"Image preprocessed to size {target_size}")
        return image
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")



def normalize_message_content(content: MessageContent, support_vision: bool = False) -> str:
    """
    Normalize message content to string format for MLX model inference.

    Handles both legacy string format and OpenAI multimodal array format.
    For Phase 1, only text content parts are processed. Image parts raise an error.
    For Phase 2, image parts are extracted for vision model processing.

    Args:
        content: Either a string or list of content parts
        support_vision: If True, extract image data for vision models

    Returns:
        Concatenated text content as a single string

    Raises:
        ValueError: If image content is provided without vision support
        TypeError: If content is neither string nor list

    Examples:
        >>> normalize_message_content("Hello world")
        'Hello world'

        >>> normalize_message_content([{"type": "text", "text": "Hello"}])
        'Hello'

        >>> normalize_message_content([
        ...     {"type": "text", "text": "First"},
        ...     {"type": "text", "text": "Second"}
        ... ])
        'First\\nSecond'
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for idx, part in enumerate(content):
            if isinstance(part, TextContentPart):
                text_parts.append(part.text)
            elif isinstance(part, ImageUrlContentPart):
                if support_vision:
                    logger.warning(f"Image content at index {idx} requires vision model support")
                else:
                    raise ValueError(
                        "Image content requires vision model support (Phase 2). "
                        "Please use a vision-enabled model or remove image content."
                    )
            elif isinstance(part, dict):
                part_type = part.get('type')
                if part_type == "text":
                    text_parts.append(part.get('text', ''))
                elif part_type == "image_url" and support_vision:
                    logger.warning(f"Image content at index {idx} requires vision model support")
                elif part_type == "image_url" and not support_vision:
                    raise ValueError(
                        "Image content requires vision model support (Phase 2). "
                        "Please use a vision-enabled model or remove image content."
                    )
                else:
                    logger.warning(f"Unknown content part type: {part_type} at index {idx}")
            else:
                raise TypeError(f"Content part at index {idx} must be dict or Pydantic model, got {type(part)}")

        return "\n".join(text_parts)

    raise TypeError(
        f"Message content must be string or list, got {type(content).__name__}"
    )
