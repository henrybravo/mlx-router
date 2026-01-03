#!/usr/bin/env python3
"""
Content normalization for OpenAI multimodal content format support
Handles both string and array content formats for message content
"""

import logging
from typing import Union, List, Literal

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


def normalize_message_content(content: MessageContent) -> str:
    """
    Normalize message content to string format for MLX model inference.

    Handles both legacy string format and OpenAI multimodal array format.
    For Phase 1, only text content parts are processed. Image parts raise an error.

    Args:
        content: Either a string or list of content parts

    Returns:
        Concatenated text content as a single string

    Raises:
        ValueError: If image content is provided (Phase 1 limitation)
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
            if hasattr(part, 'type'):
                part_type = part.type
                part_dict = part.dict()
            elif isinstance(part, dict):
                part_type = part.get('type')
                part_dict = part
            else:
                raise TypeError(f"Content part at index {idx} must be dict or Pydantic model, got {type(part)}")

            if part_type == "text":
                if hasattr(part, 'text'):
                    text_parts.append(part.text)
                else:
                    text_parts.append(part_dict.get('text', ''))
            elif part_type == "image_url":
                raise ValueError(
                    "Image content not supported in Phase 1. "
                    "Vision model support will be available in Phase 2."
                )
            else:
                logger.warning(f"Unknown content part type: {part_type} at index {idx}")

        return "\n".join(text_parts)

    raise TypeError(
        f"Message content must be string or list, got {type(content).__name__}"
    )
