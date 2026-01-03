#!/usr/bin/env python3
"""
Test script for Chandra-8bit CV/OCR model with file upload

This script demonstrates how to:
1. Load a file from local filesystem
2. Send it to MLX Router's chat completions endpoint
3. Use Chandra-8bit CV/OCR model to process file
4. Show how to add a config.json entry for this model

Usage:
    python tests/test_vision_model.py --image-file path/to/image.png

Config.json example for Chandra-8bit:
{
  "models": {
    "mlx-community/chandra-8bit": {
      "max_tokens": 8192,
      "temp": 0.7,
      "top_p": 0.9,
      "top_k": 50,
      "min_p": 0.05,
      "chat_template": "generic",
      "required_memory_gb": 4,
      "supports_tools": False,
      "memory_pressure_max_tokens": {
        "normal": 8192,
        "moderate": 4096,
        "high": 2048,
        "critical": 1024
      }
    }
  }
}
"""

import argparse
import base64
import json
import logging
import os
import requests

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8800/v1/chat/completions"
DEFAULT_API_KEY = "dummy-key"


def encode_image_to_base64(image_path):
    """Encode an image file to base64 data URI format"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_data = base64.b64encode(image_data).decode('utf-8')

        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.png':
            mime_type = 'image/png'
        elif ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif ext == '.webp':
            mime_type = 'image/webp'
        elif ext == '.bmp':
            mime_type = 'image/bmp'
        else:
            mime_type = 'image/png'

        return f"data:{mime_type};base64,{base64_data}"

    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise


def create_test_image():
    """Create a simple test image with text"""
    try:
        img = PILImage.new('RGB', (200, 100), color='white')

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', 20)
        except:
            font = ImageFont.load_default()

        draw.text((10, 30), "Test Image", fill='black', font=font)
        draw.text((10, 60), "For Chandra", fill='black', font=font)

        test_image_path = '/tmp/test_chandra_image.png'
        img.save(test_image_path, 'PNG')
        logger.info(f"Created test image: {test_image_path}")
        return test_image_path

    except Exception as e:
        logger.error(f"Error creating test image: {e}")
        raise


def send_chat_request(api_url, api_key, image_base64, text_prompt):
    """Send chat completion request to MLX Router with image"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "mlx-community/chandra-8bit",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": text_prompt
                        }
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }

        logger.info(f"Sending request to {api_url}")
        response = requests.post(api_url, json=payload, headers=headers, timeout=60)

        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description='Test Chandra-8bit CV/OCR model with file upload')
    parser.add_argument('--image-file', type=str, help='Path to image file (default: create test image)')
    parser.add_argument('--text', type=str, default='What is in this image? Describe it in detail.', help='Text prompt for image')
    parser.add_argument('--api-url', type=str, default=DEFAULT_API_URL, help='MLX Router API URL')
    parser.add_argument('--api-key', type=str, default=DEFAULT_API_KEY, help='API key (required but ignored)')
    parser.add_argument('--create-image', action='store_true', help='Create a test image instead of loading from file')

    args = parser.parse_args()

    if args.create_image:
        logger.info("Creating test image...")
        image_path = create_test_image()
    elif args.image_file:
        image_path = args.image_file
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return
        logger.info(f"Using image file: {image_path}")
    else:
        logger.info("Creating test image...")
        image_path = create_test_image()

    try:
        logger.info("Encoding image to base64...")
        image_base64 = encode_image_to_base64(image_path)
        logger.info(f"Image encoded successfully (size: {len(image_base64)} chars)")

        logger.info("Sending chat completion request...")
        response = send_chat_request(args.api_url, args.api_key, image_base64, args.text)

        logger.info("=" * 60)
        logger.info("Response:")
        logger.info(json.dumps(response, indent=2))
        logger.info("=" * 60)

        if 'choices' in response and len(response['choices']) > 0:
            assistant_message = response['choices'][0]['message']['content']
            logger.info(f"\nAssistant Response:\n{assistant_message}")

        logger.info("\n" + "=" * 60)
        logger.info("Config.json entry for Chandra-8bit:")
        logger.info("=" * 60)
        config_example = {
            "models": {
                "mlx-community/chandra-8bit": {
                    "max_tokens": 8192,
                    "temp": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "min_p": 0.05,
                    "chat_template": "generic",
                    "required_memory_gb": 4,
                    "supports_tools": False,
                    "memory_pressure_max_tokens": {
                        "normal": 8192,
                        "moderate": 4096,
                        "high": 2048,
                        "critical": 1024
                    }
                }
            }
        }
        logger.info(json.dumps(config_example, indent=2))
        logger.info("=" * 60)

        logger.info("\nTo use this model:")
        logger.info("1. Add the above config to your config.json")
        logger.info("2. Ensure Chandra-8bit is available in your model directory")
        logger.info("3. Restart MLX Router if needed")

    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        logger.info(f"Error type: {type(e).__name__}")
        return 1


if __name__ == "__main__":
    exit(main())
