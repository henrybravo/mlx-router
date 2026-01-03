#!/usr/bin/env python3
"""Regex patterns for message cleanup and formatting
"""

import re

# Pre-compile regex patterns for better performance
# General cleanup patterns - always applied
CLEANUP_PATTERNS = [
    # Llama 3.x format tokens
    (re.compile(r'<\|start_header_id\|>.*?<\|end_header_id\|>', re.DOTALL), ''),
    (re.compile(r'<\|eot_id\|>'), ''),
    (re.compile(r'<\|begin_of_text\|>'), ''),
    (re.compile(r'<\|end_of_text\|>'), ''),

    # ChatML / Qwen format tokens
    (re.compile(r'<\|im_start\|>system\n.*?\n<\|im_end\|>', re.DOTALL), ''),
    (re.compile(r'<\|im_start\|>user\n.*?\n<\|im_end\|>', re.DOTALL), ''),
    (re.compile(r'<\|im_start\|>assistant\n'), ''),
    (re.compile(r'<\|im_end\|>'), ''),

    # Phi-4 format tokens
    (re.compile(r'<\|user\|>'), ''),
    (re.compile(r'<\|assistant\|>'), ''),
    (re.compile(r'<\|system\|>'), ''),
    (re.compile(r'<\|end\|>'), ''),

    # Role prefixes
    (re.compile(r'^(Assistant|User|System):\s*', re.MULTILINE), ''),
]

# GPT-OSS specific patterns - only applied when reasoning_response is "disable"
GPT_OSS_CLEANUP_PATTERNS = [
    (re.compile(r'<\|start\|>'), ''),
    (re.compile(r'<\|channel\|>[^\n]*'), ''),  # Remove channel directives
    (re.compile(r'<\|message\|>'), ''),
    (re.compile(r'<\|end\|>'), ''),

    # Catch-all for any remaining special tokens
    (re.compile(r'<\|[a-zA-Z0-9_]+\|>'), ''),
    (re.compile(r'^\s*<\|.*?\|>\s*\n?', re.MULTILINE), ''),
]

# Pattern for detecting reasoning/meta-commentary sections in responses
REASONING_PATTERNS = [
    # Qwen3 thinking blocks
    (re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE), ''),

    # GPT-OSS style reasoning blocks (legacy - kept for backward compatibility)
    (re.compile(r'We have to produce.*?Thus answer\.', re.DOTALL | re.IGNORECASE), ''),
    (re.compile(r'We (?:must|can|need to|should).*?(?:Thus|Therefore|So)[,:]?\s*(?:answer|respond)', re.DOTALL | re.IGNORECASE), ''),

    # Generic meta-commentary patterns
    (re.compile(r'(?:^|\n)\s*(?:The user asks?|We need to respond|Provide an answer).*?(?:\n\n|Thus answer)', re.DOTALL | re.IGNORECASE), ''),
]

# GPT-OSS Harmony format channel extraction
# Pattern to extract only the final answer channel content
GPT_OSS_FINAL_PATTERN = re.compile(
    r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)',
    re.DOTALL | re.IGNORECASE
)

# Pattern to detect if response contains harmony format channels
GPT_OSS_CHANNEL_PATTERN = re.compile(r'<\|channel\|>', re.IGNORECASE)

# Pattern to replace 3+ newlines with single newline (common in model responses)
NEWLINE_PATTERN = re.compile(r'\n{3,}')
