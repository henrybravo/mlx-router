#!/usr/bin/env python3
"""Chat template definitions for various model formats
"""

CHAT_TEMPLATES = {
    'llama3': {
        'prefix': '<|begin_of_text|>',
        'role_start': '<|start_header_id|>{role}<|end_header_id|>\n\n',
        'role_end': '<|eot_id|>',
        'assistant_start': '<|start_header_id|>assistant<|end_header_id|>\n\n',
        'system_default': None,
        'tools_system_prompt': '''You have access to the following tools. To use a tool, respond with a JSON object in <tool_call> tags.

<tools>
{tools_json}
</tools>

Example response:
<tool_call>
[
  {{
    "name": "get_weather", 
    "arguments": {{
      "location": "San Francisco, CA"
    }}
  }}
]
</tool_call>

If you need to use a tool, respond ONLY with the tool_call. If no tool is needed, respond normally.'''
    },
    'deepseek': {
        'prefix': '',
        'system_format': '{content}\n',
        'user_format': '### Instruction:\n{content}\n',
        'assistant_format': '### Response:\n{content}\n',
        'assistant_start': '### Response:\n',
        'system_default': 'You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.\n'
    },
    'qwen': {
        'prefix': '',
        'role_format': '<|im_start|>{role}\n{content}<|im_end|>\n',
        'assistant_start': '<|im_start|>assistant\n',
        'system_default': None,
        'tools_system_prompt': '''You have access to the following tools. To use a tool, respond with a JSON object in <tool_call> tags.

<tools>
{tools_json}
</tools>

Example response:
<tool_call>
[
  {{
    "name": "get_weather", 
    "arguments": {{
      "location": "San Francisco, CA"
    }}
  }}
]
</tool_call>

If you need to use a tool, respond ONLY with the tool_call. If no tool is needed, respond normally.'''
    },
    'chatml': {
        'prefix': '',
        'role_format': '<|im_start|>{role}\n{content}<|im_end|>\n',
        'assistant_start': '<|im_start|>assistant\n',
        'system_default': None
    },
    'phi4': {
        'prefix': '',
        'user_format': '<|user|>\n{content}<|end|>\n',
        'assistant_format': '<|assistant|>\n{content}<|end|>\n',
        'system_format': '<|system|>\n{content}<|end|>\n',
        'assistant_start': '<|assistant|>\n',
        'system_default': None
    },
    'generic': {
        'prefix': '',
        'role_format': '<|im_start|>{role}\n{content}<|im_end|>\n',
        'assistant_start': '<|im_start|>assistant\n',
        'system_default': None
    },
    'gpt-oss': {
        'prefix': '<|startoftext|>',
        'use_tokenizer_template': True,  # Flag to use tokenizer.apply_chat_template if available
        'role_format': '<|start|>{role}<|message|>\n{content}\n<|end|>',
        'assistant_start': '<|start|>assistant<|message|>\n',
        'system_default': None,
        'eos_token': '<|return|>'
    }
}

