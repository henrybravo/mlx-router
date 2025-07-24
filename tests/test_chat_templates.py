#!/usr/bin/env python3
"""
Quick test to verify chat template compatibility after refactoring
"""

# Simulate the template functionality
CHAT_TEMPLATES = {
    'llama3': {
        'prefix': '<|begin_of_text|>',
        'role_start': '<|start_header_id|>{role}<|end_header_id|>\n\n',
        'role_end': '<|eot_id|>',
        'assistant_start': '<|start_header_id|>assistant<|end_header_id|>\n\n',
        'system_default': None
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
        'system_default': None
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
    }
}

def format_messages_test(messages, chat_template_name):
    """Test the new data-driven message formatting"""
    template = CHAT_TEMPLATES.get(chat_template_name, CHAT_TEMPLATES['generic'])
    
    # Start with prefix
    prompt = template.get('prefix', '')
    
    # Add system default if needed and no system message exists
    has_system = any(msg.get("role", "").lower() == "system" for msg in messages)
    if not has_system and template.get('system_default'):
        if template.get('system_format'):
            prompt += template['system_format'].format(content=template['system_default'])
        else:
            prompt += template['system_default']
    
    # Process messages
    for msg in messages:
        role = msg.get("role", "user").lower()
        content = msg.get("content", "").strip()
        
        if not content and role != "system":
            continue
        
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

# Test all templates used in config.json
test_messages = [{"role": "user", "content": "Hello, how are you?"}]

print("Testing chat template compatibility:")
print("=" * 50)

for template_name in ['chatml', 'llama3', 'deepseek', 'qwen', 'phi4']:
    print(f"\n{template_name.upper()} Template:")
    result = format_messages_test(test_messages, template_name)
    print(repr(result))
    print("Formatted output:")
    print(result)
    print("-" * 30)