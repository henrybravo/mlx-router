#!/usr/bin/env python3
"""
Comprehensive test suite for MLX Router
Tests chat templates, resource monitoring, and graceful shutdown functionality
"""

import sys
import os
from unittest.mock import MagicMock

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the components we want to test
from mlx_router.core.model_manager import CHAT_TEMPLATES
from mlx_router.config.model_config import ModelConfig
from mlx_router.core.resource_monitor import ResourceMonitor

class TestChatTemplates:
    """Test the data-driven chat template system"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        
    def assert_equal(self, actual, expected, test_name):
        if actual == expected:
            print(f"✅ {test_name}")
            self.passed += 1
        else:
            print(f"❌ {test_name}")
            print(f"   Expected: {repr(expected)}")
            print(f"   Actual:   {repr(actual)}")
            self.failed += 1
    
    def assert_contains(self, text, substring, test_name):
        if substring in text:
            print(f"✅ {test_name}")
            self.passed += 1
        else:
            print(f"❌ {test_name}")
            print(f"   Expected '{substring}' in: {repr(text)}")
            self.failed += 1
    
    def format_messages_test(self, messages, chat_template_name):
        """Simulate the new data-driven message formatting"""
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

    def test_all_config_templates(self):
        """Test all templates used in config.json"""
        print("\n🧪 Testing Chat Template System")
        print("=" * 50)
        
        test_messages = [{"role": "user", "content": "Hello, how are you?"}]
        
        # Test chatml template
        result = self.format_messages_test(test_messages, 'chatml')
        self.assert_contains(result, '<|im_start|>user', 'ChatML template has correct user start')
        self.assert_contains(result, '<|im_end|>', 'ChatML template has correct end markers')
        self.assert_contains(result, '<|im_start|>assistant', 'ChatML template has assistant start')
        
        # Test llama3 template
        result = self.format_messages_test(test_messages, 'llama3')
        self.assert_contains(result, '<|begin_of_text|>', 'Llama3 template has correct prefix')
        self.assert_contains(result, '<|start_header_id|>user<|end_header_id|>', 'Llama3 template has correct user format')
        self.assert_contains(result, '<|eot_id|>', 'Llama3 template has correct end token')
        self.assert_contains(result, '<|start_header_id|>assistant<|end_header_id|>', 'Llama3 template has assistant start')
        
        # Test deepseek template
        result = self.format_messages_test(test_messages, 'deepseek')
        self.assert_contains(result, '### Instruction:', 'DeepSeek template has correct instruction format')
        self.assert_contains(result, '### Response:', 'DeepSeek template has correct response format')
        self.assert_contains(result, 'AI programming assistant', 'DeepSeek template includes system prompt')
        
        # Test qwen template
        result = self.format_messages_test(test_messages, 'qwen')
        self.assert_contains(result, '<|im_start|>user', 'Qwen template has correct user start')
        self.assert_contains(result, '<|im_end|>', 'Qwen template has correct end markers')
        
        # Test phi4 template
        result = self.format_messages_test(test_messages, 'phi4')
        self.assert_contains(result, '<|user|>', 'Phi4 template has correct user format')
        self.assert_contains(result, '<|end|>', 'Phi4 template has correct end markers')
        self.assert_contains(result, '<|assistant|>', 'Phi4 template has assistant start')
        
        # Test generic fallback
        result = self.format_messages_test(test_messages, 'unknown_template')
        self.assert_contains(result, '<|im_start|>user', 'Generic template fallback works')
    
    def test_multi_message_conversation(self):
        """Test templates with multi-message conversations"""
        print("\n🧪 Testing Multi-Message Conversations")  
        print("=" * 50)
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        # Test llama3 with conversation
        result = self.format_messages_test(conversation, 'llama3')
        self.assert_contains(result, '<|start_header_id|>system<|end_header_id|>', 'Llama3 handles system messages')
        self.assert_contains(result, 'helpful assistant', 'Llama3 includes system content')
        
        # Test deepseek with conversation
        result = self.format_messages_test(conversation, 'deepseek')
        self.assert_contains(result, 'helpful assistant', 'DeepSeek handles custom system message')
        
        # Count instruction/response pairs
        instruction_count = result.count('### Instruction:')
        self.assert_equal(instruction_count, 2, 'DeepSeek has correct number of user messages')
    
    def test_empty_messages(self):
        """Test handling of empty or invalid messages"""
        print("\n🧪 Testing Edge Cases")
        print("=" * 50)
        
        # Test empty messages
        result = self.format_messages_test([], 'llama3')
        self.assert_equal(result, '<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n', 'Empty messages handled correctly')
        
        # Test messages with empty content
        empty_content_msgs = [{"role": "user", "content": ""}]
        result = self.format_messages_test(empty_content_msgs, 'chatml')
        self.assert_equal(result, '<|im_start|>assistant\n', 'Empty content messages skipped correctly')
        
        # Test messages without role
        no_role_msgs = [{"content": "Hello"}]
        result = self.format_messages_test(no_role_msgs, 'chatml')
        self.assert_contains(result, '<|im_start|>user', 'Missing role defaults to user')
        
    def run_all_tests(self):
        """Run all chat template tests"""
        self.test_all_config_templates()
        self.test_multi_message_conversation() 
        self.test_empty_messages()
        
        print(f"\n📊 Chat Template Test Results:")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%")
        
        return self.failed == 0

class TestResourceMonitoring:
    """Test the consolidated resource monitoring system"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def assert_equal(self, actual, expected, test_name):
        if actual == expected:
            print(f"✅ {test_name}")
            self.passed += 1
        else:
            print(f"❌ {test_name}")
            print(f"   Expected: {expected}")
            print(f"   Actual:   {actual}")
            self.failed += 1
    
    def assert_true(self, condition, test_name):
        if condition:
            print(f"✅ {test_name}")
            self.passed += 1
        else:
            print(f"❌ {test_name}")
            self.failed += 1
    
    def test_memory_info_structure(self):
        """Test memory info returns expected structure"""
        print("\n🧪 Testing Resource Monitoring")
        print("=" * 50)
        
        try:
            info = ResourceMonitor.get_memory_info()
            
            required_keys = ['total_gb', 'available_gb', 'used_gb', 'used_percent', 
                           'free_gb', 'swap_total_gb', 'swap_used_gb', 'swap_percent', 
                           'fragmentation_score']
            
            for key in required_keys:
                self.assert_true(key in info, f'Memory info contains {key}')
                self.assert_true(isinstance(info[key], (int, float)), f'{key} is numeric')
            
            # Test reasonable ranges
            self.assert_true(0 <= info['used_percent'] <= 100, 'Used percent in valid range')
            self.assert_true(0 <= info['swap_percent'] <= 100, 'Swap percent in valid range')
            self.assert_true(0 <= info['fragmentation_score'] <= 100, 'Fragmentation score in valid range')
            
        except Exception as e:
            print(f"❌ Memory info test failed: {e}")
            self.failed += 1
    
    def test_memory_pressure_levels(self):
        """Test memory pressure calculation"""
        pressure = ResourceMonitor.get_memory_pressure()
        valid_levels = ['normal', 'moderate', 'high', 'critical']
        self.assert_true(pressure in valid_levels, f'Memory pressure level is valid: {pressure}')
    
    def test_consolidated_memory_check(self):
        """Test the new consolidated check_memory_available method"""
        # Create a mock model for testing
        test_model = "test-model"
        ModelConfig.MODELS[test_model] = {"required_memory_gb": 1}  # Small requirement for testing
        
        try:
            can_load, mem_info = ResourceMonitor.check_memory_available(test_model, safety_margin=1.2)
            
            # Test return format
            self.assert_true(isinstance(can_load, bool), 'check_memory_available returns boolean')
            self.assert_true(isinstance(mem_info, dict), 'check_memory_available returns info dict')
            
            required_info_keys = ['available_gb', 'required_gb', 'effective_required', 'fragmentation_penalty']
            for key in required_info_keys:
                self.assert_true(key in mem_info, f'Memory check info contains {key}')
            
            # Test that effective_required >= required_gb due to safety margin
            self.assert_true(mem_info['effective_required'] >= mem_info['required_gb'], 
                           'Effective requirement includes safety margin')
            
        except Exception as e:
            print(f"❌ Consolidated memory check failed: {e}")
            self.failed += 1
        finally:
            # Clean up test model
            if test_model in ModelConfig.MODELS:
                del ModelConfig.MODELS[test_model]
    
    def run_all_tests(self):
        """Run all resource monitoring tests"""
        self.test_memory_info_structure()
        self.test_memory_pressure_levels()
        self.test_consolidated_memory_check()
        
        print(f"\n📊 Resource Monitoring Test Results:")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%")
        
        return self.failed == 0

class TestModelConfig:
    """Test ModelConfig functionality after improvements"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.original_models = ModelConfig.MODELS.copy()  # Backup original
    
    def assert_equal(self, actual, expected, test_name):
        if actual == expected:
            print(f"✅ {test_name}")
            self.passed += 1
        else:
            print(f"❌ {test_name}")
            print(f"   Expected: {expected}")
            print(f"   Actual:   {actual}")
            self.failed += 1
    
    def assert_true(self, condition, test_name):
        if condition:
            print(f"✅ {test_name}")
            self.passed += 1
        else:
            print(f"❌ {test_name}")
            self.failed += 1
    
    def test_config_loading(self):
        """Test the improved config loading functionality"""
        print("\n🧪 Testing ModelConfig System")
        print("=" * 50)
        
        # Test config replacement (not merging)
        test_config = {
            "test-model-1": {
                "chat_template": "llama3",
                "required_memory_gb": 5,
                "max_tokens": 2048
            },
            "test-model-2": {
                "chat_template": "deepseek", 
                "required_memory_gb": 10,
                "max_tokens": 4096
            }
        }
        
        original_count = len(ModelConfig.MODELS)
        ModelConfig.load_from_dict(test_config)
        
        # Should replace, not merge
        self.assert_equal(len(ModelConfig.MODELS), 2, 'Config loading replaces models (not merge)')
        self.assert_true('test-model-1' in ModelConfig.MODELS, 'Test model 1 loaded')
        self.assert_true('test-model-2' in ModelConfig.MODELS, 'Test model 2 loaded')
        
        # Test chat template retrieval
        template1 = ModelConfig.get_chat_template('test-model-1')
        self.assert_equal(template1, 'llama3', 'Chat template retrieved correctly')
        
        template2 = ModelConfig.get_chat_template('test-model-2')
        self.assert_equal(template2, 'deepseek', 'Different chat template retrieved correctly')
        
        # Test fallback for unknown model
        unknown_template = ModelConfig.get_chat_template('unknown-model')
        self.assert_equal(unknown_template, 'generic', 'Unknown model gets generic template fallback')
        
    def test_empty_config_handling(self):
        """Test handling of empty config"""
        ModelConfig.load_from_dict({})
        self.assert_equal(len(ModelConfig.MODELS), 0, 'Empty config results in empty models')
        
        ModelConfig.load_from_dict(None)
        self.assert_equal(len(ModelConfig.MODELS), 0, 'None config handled gracefully')
    
    def cleanup(self):
        """Restore original models"""
        ModelConfig.MODELS = self.original_models.copy()
    
    def run_all_tests(self):
        """Run all ModelConfig tests"""
        try:
            self.test_config_loading()
            self.test_empty_config_handling()
        finally:
            self.cleanup()  # Always restore original state
        
        print(f"\n📊 ModelConfig Test Results:")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%")
        
        return self.failed == 0

def main():
    """Run test suite"""
    print("🚀 MLX Router Comprehensive Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run chat template tests
    template_tester = TestChatTemplates()
    all_passed &= template_tester.run_all_tests()
    
    # Run resource monitoring tests
    resource_tester = TestResourceMonitoring()
    all_passed &= resource_tester.run_all_tests()
    
    # Run ModelConfig tests
    config_tester = TestModelConfig()
    all_passed &= config_tester.run_all_tests()
    
    # Overall results
    total_passed = template_tester.passed + resource_tester.passed + config_tester.passed
    total_failed = template_tester.failed + resource_tester.failed + config_tester.failed
    
    print("\n" + "=" * 60)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 60)
    print(f"✅ Total Passed: {total_passed}")
    print(f"❌ Total Failed: {total_failed}")
    print(f"📈 Overall Success Rate: {(total_passed/(total_passed+total_failed)*100):.1f}%")
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! System is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())