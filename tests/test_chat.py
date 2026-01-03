#!/usr/bin/env python3
"""
Test suite for OpenAI multimodal content format support (Phase 1: text-only)
Tests string format, array format, and mixed content support
"""

import sys
import os
import json

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mlx_router.core.content import (
    normalize_message_content,
    TextContentPart,
    ImageUrlContentPart,
    ImageUrlDetail,
    MessageContent
)


class TestContentNormalization:
    """Test content normalization for Phase 1 (text-only support)"""

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

    def assert_raises(self, exception_type, func, test_name):
        try:
            func()
            print(f"❌ {test_name}")
            print(f"   Expected {exception_type.__name__} to be raised")
            self.failed += 1
        except exception_type:
            print(f"✅ {test_name}")
            self.passed += 1
        except Exception as e:
            print(f"❌ {test_name}")
            print(f"   Expected {exception_type.__name__}, got {type(e).__name__}: {e}")
            self.failed += 1

    def test_tc1_string_content(self):
        """TC1: String content (backward compatibility)"""
        print("\n🧪 TC1: String Content (Backward Compatibility)")
        print("=" * 50)

        # Legacy string format
        content = "What existed first, the chicken or the egg?"
        result = normalize_message_content(content)
        self.assert_equal(result, content, "String content returned unchanged")

    def test_tc2_array_text_only(self):
        """TC2: Array format with text content only"""
        print("\n🧪 TC2: Array Format (Text Only)")
        print("=" * 50)

        # Array format with single text part
        content = [{"type": "text", "text": "Hello, how are you?"}]
        result = normalize_message_content(content)
        self.assert_equal(result, "Hello, how are you?", "Single text part extracted correctly")

        # Array format with system message
        content = [{"type": "text", "text": "You are helpful."}]
        result = normalize_message_content(content)
        self.assert_equal(result, "You are helpful.", "System message text extracted correctly")

    def test_tc3_mixed_format(self):
        """TC3: Mixed format (string and array in same request)"""
        print("\n🧪 TC3: Mixed Format")
        print("=" * 50)

        # This test verifies that both formats can coexist in the same request
        # The normalization happens per-message, so we test both

        # String format message
        msg1_content = "You are helpful."
        result1 = normalize_message_content(msg1_content)
        self.assert_equal(result1, msg1_content, "String message normalized correctly")

        # Array format message
        msg2_content = [{"type": "text", "text": "What existed first?"}]
        result2 = normalize_message_content(msg2_content)
        self.assert_equal(result2, "What existed first?", "Array message normalized correctly")

    def test_tc4_multiple_text_parts(self):
        """TC4: Multiple text parts in array format"""
        print("\n🧪 TC4: Multiple Text Parts")
        print("=" * 50)

        # Multiple text parts should be joined with newlines
        content = [
            {"type": "text", "text": "First paragraph."},
            {"type": "text", "text": "Second paragraph."}
        ]
        result = normalize_message_content(content)
        expected = "First paragraph.\nSecond paragraph."
        self.assert_equal(result, expected, "Multiple text parts joined with newlines")

    def test_tc5_image_content_rejected(self):
        """TC5: Image content raises error (Phase 1 limitation)"""
        print("\n🧪 TC5: Image Content Rejection (Phase 1)")
        print("=" * 50)

        # Image content should raise ValueError
        content = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]

        self.assert_raises(
            ValueError,
            lambda: normalize_message_content(content),
            "Image content rejected with ValueError"
        )

    def test_tc6_pydantic_model_support(self):
        """TC6: Pydantic model instances work correctly"""
        print("\n🧪 TC6: Pydantic Model Support")
        print("=" * 50)

        # TextContentPart Pydantic model
        content = [TextContentPart(text="Hello from Pydantic model")]
        result = normalize_message_content(content)
        self.assert_equal(result, "Hello from Pydantic model", "TextContentPart model works")

        # ImageUrlContentPart should also work (and be rejected)
        image_url = ImageUrlContentPart(image_url=ImageUrlDetail(url="http://example.com/image.png"))
        content2 = [
            TextContentPart(text="Text before image"),
            image_url
        ]
        self.assert_raises(
            ValueError,
            lambda: normalize_message_content(content2),
            "ImageUrlContentPart Pydantic model rejected"
        )

    def test_tc7_invalid_content_type(self):
        """TC7: Invalid content type raises TypeError"""
        print("\n🧪 TC7: Invalid Content Type")
        print("=" * 50)

        # Integer content should raise TypeError
        self.assert_raises(
            TypeError,
            lambda: normalize_message_content(123),
            "Integer content raises TypeError"
        )

        # None content should raise TypeError
        self.assert_raises(
            TypeError,
            lambda: normalize_message_content(None),
            "None content raises TypeError"
        )

    def test_tc8_empty_string_content(self):
        """TC8: Empty string content"""
        print("\n🧪 TC8: Empty String Content")
        print("=" * 50)

        # Empty string should return empty string
        result = normalize_message_content("")
        self.assert_equal(result, "", "Empty string returned unchanged")

    def test_tc9_empty_array_content(self):
        """TC9: Empty array content"""
        print("\n🧪 TC9: Empty Array Content")
        print("=" * 50)

        # Empty array should return empty string
        result = normalize_message_content([])
        self.assert_equal(result, "", "Empty array returns empty string")

    def test_tc10_empty_text_parts(self):
        """TC10: Text parts with empty strings"""
        print("\n🧪 TC10: Text Parts with Empty Strings")
        print("=" * 50)

        # Text parts with empty strings
        content = [
            {"type": "text", "text": "First"},
            {"type": "text", "text": ""},
            {"type": "text", "text": "Third"}
        ]
        result = normalize_message_content(content)
        expected = "First\n\nThird"
        self.assert_equal(result, expected, "Empty text parts preserved in output")

    def test_tc11_unknown_content_type(self):
        """TC11: Unknown content part type with warning"""
        print("\n🧪 TC11: Unknown Content Type")
        print("=" * 50)

        # Unknown content type should be logged as warning but not raise error
        content = [
            {"type": "text", "text": "Known type"},
            {"type": "unknown_type", "data": "some data"}
        ]
        result = normalize_message_content(content)
        expected = "Known type"
        self.assert_equal(result, expected, "Unknown content type ignored, text extracted")

    def test_tc12_complex_multimodal_structure(self):
        """TC12: Complex multimodal message structure"""
        print("\n🧪 TC12: Complex Multimodal Structure")
        print("=" * 50)

        # Complex structure with multiple text parts
        content = [
            TextContentPart(text="System instruction: You are helpful."),
            TextContentPart(text="User question: What is AI?")
        ]
        result = normalize_message_content(content)
        expected = "System instruction: You are helpful.\nUser question: What is AI?"
        self.assert_equal(result, expected, "Complex Pydantic model list normalized correctly")

    def test_tc13_image_url_content_with_vision_disabled(self):
        """TC13: Image URL content with vision disabled raises error"""
        print("\n🧪 TC13: Image URL Content (Vision Disabled)")
        print("=" * 50)

        # Image content should raise ValueError when support_vision=False
        content = [
            TextContentPart(text="What is in this image?"),
            ImageUrlContentPart(image_url=ImageUrlDetail(url="http://example.com/image.png"))
        ]

        self.assert_raises(
            ValueError,
            lambda: normalize_message_content(content, support_vision=False),
            "Image content rejected with ValueError when vision disabled"
        )

    def test_tc14_image_url_content_with_vision_enabled(self):
        """TC14: Image URL content with vision enabled logs warning"""
        print("\n🧪 TC14: Image URL Content (Vision Enabled)")
        print("=" * 50)

        # Image content should log warning when support_vision=True
        content = [
            TextContentPart(text="What is in this image?"),
            ImageUrlContentPart(image_url=ImageUrlDetail(url="http://example.com/image.png"))
        ]

        result = normalize_message_content(content, support_vision=True)
        # Should return only text parts
        expected = "What is in this image?"
        self.assert_equal(result, expected, "Image content filtered out, text returned")

    def test_tc15_vision_model_detection(self):
        """TC15: Vision model keyword detection"""
        print("\n🧪 TC15: Vision Model Detection")
        print("=" * 50)

        from mlx_router.api.app import VISION_ENABLED_MODELS

        # Check common vision model keywords
        for keyword in ['vl', 'vision', 'llava', 'qwen-vl', 'nvl']:
            self.assert_equal(
                keyword in VISION_ENABLED_MODELS,
                True,
                f"Vision keyword '{keyword}' should be in enabled list"
            )

    def test_tc16_text_model_no_vision_keywords(self):
        """TC16: Text models without vision keywords"""
        print("\n🧪 TC16: Text Model (No Vision Keywords)")
        print("=" * 50)

        from mlx_router.api.app import VISION_ENABLED_MODELS

        # Text models should not trigger vision support
        text_models = [
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "mlx-community/Qwen3-30B-A3B-8bit",
            "deepseek-ai/deepseek-coder-6.7b-instruct"
        ]

        for model_name in text_models:
            support_vision = any(keyword.lower() in model_name.lower() for keyword in VISION_ENABLED_MODELS)
            self.assert_equal(
                support_vision,
                False,
                f"Text model '{model_name}' should not have vision support"
            )

    def run_all_tests(self):
        """Run all test cases and report results"""
        print("\n" + "=" * 60)
        print("🚀 Running OpenAI Multimodal Content Format Tests (Phase 1 & 2)")
        print("=" * 60)

        self.test_tc1_string_content()
        self.test_tc2_array_text_only()
        self.test_tc3_mixed_format()
        self.test_tc4_multiple_text_parts()
        self.test_tc5_image_content_rejected()
        self.test_tc6_pydantic_model_support()
        self.test_tc7_invalid_content_type()
        self.test_tc8_empty_string_content()
        self.test_tc9_empty_array_content()
        self.test_tc10_empty_text_parts()
        self.test_tc11_unknown_content_type()
        self.test_tc12_complex_multimodal_structure()
        self.test_tc13_image_url_content_with_vision_disabled()
        self.test_tc14_image_url_content_with_vision_enabled()
        self.test_tc15_vision_model_detection()
        self.test_tc16_text_model_no_vision_keywords()

        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.passed} passed, {self.failed} failed")
        print("=" * 60)

        return self.failed == 0


if __name__ == "__main__":
    tester = TestContentNormalization()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
