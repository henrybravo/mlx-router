#!/usr/bin/env python3
"""
Test suite for memory pressure monitoring and management
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the mlx_router package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mlx_router.core.resource_monitor import ResourceMonitor
from mlx_router.config.model_config import ModelConfig


class TestMemoryPressure(unittest.TestCase):
    """Test memory pressure detection and management"""

    def setUp(self):
        """Set up test fixtures"""
        # Clear any cached memory info
        ResourceMonitor._cached_memory_info = None
        ResourceMonitor._last_memory_check = 0

    def tearDown(self):
        """Clean up after tests"""
        # Clear cached memory info
        ResourceMonitor._cached_memory_info = None
        ResourceMonitor._last_memory_check = 0

    @patch('mlx_router.core.resource_monitor.psutil.virtual_memory')
    @patch('mlx_router.core.resource_monitor.psutil.swap_memory')
    def test_memory_pressure_levels(self, mock_swap, mock_memory):
        """Test that memory pressure levels are calculated correctly"""
        print("\n=== Testing Memory Pressure Levels ===")

        # Set test thresholds
        ResourceMonitor.set_memory_threshold_gb(80.0)
        ResourceMonitor.set_swap_thresholds(95.0, 85.0)  # Critical at 95%, high at 85%
        print(f"Set thresholds: memory=80.0GB, swap_critical=95.0%, swap_high=85.0%")

        # Test data: 128GB system with 80GB threshold
        # Available memory = 128GB - used
        test_cases = [
            # (available_gb, swap_percent, expected_pressure)
            (100.0, 0, "normal"),     # Plenty available
            (85.0, 0, "normal"),      # Above threshold
            (70.0, 0, "normal"),      # Above 80% of threshold (64GB)
            (55.0, 0, "moderate"),    # Below 80% but above 60% of threshold
            (40.0, 0, "high"),        # Below 60% but above 40% of threshold
            (25.0, 0, "critical"),    # Below 40% of threshold (32GB)
            (85.0, 90, "high"),       # High due to swap (90% > 85%)
            (60.0, 96, "critical"),   # Critical due to swap (96% > 95%)
        ]

        for available_gb, swap_percent, expected in test_cases:
            with self.subTest(available_gb=available_gb, swap_percent=swap_percent):
                print(f"\nTesting: {available_gb}GB available, {swap_percent}% swap -> expected: {expected}")

                # Mock memory info
                mock_memory.return_value = MagicMock()
                mock_memory.return_value.total = 128 * (1024**3)  # 128GB
                mock_memory.return_value.available = available_gb * (1024**3)
                mock_memory.return_value.used = (128 - available_gb) * (1024**3)
                mock_memory.return_value.percent = ((128 - available_gb) / 128) * 100
                mock_memory.return_value.free = 10 * (1024**3)  # 10GB free
                mock_memory.return_value.buffers = 1 * (1024**3)
                mock_memory.return_value.cached = 2 * (1024**3)

                mock_swap.return_value = MagicMock()
                mock_swap.return_value.total = 16 * (1024**3)  # 16GB swap
                mock_swap.return_value.used = (16 * swap_percent/100) * (1024**3)
                mock_swap.return_value.percent = swap_percent

                # Clear cache to force fresh calculation
                ResourceMonitor._cached_memory_info = None

                # Test pressure calculation
                pressure = ResourceMonitor.get_memory_pressure()
                print(f"Result: {pressure}")
                self.assertEqual(pressure, expected,
                    f"Expected {expected} pressure for {available_gb}GB available, {swap_percent}% swap, got {pressure}")

    @patch('mlx_router.core.resource_monitor.psutil.virtual_memory')
    @patch('mlx_router.core.resource_monitor.psutil.swap_memory')
    def test_memory_info_accuracy(self, mock_swap, mock_memory):
        """Test that memory info is calculated accurately"""
        print("\n=== Testing Memory Info Accuracy ===")

        # Mock 128GB system with 51.2% usage (user's scenario)
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.total = 128 * (1024**3)
        mock_memory.return_value.available = 128 * 0.488 * (1024**3)  # ~62.5GB available
        mock_memory.return_value.used = 128 * 0.512 * (1024**3)     # ~65.5GB used
        mock_memory.return_value.percent = 51.2
        mock_memory.return_value.free = 10 * (1024**3)
        mock_memory.return_value.buffers = 1 * (1024**3)
        mock_memory.return_value.cached = 2 * (1024**3)

        mock_swap.return_value = MagicMock()
        mock_swap.return_value.total = 16 * (1024**3)
        mock_swap.return_value.used = 0
        mock_swap.return_value.percent = 0

        # Clear cache
        ResourceMonitor._cached_memory_info = None

        # Get memory info
        info = ResourceMonitor.get_memory_info(use_cache=False)
        print(f"Memory info: total={info['total_gb']:.1f}GB, used={info['used_gb']:.1f}GB, available={info['available_gb']:.1f}GB, swap={info['swap_percent']}%")

        # Verify calculations
        self.assertAlmostEqual(info['total_gb'], 128.0, places=1)
        self.assertAlmostEqual(info['used_gb'], 65.5, places=1)
        self.assertAlmostEqual(info['available_gb'], 62.5, places=1)
        self.assertAlmostEqual(info['used_percent'], 51.2, places=1)
        self.assertEqual(info['swap_percent'], 0)

        # Set thresholds for this test
        ResourceMonitor.set_memory_threshold_gb(80.0)
        ResourceMonitor.set_swap_thresholds(95.0, 85.0)
        print(f"Set thresholds: memory=80.0GB, swap_critical=95.0%, swap_high=85.0%")

        # With 62.5GB available and 0% swap, pressure should be "moderate" (62.5 < 64GB = 0.8*80)
        pressure = ResourceMonitor.get_memory_pressure()
        print(f"Pressure result: {pressure}")
        self.assertEqual(pressure, "moderate",
            f"62.5GB available should be 'moderate' pressure, got '{pressure}'")

    @patch('mlx_router.core.resource_monitor.psutil.virtual_memory')
    @patch('mlx_router.core.resource_monitor.psutil.swap_memory')
    def test_memory_pressure_max_tokens(self, mock_swap, mock_memory):
        """Test that memory pressure affects max tokens correctly"""
        print("\n=== Testing Memory Pressure Max Tokens ===")

        # Mock normal memory conditions
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.total = 128 * (1024**3)
        mock_memory.return_value.available = 100 * (1024**3)  # 100GB available
        mock_memory.return_value.used = 28 * (1024**3)       # 28GB used
        mock_memory.return_value.percent = 21.9
        mock_memory.return_value.free = 90 * (1024**3)
        mock_memory.return_value.buffers = 1 * (1024**3)
        mock_memory.return_value.cached = 2 * (1024**3)

        mock_swap.return_value = MagicMock()
        mock_swap.return_value.total = 16 * (1024**3)
        mock_swap.return_value.used = 0
        mock_swap.return_value.percent = 0

        # Test model config
        model_name = "test-model"
        config = {
            "max_tokens": 16384,
            "memory_pressure_max_tokens": {
                "normal": 16384,
                "moderate": 8192,
                "high": 4096,
                "critical": 2048
            }
        }
        print(f"Model config: {config}")

        # Mock ModelConfig.get_config
        with patch.object(ModelConfig, 'get_config', return_value=config):
            # Test different pressure levels
            test_cases = [
                ("normal", 16384),
                ("moderate", 8192),
                ("high", 4096),
                ("critical", 2048),
            ]

            for pressure_level, expected_tokens in test_cases:
                with self.subTest(pressure_level=pressure_level):
                    print(f"Testing pressure level: {pressure_level} -> expected tokens: {expected_tokens}")
                    # Mock the pressure level
                    with patch.object(ResourceMonitor, 'get_memory_pressure', return_value=pressure_level):
                        actual_tokens = ResourceMonitor.get_memory_pressure_max_tokens(model_name, pressure_level)
                        print(f"Result: {actual_tokens} tokens")
                        self.assertEqual(actual_tokens, expected_tokens,
                            f"Expected {expected_tokens} tokens for {pressure_level} pressure, got {actual_tokens}")

    @patch('mlx_router.core.resource_monitor.psutil.virtual_memory')
    @patch('mlx_router.core.resource_monitor.psutil.swap_memory')
    def test_memory_fragmentation_calculation(self, mock_swap, mock_memory):
        """Test memory fragmentation score calculation"""
        print("\n=== Testing Memory Fragmentation Calculation ===")

        # Mock memory with fragmentation (available < free due to fragmentation)
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.total = 32 * (1024**3)
        mock_memory.return_value.available = 8 * (1024**3)   # 8GB available
        mock_memory.return_value.free = 12 * (1024**3)      # 12GB free (but fragmented)
        mock_memory.return_value.buffers = 1 * (1024**3)
        mock_memory.return_value.cached = 1 * (1024**3)
        mock_memory.return_value.percent = 75.0

        mock_swap.return_value = MagicMock()
        mock_swap.return_value.total = 8 * (1024**3)
        mock_swap.return_value.used = 0
        mock_swap.return_value.percent = 0

        # Clear cache
        ResourceMonitor._cached_memory_info = None

        info = ResourceMonitor.get_memory_info(use_cache=False)
        print(f"Fragmented memory: available={info['available_gb']:.1f}GB, free={info['free_gb']:.1f}GB, fragmentation_score={info['fragmentation_score']}")

        # Fragmentation score should be > 0 when available < free + buffers + cached
        self.assertGreater(info['fragmentation_score'], 0,
            "Fragmentation score should be > 0 when memory is fragmented")

        # Test with non-fragmented memory
        mock_memory.return_value.available = 14 * (1024**3)  # 14GB available (> free + buffers + cached)

        ResourceMonitor._cached_memory_info = None
        info = ResourceMonitor.get_memory_info(use_cache=False)
        print(f"Non-fragmented memory: available={info['available_gb']:.1f}GB, free={info['free_gb']:.1f}GB, fragmentation_score={info['fragmentation_score']}")

        self.assertEqual(info['fragmentation_score'], 0,
            "Fragmentation score should be 0 when memory is not fragmented")

    def test_memory_cache(self):
        """Test that memory info caching works correctly"""
        print("\n=== Testing Memory Cache ===")

        # This test would need actual psutil calls, so we'll skip detailed implementation
        # In a real test environment, we'd verify that cached values are returned within the cache duration

        # Clear cache
        ResourceMonitor._cached_memory_info = None
        ResourceMonitor._last_memory_check = 0
        print("Cleared memory cache")

        # First call should populate cache
        info1 = ResourceMonitor.get_memory_info(use_cache=True)
        print(f"First call: got memory info (cached: {info1 is not None})")

        # Second call within cache duration should return cached data
        import time
        time.sleep(0.1)  # Sleep less than cache duration (2 seconds)
        info2 = ResourceMonitor.get_memory_info(use_cache=True)
        print(f"Second call: got memory info (cached: {info2 is not None})")

        # Should be the same object (cached)
        is_cached = info1 is info2
        print(f"Cache test result: objects are identical = {is_cached}")
        self.assertIs(info1, info2, "Memory info should be cached within cache duration")


if __name__ == '__main__':
    unittest.main()