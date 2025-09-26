#!/usr/bin/env python3
"""
Resource monitoring for Apple Silicon optimization
"""

import time
import logging
import psutil
from mlx_router.config.model_config import ModelConfig

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor system resources for Apple Silicon optimization"""
    _last_memory_check = 0
    _cached_memory_info = None
    _memory_cache_duration = 2.0
    _memory_threshold_gb = 80.0  # Default fallback value
    _swap_critical_percent = 90.0  # Default swap critical threshold
    _swap_high_percent = 75.0  # Default swap high threshold
    
    @classmethod
    def set_memory_threshold_gb(cls, threshold_gb: float):
        """Set the memory threshold in GB for pressure calculations"""
        cls._memory_threshold_gb = threshold_gb
        logger.info(f"Memory threshold set to {threshold_gb}GB")

    @classmethod
    def set_swap_thresholds(cls, critical_percent: float, high_percent: float):
        """Set the swap usage thresholds for pressure calculations"""
        cls._swap_critical_percent = critical_percent
        cls._swap_high_percent = high_percent
        logger.info(f"Swap thresholds set to critical={critical_percent}%, high={high_percent}%")

    @staticmethod
    def get_memory_info(use_cache=True):
        current_time = time.time()
        if (use_cache and ResourceMonitor._cached_memory_info and 
            current_time - ResourceMonitor._last_memory_check < ResourceMonitor._memory_cache_duration):
            return ResourceMonitor._cached_memory_info
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        memory_info = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "used_percent": memory.percent,
            "free_gb": memory.free / (1024**3),
            "buffers_gb": getattr(memory, 'buffers', 0) / (1024**3),
            "cached_gb": getattr(memory, 'cached', 0) / (1024**3),
            "swap_total_gb": swap.total / (1024**3),
            "swap_used_gb": swap.used / (1024**3),
            "swap_percent": swap.percent,
            "fragmentation_score": ResourceMonitor._calculate_fragmentation_score(memory)
        }
        ResourceMonitor._cached_memory_info = memory_info
        ResourceMonitor._last_memory_check = current_time
        return memory_info
    @staticmethod
    def _calculate_fragmentation_score(memory):
        """Calculate a simple fragmentation score (0-100, lower is better)"""
        # Simple heuristic: high fragmentation when available memory is much less than free memory
        if hasattr(memory, 'free') and memory.available > 0:
            # If available is much less than free, suggests fragmentation
            fragmentation_ratio = 1.0 - (memory.available / max(memory.free + getattr(memory, 'buffers', 0) + getattr(memory, 'cached', 0), memory.available))
            return min(100, max(0, fragmentation_ratio * 100))
        return 0  # Unable to calculate
    
    @staticmethod
    def check_memory_available(model_name, safety_margin=1.5, use_cache=True):
        """Check if sufficient memory is available for the specified model"""
        required_gb = ModelConfig.get_config(model_name).get("required_memory_gb", 8)
        info = ResourceMonitor.get_memory_info(use_cache=use_cache)
        
        # Apply safety margin and consider fragmentation
        effective_required = required_gb * safety_margin
        
        # Reduce fragmentation penalty if we have abundant memory
        fragmentation_penalty = 1.0
        if info["fragmentation_score"] > 50:
            if info["available_gb"] > required_gb * 4:  # Abundant memory
                fragmentation_penalty = 1.1  # Minimal penalty
            elif info["available_gb"] > required_gb * 2:  # Sufficient memory
                fragmentation_penalty = 1.15  # Small penalty
            else:
                fragmentation_penalty = 1.2  # Standard penalty
            logger.debug(f"Memory fragmentation detected ({info['fragmentation_score']:.1f}), applying {fragmentation_penalty}x penalty")
        
        effective_required *= fragmentation_penalty
        
        return info["available_gb"] >= effective_required, {
            'available_gb': info["available_gb"],
            'required_gb': required_gb,
            'effective_required': effective_required,
            'fragmentation_penalty': fragmentation_penalty
        }
    @staticmethod
    def get_memory_pressure():
        """Get memory pressure level for Apple Silicon optimization"""
        info = ResourceMonitor.get_memory_info()
        available_gb = info["available_gb"]
        swap_percent = info["swap_percent"]
        threshold = ResourceMonitor._memory_threshold_gb
        swap_critical = ResourceMonitor._swap_critical_percent
        swap_high = ResourceMonitor._swap_high_percent

        # Critical: very low available memory OR high swap usage
        if available_gb < threshold * 0.4 or swap_percent > swap_critical:
            return "critical"
        # High: low available memory OR moderate swap usage
        elif available_gb < threshold * 0.6 or swap_percent > swap_high:
            return "high"
        # Moderate: somewhat low available memory
        elif available_gb < threshold * 0.8:
            return "moderate"
        # Normal: sufficient available memory
        else:
            return "normal"
    
    @staticmethod
    def get_memory_pressure_max_tokens(model_name, pressure_level):
        """Get max tokens for current memory pressure level"""
        model_config = ModelConfig.get_config(model_name)
        pressure_tokens = model_config.get("memory_pressure_max_tokens", {})
        
        if pressure_level in pressure_tokens:
            return pressure_tokens[pressure_level]
        
        # Fallback to standard max_tokens if no pressure-specific config
        return model_config.get("max_tokens", 4096)
    
    @staticmethod
    def should_defer_model_load(model_name):
        """Check if model loading should be deferred due to memory pressure"""
        pressure = ResourceMonitor.get_memory_pressure()
        info = ResourceMonitor.get_memory_info(use_cache=False)  # Fresh info for loading decisions
        required_gb = ModelConfig.get_config(model_name).get("required_memory_gb", 8)
        
        # Dynamic swap threshold based on available memory
        swap_threshold = 95 if info["available_gb"] > 50 else 90 if info["available_gb"] > 20 else 50
        
        # Consider swap usage as additional pressure indicator
        if info["swap_percent"] > swap_threshold:
            return True, f"High swap usage ({info['swap_percent']:.1f}% > {swap_threshold}%) indicates memory pressure"
        
        # Bypass memory pressure checks if we have abundant available RAM for the model
        if info["available_gb"] > required_gb * 3:  # 3x safety margin means we can bypass some restrictions
            logger.debug(f"Abundant memory available ({info['available_gb']:.1f}GB > {required_gb * 3}GB), bypassing pressure checks")
            return False, f"Abundant memory available ({info['available_gb']:.1f}GB)"
        
        # Use consolidated memory checking based on pressure level
        if pressure == "critical":
            # Even in critical pressure, allow loading if we have enough available memory
            can_load, mem_info = ResourceMonitor.check_memory_available(model_name, safety_margin=2.0, use_cache=False)
            if can_load:
                logger.warning(f"Critical pressure but sufficient memory available, allowing load")
                return False, f"Critical pressure bypassed due to sufficient available memory"
            return True, f"Critical memory pressure ({info['used_percent']:.1f}%)"
        elif pressure == "high":
            can_load, mem_info = ResourceMonitor.check_memory_available(model_name, safety_margin=1.2, use_cache=False)
            if not can_load:
                return True, f"High memory pressure with insufficient memory (need {mem_info['effective_required']:.1f}GB, available {mem_info['available_gb']:.1f}GB)"
        elif pressure == "moderate":
            # Only defer if both fragmentation is high AND memory is tight
            if info["fragmentation_score"] > 70:
                can_load, mem_info = ResourceMonitor.check_memory_available(model_name, safety_margin=1.5, use_cache=False)
                if not can_load:
                    return True, f"Moderate memory pressure with fragmentation concerns (frag score: {info['fragmentation_score']:.1f})"
        
        return False, "Memory sufficient for loading"