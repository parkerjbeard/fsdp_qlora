"""
Comprehensive error handling utilities for robust operation.

This module provides:
- OOM (Out of Memory) detection and recovery
- Intelligent error messages with suggestions
- Graceful degradation mechanisms
- Resource monitoring and management
"""

import logging
import traceback
import psutil
import torch
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Callable, Any
from functools import wraps
import time

logger = logging.getLogger(__name__)


@dataclass 
class ErrorContext:
    """Context information for better error reporting."""
    operation: str
    backend: Optional[str] = None
    model_size: Optional[float] = None  # in GB
    quantization_bits: Optional[int] = None
    batch_size: Optional[int] = None
    device: Optional[str] = None
    additional_info: Optional[dict[str, Any]] = None


class MemoryError(Exception):
    """Custom exception for memory-related errors."""
    pass


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class BackendError(Exception):
    """Custom exception for backend-specific errors."""
    pass


def get_memory_info(device: str = "cpu") -> dict[str, float]:
    """Get current memory information for the specified device."""
    info = {}
    
    if device == "cpu":
        vm = psutil.virtual_memory()
        info["total_gb"] = vm.total / 1e9
        info["available_gb"] = vm.available / 1e9
        info["used_gb"] = vm.used / 1e9
        info["percent"] = vm.percent
        
    elif device == "cuda" and torch.cuda.is_available():
        info["allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        info["reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        info["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["available_gb"] = info["total_gb"] - info["allocated_gb"]
        
    elif device == "mps" and torch.backends.mps.is_available():
        try:
            info["allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
            info["reserved_gb"] = torch.mps.driver_allocated_memory() / 1e9
            # MPS doesn't provide total memory, use a conservative estimate
            info["total_gb"] = 64.0  # Conservative estimate for M1/M2
            info["available_gb"] = info["total_gb"] - info["allocated_gb"]
        except Exception as e:
            logger.warning(f"Could not get MPS memory info: {e}")
            info = {"error": str(e)}
    
    return info


def detect_oom_error(exception: Exception, device: str = "cpu") -> bool:
    """Detect if an exception is due to out-of-memory conditions."""
    error_msg = str(exception).lower()
    
    oom_indicators = [
        "out of memory",
        "oom",
        "cuda out of memory",
        "mps out of memory",
        "allocation failed",
        "insufficient memory",
        "cannot allocate memory",
        "memory error",
    ]
    
    return any(indicator in error_msg for indicator in oom_indicators)


def suggest_memory_optimization(
    context: ErrorContext,
    memory_info: dict[str, float]
) -> list[str]:
    """Suggest optimizations based on error context and memory state."""
    suggestions = []
    
    # Batch size suggestions
    if context.batch_size and context.batch_size > 1:
        suggestions.append(f"Reduce batch size from {context.batch_size} to {context.batch_size // 2}")
    
    # Quantization suggestions
    if context.quantization_bits:
        if context.quantization_bits > 4:
            suggestions.append(f"Use lower quantization bits (current: {context.quantization_bits})")
        suggestions.append("Enable gradient checkpointing to save memory")
    
    # Model-specific suggestions
    if context.model_size:
        if context.model_size > 7:
            suggestions.append("Consider using model sharding or FSDP")
        suggestions.append("Enable CPU offloading for optimizer states")
    
    # Backend-specific suggestions
    if context.backend == "mps":
        suggestions.append("Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to reduce memory fragmentation")
        suggestions.append("Use torch.mps.empty_cache() between batches")
    elif context.backend == "cuda":
        suggestions.append("Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
        suggestions.append("Use torch.cuda.empty_cache() between batches")
    
    # General suggestions
    suggestions.append("Enable memory-efficient attention mechanisms")
    suggestions.append("Use mixed precision training (fp16/bf16)")
    
    return suggestions


def format_error_message(
    exception: Exception,
    context: ErrorContext,
    include_traceback: bool = True
) -> str:
    """Format a comprehensive error message with context and suggestions."""
    lines = [
        f"\n{'='*60}",
        f"ERROR: {context.operation}",
        f"{'='*60}",
        f"Exception Type: {type(exception).__name__}",
        f"Message: {str(exception)}",
    ]
    
    # Add context information
    if context.backend:
        lines.append(f"Backend: {context.backend}")
    if context.device:
        lines.append(f"Device: {context.device}")
    if context.model_size:
        lines.append(f"Model Size: {context.model_size:.1f}GB")
    if context.quantization_bits:
        lines.append(f"Quantization: {context.quantization_bits}-bit")
    if context.batch_size:
        lines.append(f"Batch Size: {context.batch_size}")
    
    # Add memory information
    if context.device:
        memory_info = get_memory_info(context.device)
        if memory_info and "error" not in memory_info:
            lines.append("\nMemory Status:")
            for key, value in memory_info.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.2f}")
    
    # Add suggestions if OOM
    if detect_oom_error(exception, context.device or "cpu"):
        suggestions = suggest_memory_optimization(context, memory_info)
        if suggestions:
            lines.append("\nSuggested Solutions:")
            for i, suggestion in enumerate(suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
    
    # Add traceback if requested
    if include_traceback:
        lines.append("\nTraceback:")
        lines.append(traceback.format_exc())
    
    lines.append(f"{'='*60}\n")
    
    return "\n".join(lines)


@contextmanager
def error_handler(context: ErrorContext, reraise: bool = True):
    """Context manager for comprehensive error handling."""
    start_time = time.time()
    
    try:
        yield
    except Exception as e:
        # Log the formatted error
        error_msg = format_error_message(e, context)
        logger.error(error_msg)
        
        # Clean up resources if OOM
        if detect_oom_error(e, context.device or "cpu"):
            logger.info("Attempting memory cleanup...")
            
            if context.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif context.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Log memory after cleanup
            memory_after = get_memory_info(context.device or "cpu")
            logger.info(f"Memory after cleanup: {memory_after}")
        
        # Reraise if requested
        if reraise:
            raise
    finally:
        elapsed = time.time() - start_time
        logger.debug(f"{context.operation} took {elapsed:.2f}s")


def with_recovery(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    recover_fn: Optional[Callable] = None
):
    """Decorator for automatic retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Log the attempt
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                    )
                    
                    # Call recovery function if provided
                    if recover_fn:
                        try:
                            recover_fn(e)
                        except Exception as recover_error:
                            logger.error(f"Recovery function failed: {recover_error}")
                    
                    # Wait before retry (except on last attempt)
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor ** attempt
                        logger.info(f"Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
            
            # All retries failed
            logger.error(f"All {max_retries} attempts failed")
            raise last_exception
        
        return wrapper
    return decorator


class ResourceMonitor:
    """Monitor system resources and warn about potential issues."""
    
    def __init__(
        self,
        device: str = "cpu",
        memory_threshold: float = 0.9,  # 90% usage
        check_interval: float = 10.0    # seconds
    ):
        self.device = device
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self._last_check = 0
        self._warnings_issued = set()
    
    def check_resources(self) -> Optional[str]:
        """Check resources and return warning if needed."""
        current_time = time.time()
        
        # Rate limit checks
        if current_time - self._last_check < self.check_interval:
            return None
        
        self._last_check = current_time
        
        # Get memory info
        memory_info = get_memory_info(self.device)
        
        if "error" in memory_info:
            return None
        
        # Check memory usage
        if "percent" in memory_info:
            usage_percent = memory_info["percent"] / 100
        elif "allocated_gb" in memory_info and "total_gb" in memory_info:
            usage_percent = memory_info["allocated_gb"] / memory_info["total_gb"]
        else:
            return None
        
        # Issue warning if threshold exceeded
        if usage_percent > self.memory_threshold:
            warning_key = f"memory_{int(usage_percent * 100)}"
            
            if warning_key not in self._warnings_issued:
                self._warnings_issued.add(warning_key)
                warning = (
                    f"High memory usage detected on {self.device}: "
                    f"{usage_percent * 100:.1f}% used"
                )
                logger.warning(warning)
                return warning
        
        return None


def validate_configuration(config: dict[str, Any]) -> list[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check batch size
    if "batch_size" in config:
        if config["batch_size"] > 128:
            issues.append(f"Very large batch size ({config['batch_size']}) may cause OOM")
    
    # Check quantization settings
    if "quantization_bits" in config:
        if config["quantization_bits"] not in [4, 8, 16]:
            issues.append(
                f"Unusual quantization bits ({config['quantization_bits']}). "
                "Common values are 4, 8, or 16"
            )
    
    # Check mixed precision settings
    if "dtype" in config:
        if config["dtype"] == "bfloat16" and config.get("backend") == "mps":
            issues.append("BFloat16 is not supported on MPS backend")
    
    # Check memory settings
    if "gradient_checkpointing" not in config and config.get("model_size", 0) > 7:
        issues.append("Consider enabling gradient checkpointing for large models")
    
    return issues


# Specific error handlers for common scenarios

def handle_import_error(
    module_name: str,
    fallback: Optional[Callable] = None,
    install_cmd: Optional[str] = None
) -> Any:
    """Handle import errors with helpful messages."""
    try:
        import importlib
        module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        error_msg = f"Failed to import '{module_name}': {str(e)}"
        
        if install_cmd:
            error_msg += f"\nInstall with: {install_cmd}"
        
        logger.error(error_msg)
        
        if fallback:
            logger.info(f"Using fallback for '{module_name}'")
            return fallback()
        else:
            raise ImportError(error_msg)


def handle_backend_error(
    exception: Exception,
    backend: str,
    fallback_backend: Optional[str] = None
) -> Optional[str]:
    """Handle backend-specific errors with fallback options."""
    logger.error(f"Backend '{backend}' failed: {str(exception)}")
    
    if fallback_backend:
        logger.info(f"Attempting to fall back to '{fallback_backend}'")
        return fallback_backend
    else:
        # Suggest alternative backends
        suggestions = []
        
        if backend == "cuda":
            suggestions.extend(["mps", "cpu"])
        elif backend == "mps":
            suggestions.extend(["cpu"])
        elif backend == "mlx":
            suggestions.extend(["mps", "cpu"])
        
        if suggestions:
            logger.info(f"Consider using one of: {', '.join(suggestions)}")
        
        raise BackendError(f"Backend '{backend}' failed with no fallback available")


# Export public API
__all__ = [
    'ErrorContext',
    'MemoryError',
    'ConfigurationError', 
    'BackendError',
    'get_memory_info',
    'detect_oom_error',
    'suggest_memory_optimization',
    'format_error_message',
    'error_handler',
    'with_recovery',
    'ResourceMonitor',
    'validate_configuration',
    'handle_import_error',
    'handle_backend_error',
]