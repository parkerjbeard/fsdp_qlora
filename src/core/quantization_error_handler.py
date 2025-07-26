"""
Enhanced quantization with comprehensive error handling.

This module wraps quantization operations with proper error handling,
OOM detection, and automatic recovery mechanisms.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Any, Callable

from src.utils.error_handling import (
    ErrorContext,
    MemoryError,
    error_handler,
    detect_oom_error,
    get_memory_info,
    with_recovery,
    ResourceMonitor,
    format_error_message,
)
from src.core.quantization_wrapper import QuantizationAdapter, QuantizationConfig
from src.core.backend_manager import Backend

logger = logging.getLogger(__name__)


class QuantizationErrorHandler:
    """Wrapper for quantization operations with error handling."""
    
    def __init__(
        self,
        backend: Backend,
        config: QuantizationConfig,
        enable_monitoring: bool = True
    ):
        self.backend = backend
        self.config = config
        self.enable_monitoring = enable_monitoring
        
        # Initialize resource monitor
        device = self._get_device_str()
        self.monitor = ResourceMonitor(device=device) if enable_monitoring else None
        
        # Track quantization statistics
        self.stats = {
            "successful_quantizations": 0,
            "failed_quantizations": 0,
            "oom_errors": 0,
            "recovered_errors": 0,
        }
    
    def _get_device_str(self) -> str:
        """Get device string for monitoring."""
        if self.backend == Backend.CUDA:
            return "cuda"
        elif self.backend == Backend.MPS:
            return "mps"
        else:
            return "cpu"
    
    def _cleanup_memory(self):
        """Clean up memory after error."""
        device = self._get_device_str()
        
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info(f"Memory cleanup completed for {device}")
    
    def _reduce_batch_size(self, current_batch_size: int) -> int:
        """Suggest reduced batch size after OOM."""
        new_batch_size = max(1, current_batch_size // 2)
        logger.info(f"Reducing batch size from {current_batch_size} to {new_batch_size}")
        return new_batch_size
    
    def _reduce_quantization_bits(self, current_bits: int) -> Optional[int]:
        """Suggest reduced quantization bits after OOM."""
        bit_options = [16, 8, 4]
        
        try:
            current_idx = bit_options.index(current_bits)
            if current_idx < len(bit_options) - 1:
                new_bits = bit_options[current_idx + 1]
                logger.info(f"Reducing quantization from {current_bits}-bit to {new_bits}-bit")
                return new_bits
        except ValueError:
            pass
        
        return None
    
    @with_recovery(max_retries=3, backoff_factor=2.0)
    def quantize_model_safe(
        self,
        model: nn.Module,
        quantization_adapter: QuantizationAdapter,
        batch_size: Optional[int] = None,
    ) -> nn.Module:
        """Quantize model with error handling and recovery."""
        # Check resources before starting
        if self.monitor:
            warning = self.monitor.check_resources()
            if warning:
                logger.warning(f"Resource warning before quantization: {warning}")
        
        # Create error context
        context = ErrorContext(
            operation="Model Quantization",
            backend=str(self.backend),
            quantization_bits=self.config.bits,
            batch_size=batch_size,
            device=self._get_device_str(),
            model_size=sum(p.numel() for p in model.parameters()) * 4 / 1e9,  # Approximate GB
        )
        
        try:
            with error_handler(context, reraise=True):
                # Log memory before quantization
                memory_before = get_memory_info(context.device)
                logger.info(f"Memory before quantization: {memory_before}")
                
                # Perform quantization
                quantized_model = quantization_adapter.quantize_model(model)
                
                # Log memory after quantization
                memory_after = get_memory_info(context.device)
                logger.info(f"Memory after quantization: {memory_after}")
                
                # Update statistics
                self.stats["successful_quantizations"] += 1
                
                return quantized_model
                
        except Exception as e:
            self.stats["failed_quantizations"] += 1
            
            if detect_oom_error(e, context.device):
                self.stats["oom_errors"] += 1
                logger.error("OOM error detected during quantization")
                
                # Clean up memory
                self._cleanup_memory()
                
                # Try with reduced settings
                if batch_size and batch_size > 1:
                    # Suggest batch size reduction
                    new_batch_size = self._reduce_batch_size(batch_size)
                    context.additional_info = {"suggested_batch_size": new_batch_size}
                
                # Check if we can reduce quantization bits
                if self.config.bits > 4:
                    new_bits = self._reduce_quantization_bits(self.config.bits)
                    if new_bits:
                        context.additional_info = context.additional_info or {}
                        context.additional_info["suggested_bits"] = new_bits
            
            raise
    
    def quantize_layer_safe(
        self,
        layer: nn.Module,
        layer_name: str,
        quantization_fn: Callable,
    ) -> nn.Module:
        """Quantize individual layer with error handling."""
        context = ErrorContext(
            operation=f"Layer Quantization: {layer_name}",
            backend=str(self.backend),
            quantization_bits=self.config.bits,
            device=self._get_device_str(),
        )
        
        try:
            return quantization_fn(layer)
        except Exception as e:
            # Log the error with context
            error_msg = format_error_message(e, context)
            logger.error(error_msg)
            
            if detect_oom_error(e, context.device):
                logger.warning(f"Skipping quantization for layer {layer_name} due to OOM")
                self._cleanup_memory()
            else:
                # For non-OOM errors, just log
                logger.error(f"Failed to quantize layer {layer_name}: {e}")
            
            # Always return original layer on failure
            return layer
    
    def save_quantized_model_safe(
        self,
        model: nn.Module,
        save_path: str,
        quantization_adapter: QuantizationAdapter,
    ):
        """Save quantized model with error handling."""
        context = ErrorContext(
            operation="Save Quantized Model",
            backend=str(self.backend),
            device=self._get_device_str(),
        )
        
        @with_recovery(max_retries=2)
        def _save():
            with error_handler(context):
                quantization_adapter.save_quantized_model(model, save_path)
                logger.info(f"Successfully saved quantized model to {save_path}")
        
        _save()
    
    def load_quantized_model_safe(
        self,
        model_path: str,
        model_class: type,
        quantization_adapter: QuantizationAdapter,
        device: Optional[str] = None,
    ) -> nn.Module:
        """Load quantized model with error handling."""
        context = ErrorContext(
            operation="Load Quantized Model",
            backend=str(self.backend),
            device=device or self._get_device_str(),
        )
        
        @with_recovery(max_retries=2)
        def _load():
            with error_handler(context):
                # Try to load with memory mapping first
                try:
                    model = quantization_adapter.load_quantized_model(
                        model_path,
                        model_class,
                        device=device,
                        mmap=True,  # Memory-mapped loading
                    )
                except TypeError:
                    # Fallback if mmap not supported
                    model = quantization_adapter.load_quantized_model(
                        model_path,
                        model_class,
                        device=device,
                    )
                
                logger.info(f"Successfully loaded quantized model from {model_path}")
                return model
        
        return _load()
    
    def benchmark_quantization_safe(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        quantization_adapter: QuantizationAdapter,
        num_warmup: int = 5,
        num_runs: int = 20,
    ) -> dict[str, Any]:
        """Benchmark quantization with error handling."""
        context = ErrorContext(
            operation="Quantization Benchmark",
            backend=str(self.backend),
            quantization_bits=self.config.bits,
            device=self._get_device_str(),
        )
        
        results = {
            "status": "failed",
            "error": None,
            "timings": [],
        }
        
        try:
            with error_handler(context, reraise=False):
                import time
                
                # Warmup
                for _ in range(num_warmup):
                    dummy_input = torch.randn(input_shape).to(context.device)
                    with torch.no_grad():
                        _ = model(dummy_input)
                
                # Benchmark
                timings = []
                for _ in range(num_runs):
                    dummy_input = torch.randn(input_shape).to(context.device)
                    
                    start = time.time()
                    with torch.no_grad():
                        _ = model(dummy_input)
                    
                    if context.device == "cuda":
                        torch.cuda.synchronize()
                    elif context.device == "mps":
                        torch.mps.synchronize()
                    
                    elapsed = time.time() - start
                    timings.append(elapsed)
                
                # Calculate statistics
                import numpy as np
                results.update({
                    "status": "success",
                    "timings": timings,
                    "mean_ms": np.mean(timings) * 1000,
                    "std_ms": np.std(timings) * 1000,
                    "min_ms": np.min(timings) * 1000,
                    "max_ms": np.max(timings) * 1000,
                })
                
        except Exception as e:
            results["error"] = str(e)
            if detect_oom_error(e, context.device):
                results["error_type"] = "OOM"
                self._cleanup_memory()
        
        return results
    
    def get_statistics(self) -> dict[str, Any]:
        """Get quantization error statistics."""
        stats = self.stats.copy()
        
        # Calculate success rate
        total = stats["successful_quantizations"] + stats["failed_quantizations"]
        if total > 0:
            stats["success_rate"] = stats["successful_quantizations"] / total
        else:
            stats["success_rate"] = 0.0
        
        # Add memory info
        stats["current_memory"] = get_memory_info(self._get_device_str())
        
        return stats


def create_safe_quantization_adapter(
    backend: Backend,
    config: QuantizationConfig,
    base_adapter: QuantizationAdapter,
) -> QuantizationAdapter:
    """Create a quantization adapter with error handling wrapper."""
    
    error_handler = QuantizationErrorHandler(backend, config)
    
    class SafeQuantizationAdapter(QuantizationAdapter):
        """Quantization adapter with integrated error handling."""
        
        def __init__(self):
            # Set attributes before calling super to avoid issues with _validate_backend_support
            self.base_adapter = base_adapter
            self.error_handler = error_handler
            # Call parent init
            super().__init__(backend, config)
        
        def _validate_backend_support(self):
            """Delegate to base adapter."""
            # Skip validation during __init__ when base_adapter isn't set yet
            if hasattr(self, 'base_adapter') and hasattr(self.base_adapter, '_validate_backend_support'):
                self.base_adapter._validate_backend_support()
        
        def create_quantized_linear(self, in_features: int, out_features: int, bias: bool = True):
            """Delegate to base adapter."""
            return self.base_adapter.create_quantized_linear(in_features, out_features, bias)
        
        def prepare_model_for_training(self, model: nn.Module) -> nn.Module:
            """Delegate to base adapter."""
            return self.base_adapter.prepare_model_for_training(model)
        
        def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
            """Quantize model with error handling."""
            batch_size = kwargs.get("batch_size")
            return self.error_handler.quantize_model_safe(
                model,
                self.base_adapter,
                batch_size=batch_size
            )
        
        def quantize_weights(self, weights: torch.Tensor) -> Any:
            """Quantize weights with error handling."""
            try:
                return self.base_adapter.quantize_weights(weights)
            except Exception as e:
                if detect_oom_error(e, self.error_handler._get_device_str()):
                    self.error_handler._cleanup_memory()
                    raise MemoryError(f"OOM during weight quantization: {e}")
                raise
        
        def dequantize_weights(self, quantized_weights: Any) -> torch.Tensor:
            """Dequantize weights with error handling."""
            return self.base_adapter.dequantize_weights(quantized_weights)
        
        def save_quantized_model(self, model: nn.Module, save_path: str):
            """Save model with error handling."""
            self.error_handler.save_quantized_model_safe(
                model,
                save_path,
                self.base_adapter
            )
        
        def load_quantized_model(
            self,
            model_path: str,
            model_class: type,
            device: Optional[str] = None,
            **kwargs
        ) -> nn.Module:
            """Load model with error handling."""
            return self.error_handler.load_quantized_model_safe(
                model_path,
                model_class,
                self.base_adapter,
                device=device
            )
        
        def get_memory_usage(self, model: nn.Module) -> dict[str, float]:
            """Get memory usage with error handling."""
            try:
                return self.base_adapter.get_memory_usage(model)
            except Exception as e:
                logger.error(f"Failed to get memory usage: {e}")
                return {"error": str(e)}
        
        def get_statistics(self) -> dict[str, Any]:
            """Get error statistics."""
            return self.error_handler.get_statistics()
    
    return SafeQuantizationAdapter()


# Export public API
__all__ = [
    'QuantizationErrorHandler',
    'create_safe_quantization_adapter',
]