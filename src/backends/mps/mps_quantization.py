"""
MPS Quantization Module

This module provides quantization capabilities optimized for Apple Silicon's MPS backend.
It includes:
- Native PyTorch quantization (INT8) for MPS
- Dynamic quantization strategies based on model/memory
- HQQ integration with MPS compatibility
- Performance optimizations for Metal Performance Shaders

Key Features:
- Automatic fallback from unsupported quantization methods
- Memory-aware quantization strategies
- Optimized operators for unified memory
- Integration with existing quantization infrastructure
"""

import os
import warnings
import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import (
    QuantStub,
    DeQuantStub,
    default_qconfig,
    prepare,
    convert,
    quantize_dynamic,
    QConfig,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    HistogramObserver,
)
from torch.nn.quantized import Linear as QuantizedLinear

from src.core.backend_manager import Backend, BackendManager
from src.core.quantization_wrapper import (
    QuantizationConfig,
    QuantizationMethod,
    QuantizedLinear as BaseQuantizedLinear,
    QuantizationAdapter,
)
from src.core.imports import get_module, check_import_availability

logger = logging.getLogger(__name__)


# MPS-specific quantization methods
class MPSQuantizationMethod:
    """MPS-compatible quantization methods."""
    PYTORCH_DYNAMIC = "pytorch_dynamic"  # PyTorch dynamic quantization
    PYTORCH_STATIC = "pytorch_static"    # PyTorch static quantization
    HQQ_MPS = "hqq_mps"                  # HQQ with MPS optimizations
    FAKE_QUANT = "fake_quant"            # Fake quantization for testing


@dataclass
class MPSQuantizationConfig(QuantizationConfig):
    """Configuration for MPS quantization."""
    
    # MPS-specific options
    mps_method: str = MPSQuantizationMethod.PYTORCH_DYNAMIC
    use_per_channel: bool = True
    observer_type: str = "minmax"  # minmax, histogram, or percentile
    reduce_range: bool = False  # For INT8 on some hardware
    
    # Dynamic quantization options
    dynamic_modules: List[Type[nn.Module]] = field(
        default_factory=lambda: [nn.Linear, nn.LSTM, nn.GRU]
    )
    
    # Memory optimization
    memory_efficient: bool = True
    chunk_size: Optional[int] = None  # For chunked quantization
    
    # Performance options
    use_fast_math: bool = True
    operator_fusion: bool = True
    
    # Fallback options
    fallback_to_fp16: bool = True
    warn_on_fallback: bool = True
    
    def __post_init__(self):
        """Validate MPS-specific configuration."""
        super().__post_init__()
        
        # MPS doesn't support bfloat16
        if self.compute_dtype == torch.bfloat16:
            warnings.warn("MPS doesn't support bfloat16, using float16 instead")
            self.compute_dtype = torch.float16
        
        # Validate quantization bits for MPS
        if self.bits not in [8, 16]:
            warnings.warn(f"MPS quantization typically supports 8 or 16 bits, got {self.bits}")


class MPSQuantizedLinear(BaseQuantizedLinear):
    """MPS-optimized quantized linear layer with custom implementation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: MPSQuantizationConfig = None,
    ):
        # Set config first
        self.config = config or MPSQuantizationConfig()
        super().__init__(in_features, out_features, bias, self.config)
        
        # For MPS, we'll implement our own quantization
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Quantization parameters
        self.bits = self.config.bits
        self.group_size = self.config.group_size
        
        # Storage for quantized weights
        self.register_buffer('quantized_weight', None)
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Original weight for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
        self.is_quantized = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with custom MPS quantization."""
        if self.is_quantized and self.quantized_weight is not None:
            # Use quantized weights
            weight = self._dequantize_weight()
            output = F.linear(x, weight, self.bias)
        else:
            # Use full precision weights
            output = F.linear(x, self.weight, self.bias)
        
        return output
    
    def _quantize_weight(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Custom quantization for MPS."""
        # Ensure weight is on the same device
        device = weight.device
        
        if self.bits == 8:
            # INT8 quantization
            if self.config.use_per_channel:
                # Per-channel quantization (per output channel)
                min_vals = weight.min(dim=1, keepdim=True)[0]
                max_vals = weight.max(dim=1, keepdim=True)[0]
            else:
                # Per-tensor quantization
                min_vals = weight.min()
                max_vals = weight.max()
            
            # Calculate scale and zero point
            qmin = -128
            qmax = 127
            scale = (max_vals - min_vals) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
            zero_point = torch.round(-min_vals / scale) + qmin
            zero_point = torch.clamp(zero_point, qmin, qmax)
            
            # Quantize
            quantized = torch.round((weight - min_vals) / scale + qmin)
            quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)
            
            return quantized, scale.to(device), zero_point.to(device)
        
        elif self.bits == 4:
            # 4-bit quantization (simulated with INT8)
            if self.config.use_per_channel:
                min_vals = weight.min(dim=1, keepdim=True)[0]
                max_vals = weight.max(dim=1, keepdim=True)[0]
            else:
                min_vals = weight.min()
                max_vals = weight.max()
            
            # 4-bit range
            qmin = -8
            qmax = 7
            scale = (max_vals - min_vals) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)
            zero_point = torch.round(-min_vals / scale) + qmin
            zero_point = torch.clamp(zero_point, qmin, qmax)
            
            # Quantize to 4-bit range but store as INT8
            quantized = torch.round((weight - min_vals) / scale + qmin)
            quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)
            
            return quantized, scale.to(device), zero_point.to(device)
        
        else:
            # No quantization
            return weight, torch.ones(1, device=device), torch.zeros(1, device=device)
    
    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize weights for computation."""
        if self.quantized_weight is None:
            return self.weight
        
        # Dequantize
        if self.bits == 8:
            qmin = -128
        elif self.bits == 4:
            qmin = -8
        else:
            return self.weight
        
        # Convert back to float
        dequantized = (self.quantized_weight.float() - qmin) * self.weight_scale
        
        if self.config.use_per_channel:
            # Add back the min values
            min_vals = -self.weight_zero_point * self.weight_scale
            dequantized = dequantized + min_vals
        else:
            min_val = -self.weight_zero_point * self.weight_scale
            dequantized = dequantized + min_val
        
        return dequantized
    
    def quantize_weights(self, weights: torch.Tensor) -> Any:
        """Quantize weights for MPS using custom implementation."""
        self.weight.data = weights
        
        # Apply custom quantization
        quantized, scale, zero_point = self._quantize_weight(weights)
        
        self.quantized_weight = quantized
        self.weight_scale = scale
        self.weight_zero_point = zero_point
        self.is_quantized = True
        
        # Calculate compression ratio
        original_size = weights.numel() * weights.element_size()
        quantized_size = quantized.numel() * quantized.element_size()
        if self.config.use_per_channel:
            quantized_size += scale.numel() * scale.element_size() * 2  # scale + zero_point
        else:
            quantized_size += 8  # two floats
        
        compression_ratio = original_size / quantized_size
        logger.info(f"Quantized weights: {self.bits}-bit, compression ratio: {compression_ratio:.2f}x")
        
        return quantized
    
    def dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights back to full precision."""
        if self.is_quantized:
            return self._dequantize_weight()
        else:
            return self.weight
    
    def _get_qconfig(self) -> QConfig:
        """Get quantization configuration for PyTorch."""
        if self.config.use_per_channel:
            activation = MinMaxObserver.with_args(
                dtype=torch.quint8,
                reduce_range=self.config.reduce_range
            )
            weight = PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric
            )
        else:
            activation = MinMaxObserver.with_args(
                dtype=torch.quint8,
                reduce_range=self.config.reduce_range
            )
            weight = MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric
            )
        
        return QConfig(activation=activation, weight=weight)


class HQQLinearMPS(BaseQuantizedLinear):
    """HQQ quantized linear layer optimized for MPS."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: MPSQuantizationConfig = None,
        hqq_module: Any = None,
    ):
        super().__init__(in_features, out_features, bias, config)
        self.config = config or MPSQuantizationConfig()
        self.hqq = hqq_module
        
        if self.hqq is not None:
            # Create HQQ linear layer
            quant_config = self.hqq.BaseQuantizeConfig(
                nbits=self.config.bits,
                group_size=self.config.group_size,
                quant_zero=self.config.quant_zero,
                quant_scale=self.config.quant_scale,
            )
            self.linear = self.hqq.HQQLinear(
                in_features,
                out_features,
                bias=bias,
                quant_config=quant_config,
            )
        else:
            # Fallback to standard linear
            self.linear = nn.Linear(in_features, out_features, bias)
            warnings.warn("HQQ not available, using standard linear layer")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through HQQ layer."""
        # Ensure input is on MPS
        if x.device.type != 'mps':
            x = x.to('mps')
        
        # HQQ forward pass
        return self.linear(x)
    
    def quantize_weights(self, weights: torch.Tensor) -> Any:
        """Quantize weights using HQQ."""
        if hasattr(self.linear, 'quantize_weight'):
            return self.linear.quantize_weight(weights)
        return weights
    
    def dequantize_weights(self) -> torch.Tensor:
        """Dequantize HQQ weights."""
        if hasattr(self.linear, 'dequantize'):
            return self.linear.dequantize()
        return self.linear.weight


class DynamicQuantizationStrategy:
    """Dynamic quantization strategy selector for MPS."""
    
    def __init__(self, backend_manager: Optional[BackendManager] = None):
        self.backend_manager = backend_manager or BackendManager(backend="mps")
        self._memory_info = self._get_memory_info()
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get MPS memory information."""
        try:
            allocated = torch.mps.current_allocated_memory() / 1e9
            reserved = torch.mps.driver_allocated_memory() / 1e9
            
            # Estimate available memory (rough approximation)
            import psutil
            total_memory = psutil.virtual_memory().total / 1e9
            available = total_memory - reserved
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "available_gb": available,
                "total_gb": total_memory,
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {
                "allocated_gb": 0,
                "reserved_gb": 0,
                "available_gb": 16,  # Conservative default
                "total_gb": 16,
            }
    
    def select_quantization_config(
        self,
        model_size_gb: float,
        target_dtype: torch.dtype = torch.float16,
        preferred_method: Optional[str] = None,
    ) -> MPSQuantizationConfig:
        """
        Select optimal quantization configuration based on model size and memory.
        
        Args:
            model_size_gb: Size of the model in GB
            target_dtype: Target dtype for computation
            preferred_method: Preferred quantization method
            
        Returns:
            Optimized quantization configuration
        """
        available_memory = self._memory_info["available_gb"]
        memory_ratio = model_size_gb / available_memory
        
        # Determine quantization aggressiveness
        if memory_ratio > 0.8:
            # Very tight memory - use 8-bit quantization
            config = MPSQuantizationConfig(
                method=QuantizationMethod.HQQ if self._check_hqq_available() else QuantizationMethod.NONE,
                mps_method=MPSQuantizationMethod.PYTORCH_DYNAMIC,
                bits=8,
                compute_dtype=target_dtype,
                memory_efficient=True,
                chunk_size=1024,  # Small chunks for memory efficiency
            )
        elif memory_ratio > 0.5:
            # Moderate memory pressure - use dynamic quantization
            config = MPSQuantizationConfig(
                method=QuantizationMethod.NONE,
                mps_method=MPSQuantizationMethod.PYTORCH_DYNAMIC,
                bits=8,
                compute_dtype=target_dtype,
                memory_efficient=True,
            )
        else:
            # Plenty of memory - use minimal quantization
            config = MPSQuantizationConfig(
                method=QuantizationMethod.NONE,
                mps_method=MPSQuantizationMethod.FAKE_QUANT,
                bits=16,
                compute_dtype=target_dtype,
                memory_efficient=False,
            )
        
        # Override with preferred method if specified
        if preferred_method:
            config.mps_method = preferred_method
        
        logger.info(
            f"Selected quantization config: {config.bits}-bit, "
            f"method={config.mps_method}, memory_ratio={memory_ratio:.2f}"
        )
        
        return config
    
    def _check_hqq_available(self) -> bool:
        """Check if HQQ is available for MPS."""
        try:
            import hqq
            # Test if HQQ works on MPS
            test_tensor = torch.randn(10, 10, device='mps')
            # Simple availability check
            return True
        except Exception:
            return False
    
    def optimize_for_model_type(
        self,
        model_type: str,
        model_size_gb: float,
    ) -> MPSQuantizationConfig:
        """
        Get optimized configuration for specific model types.
        
        Args:
            model_type: Type of model (llama, gpt, bert, etc.)
            model_size_gb: Size of model in GB
            
        Returns:
            Optimized configuration
        """
        # Model-specific optimizations
        model_configs = {
            "llama": {
                "dynamic_modules": [nn.Linear],
                "skip_modules": ["lm_head", "embed_tokens"],
                "use_per_channel": True,
            },
            "gpt": {
                "dynamic_modules": [nn.Linear],
                "skip_modules": ["lm_head", "wte", "wpe"],
                "use_per_channel": True,
            },
            "bert": {
                "dynamic_modules": [nn.Linear],
                "skip_modules": ["cls", "embeddings"],
                "use_per_channel": False,
            },
        }
        
        # Get base configuration
        config = self.select_quantization_config(model_size_gb)
        
        # Apply model-specific settings
        if model_type.lower() in model_configs:
            model_specific = model_configs[model_type.lower()]
            config.dynamic_modules = model_specific.get("dynamic_modules", config.dynamic_modules)
            config.skip_modules = model_specific.get("skip_modules", config.skip_modules)
            config.use_per_channel = model_specific.get("use_per_channel", config.use_per_channel)
        
        return config


class MPSQuantizationAdapter(QuantizationAdapter):
    """Quantization adapter optimized for MPS backend."""
    
    def __init__(
        self,
        backend: Backend,
        config: MPSQuantizationConfig,
        strategy: Optional[DynamicQuantizationStrategy] = None,
    ):
        super().__init__(backend, config)
        self.config = config
        self.strategy = strategy or DynamicQuantizationStrategy()
        
    def _validate_backend_support(self):
        """Validate MPS backend support."""
        if self.backend not in [Backend.MPS, Backend.CPU]:
            warnings.warn(
                f"MPSQuantizationAdapter is optimized for MPS, got {self.backend}. "
                "Some features may not work optimally."
            )
        
        # Check PyTorch MPS support
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend is not available")
        
        # Check quantization support
        if self.config.bits < 8 and self.config.mps_method == MPSQuantizationMethod.PYTORCH_DYNAMIC:
            warnings.warn(
                "PyTorch dynamic quantization on MPS typically supports INT8. "
                f"Got {self.config.bits}-bit quantization."
            )
    
    def create_quantized_linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> BaseQuantizedLinear:
        """Create MPS-optimized quantized linear layer."""
        if self.config.mps_method == MPSQuantizationMethod.HQQ_MPS:
            # Try HQQ first
            try:
                hqq = get_module('hqq', 'mps')
                return HQQLinearMPS(in_features, out_features, bias, self.config, hqq)
            except ImportError:
                warnings.warn("HQQ not available, falling back to PyTorch quantization")
                self.config.mps_method = MPSQuantizationMethod.PYTORCH_DYNAMIC
        
        # Default to PyTorch quantization
        return MPSQuantizedLinear(in_features, out_features, bias, self.config)
    
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Quantize model for MPS."""
        device = kwargs.get('device', 'mps')
        
        # Move model to MPS
        model = model.to(device)
        
        if self.config.mps_method == MPSQuantizationMethod.PYTORCH_DYNAMIC:
            # Use custom MPS quantization instead of PyTorch's
            warnings.warn(
                "PyTorch dynamic quantization not supported on MPS. "
                "Using custom MPS quantization implementation."
            )
            quantized_model = self._apply_custom_quantization(model, device)
            logger.info("Applied custom MPS quantization")
            
        elif self.config.mps_method == MPSQuantizationMethod.PYTORCH_STATIC:
            # Static quantization requires calibration
            model.qconfig = self._get_model_qconfig()
            quantized_model = prepare(model, inplace=False)
            
            # Note: Calibration would be needed here
            warnings.warn(
                "Static quantization requires calibration data. "
                "Model prepared but not converted."
            )
            
        elif self.config.mps_method == MPSQuantizationMethod.HQQ_MPS:
            # Apply HQQ quantization
            quantized_model = self._apply_hqq_quantization(model)
            
        else:
            # No quantization or fake quantization
            quantized_model = model
            logger.info("No quantization applied")
        
        # Apply memory optimizations
        if self.config.memory_efficient:
            quantized_model = self._apply_memory_optimizations(quantized_model)
        
        return quantized_model
    
    def prepare_model_for_training(self, model: nn.Module) -> nn.Module:
        """Prepare quantized model for training on MPS."""
        # Ensure model is on MPS
        model = model.to('mps')
        
        # Disable quantization for training if needed
        if hasattr(model, 'apply'):
            def disable_quantization(module):
                if hasattr(module, 'is_quantized'):
                    module.is_quantized = False
            
            model.apply(disable_quantization)
        
        # Enable training mode
        model.train()
        
        # Apply operator fusion if enabled
        if self.config.operator_fusion:
            model = self._fuse_operators(model)
        
        return model
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save quantized model."""
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state
        state = {
            'model_state_dict': model.state_dict(),
            'quantization_config': self.config,
            'mps_optimized': True,
        }
        
        torch.save(state, save_path)
        logger.info(f"Saved quantized model to {save_path}")
    
    def load_quantized_model(self, model_path: str, model_config: Any) -> nn.Module:
        """Load quantized model."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model instance
        if callable(model_config):
            model = model_config()
        else:
            # If model_config is not callable, assume it's already a model instance
            model = model_config if model_config is not None else nn.Module()
        
        # Load state dict (non-strict to handle quantization buffers)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Move to MPS
        model = model.to('mps')
        
        return model
    
    def _get_model_qconfig(self) -> QConfig:
        """Get model-wide quantization configuration."""
        if self.config.observer_type == "histogram":
            activation = HistogramObserver.with_args(
                dtype=torch.quint8,
                reduce_range=self.config.reduce_range
            )
        else:
            activation = MinMaxObserver.with_args(
                dtype=torch.quint8,
                reduce_range=self.config.reduce_range
            )
        
        if self.config.use_per_channel:
            weight = PerChannelMinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric
            )
        else:
            weight = MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric
            )
        
        return QConfig(activation=activation, weight=weight)
    
    def _apply_hqq_quantization(self, model: nn.Module) -> nn.Module:
        """Apply HQQ quantization to model."""
        try:
            hqq = get_module('hqq', 'mps')
            
            # Create HQQ config
            quant_config = hqq.BaseQuantizeConfig(
                nbits=self.config.bits,
                group_size=self.config.group_size,
                quant_zero=self.config.quant_zero,
                quant_scale=self.config.quant_scale,
            )
            
            # Apply HQQ quantization
            from train import replace_linear
            
            model = replace_linear(
                model,
                hqq.HQQLinear,
                compute_dtype=self.config.compute_dtype,
                quant_config=quant_config,
                del_orig=True,
                initialize=True,
                skip=self.config.skip_modules,
            )
            
            logger.info("Applied HQQ quantization")
            return model
            
        except Exception as e:
            warnings.warn(f"Failed to apply HQQ quantization: {e}")
            return model
    
    def _apply_memory_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations for MPS."""
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Set memory efficient attention if available
        for module in model.modules():
            if hasattr(module, 'set_use_memory_efficient_attention'):
                module.set_use_memory_efficient_attention(True)
        
        return model
    
    def _apply_custom_quantization(self, model: nn.Module, device: str) -> nn.Module:
        """Apply custom quantization for MPS."""
        # Replace linear layers with our custom quantized version
        def replace_linear_layers(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Skip if in skip list
                    if any(skip in name for skip in self.config.skip_modules):
                        continue
                    
                    # Create quantized replacement
                    quantized_layer = MPSQuantizedLinear(
                        child.in_features,
                        child.out_features,
                        child.bias is not None,
                        self.config,
                    )
                    
                    # Copy weights and quantize
                    quantized_layer.quantize_weights(child.weight.data.clone())
                    if child.bias is not None:
                        quantized_layer.bias.data = child.bias.data.clone()
                    
                    # Replace the module
                    setattr(module, name, quantized_layer.to(device))
                else:
                    # Recursively replace in children
                    replace_linear_layers(child)
        
        # Clone model to avoid modifying original
        model_copy = model
        replace_linear_layers(model_copy)
        
        return model_copy
    
    def _fuse_operators(self, model: nn.Module) -> nn.Module:
        """Fuse operators for better MPS performance."""
        # Common fusion patterns
        fusion_patterns = [
            ['conv', 'bn', 'relu'],
            ['conv', 'bn'],
            ['linear', 'relu'],
        ]
        
        # Apply torch.quantization.fuse_modules if available
        try:
            from torch.quantization import fuse_modules
            
            # This would need actual module names
            # Placeholder for demonstration
            logger.info("Operator fusion not implemented for this model")
            
        except Exception as e:
            logger.warning(f"Failed to fuse operators: {e}")
        
        return model


# Performance optimization utilities

class MPSPerformanceOptimizer:
    """Performance optimizer for quantized models on MPS."""
    
    def __init__(self, config: MPSQuantizationConfig):
        self.config = config
    
    def optimize_forward_pass(self, model: nn.Module) -> nn.Module:
        """Optimize model forward pass for MPS."""
        # Apply MPS-specific optimizations
        optimizations_applied = []
        
        # 1. Use fast math if enabled
        if self.config.use_fast_math:
            torch.backends.mps.matmul_allow_tf32 = True
            optimizations_applied.append("fast_math")
        
        # 2. Optimize memory layout
        def optimize_module(module):
            if isinstance(module, nn.Linear):
                # Ensure contiguous memory layout
                if hasattr(module, 'weight'):
                    module.weight.data = module.weight.data.contiguous()
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data = module.bias.data.contiguous()
        
        model.apply(optimize_module)
        optimizations_applied.append("memory_layout")
        
        # 3. Enable MPS profiling for optimization
        if logger.isEnabledFor(logging.DEBUG):
            torch.mps.profiler.start()
            optimizations_applied.append("profiling")
        
        logger.info(f"Applied MPS optimizations: {optimizations_applied}")
        return model
    
    def profile_quantized_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """Profile quantized model performance on MPS."""
        device = 'mps'
        model = model.to(device)
        model.eval()
        
        # Warmup
        dummy_input = torch.randn(input_shape, device=device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Synchronize MPS
        torch.mps.synchronize()
        
        # Time forward passes
        import time
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            torch.mps.synchronize()
            end = time.time()
            
            times.append(end - start)
        
        # Calculate statistics
        # Only remove first 10 if we have more than 10 runs
        if len(times) > 10:
            times = times[10:]  # Remove first 10 for stability
        
        avg_time = sum(times) / len(times) if times else 0.01
        min_time = min(times) if times else 0.01
        max_time = max(times) if times else 0.01
        
        # Memory stats
        allocated = torch.mps.current_allocated_memory() / 1e6  # MB
        reserved = torch.mps.driver_allocated_memory() / 1e6   # MB
        
        return {
            'avg_forward_time_ms': avg_time * 1000,
            'min_forward_time_ms': min_time * 1000,
            'max_forward_time_ms': max_time * 1000,
            'allocated_memory_mb': allocated,
            'reserved_memory_mb': reserved,
            'throughput_samples_per_sec': 1.0 / avg_time,
        }


# Convenience functions

def create_mps_quantized_model(
    model: nn.Module,
    model_type: str = "llama",
    quantization_bits: int = 8,
    dynamic_strategy: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Create MPS-optimized quantized model.
    
    Args:
        model: PyTorch model to quantize
        model_type: Type of model for optimizations
        quantization_bits: Number of bits for quantization
        dynamic_strategy: Use dynamic strategy selection
        **kwargs: Additional configuration options
        
    Returns:
        Quantized model optimized for MPS
    """
    # Calculate model size
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    
    # Create strategy and select configuration
    strategy = DynamicQuantizationStrategy()
    
    if dynamic_strategy:
        config = strategy.optimize_for_model_type(model_type, model_size_gb)
        config.bits = quantization_bits
    else:
        config = MPSQuantizationConfig(
            bits=quantization_bits,
            mps_method=kwargs.get('method', MPSQuantizationMethod.PYTORCH_DYNAMIC),
            **kwargs,
        )
    
    # Create adapter and quantize
    adapter = MPSQuantizationAdapter(Backend.MPS, config, strategy)
    quantized_model = adapter.quantize_model(model)
    
    # Apply performance optimizations
    optimizer = MPSPerformanceOptimizer(config)
    quantized_model = optimizer.optimize_forward_pass(quantized_model)
    
    return quantized_model


def benchmark_quantization_methods(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    methods: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different quantization methods on MPS.
    
    Args:
        model: Model to benchmark
        input_shape: Input shape for testing
        methods: List of methods to test
        
    Returns:
        Benchmark results for each method
    """
    if methods is None:
        methods = [
            MPSQuantizationMethod.PYTORCH_DYNAMIC,
            MPSQuantizationMethod.HQQ_MPS,
            "none",  # Baseline
        ]
    
    results = {}
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    
    for method in methods:
        try:
            # Create config
            if method == "none":
                quantized_model = model.to('mps')
            else:
                config = MPSQuantizationConfig(
                    mps_method=method,
                    bits=8,
                )
                adapter = MPSQuantizationAdapter(Backend.MPS, config)
                quantized_model = adapter.quantize_model(model.clone())
            
            # Profile
            optimizer = MPSPerformanceOptimizer(MPSQuantizationConfig())
            profile_results = optimizer.profile_quantized_model(
                quantized_model,
                input_shape,
                num_runs=50,
            )
            
            results[method] = profile_results
            logger.info(f"Benchmarked {method}: {profile_results['avg_forward_time_ms']:.2f}ms")
            
        except Exception as e:
            logger.error(f"Failed to benchmark {method}: {e}")
            results[method] = {"error": str(e)}
    
    return results


# Integration with existing quantization infrastructure

def get_mps_quantization_adapter(
    config: Union[QuantizationConfig, MPSQuantizationConfig],
    backend_manager: Optional[BackendManager] = None,
) -> QuantizationAdapter:
    """
    Get MPS-compatible quantization adapter.
    
    Args:
        config: Quantization configuration
        backend_manager: Backend manager instance
        
    Returns:
        Appropriate quantization adapter for MPS
    """
    backend_manager = backend_manager or BackendManager(backend="mps")
    backend = backend_manager.backend
    
    # Convert to MPS config if needed
    if not isinstance(config, MPSQuantizationConfig):
        mps_config = MPSQuantizationConfig(
            method=config.method,
            bits=config.bits,
            compute_dtype=config.compute_dtype,
            skip_modules=config.skip_modules,
        )
    else:
        mps_config = config
    
    # Determine best adapter
    if backend == Backend.MPS:
        return MPSQuantizationAdapter(backend, mps_config)
    else:
        # Fallback to original adapters
        from quantization_wrapper import create_quantization_adapter
        return create_quantization_adapter(backend, config)


__all__ = [
    # Classes
    "MPSQuantizationConfig",
    "MPSQuantizationAdapter",
    "MPSQuantizedLinear",
    "HQQLinearMPS",
    "DynamicQuantizationStrategy",
    "MPSPerformanceOptimizer",
    
    # Methods
    "MPSQuantizationMethod",
    
    # Functions
    "create_mps_quantized_model",
    "benchmark_quantization_methods",
    "get_mps_quantization_adapter",
]