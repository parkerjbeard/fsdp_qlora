"""
Unified Quantization API

Provides a single interface that automatically selects the best quantization
backend based on hardware, model type, and requirements.

Supported backends:
- MLX (optimal for Apple Silicon)
- Quanto (PyTorch with better MPS support)
- Custom MPS (fallback for basic PyTorch)
"""

import os
import platform
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from enum import Enum

import torch
import torch.nn as nn

from src.core.backend_manager import Backend

logger = logging.getLogger(__name__)


class QuantizationBackend(Enum):
    """Available quantization backends."""
    MLX = "mlx"          # Apple's MLX (best for Apple Silicon)
    QUANTO = "quanto"     # HuggingFace Quanto (good MPS support)
    MPS_CUSTOM = "mps_custom"  # Custom MPS implementation
    AUTO = "auto"        # Automatically select best backend


@dataclass
class UnifiedQuantizationConfig:
    """Unified configuration for all quantization backends."""
    
    # Backend selection
    backend: QuantizationBackend = QuantizationBackend.AUTO
    
    # Common quantization settings
    bits: int = 4
    group_size: int = 64
    skip_modules: list[str] = field(default_factory=lambda: ["lm_head"])
    
    # Mixed precision
    layer_bits: dict[str, int] = field(default_factory=dict)
    embedding_bits: Optional[int] = None
    output_bits: Optional[int] = None
    
    # Training settings
    enable_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Performance settings
    memory_efficient: bool = True
    calibration_samples: int = 100
    calibration_data: Optional[Any] = None
    
    # Backend-specific overrides
    backend_config: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and set defaults."""
        # Set embedding/output bits if not specified
        if self.embedding_bits is None:
            self.embedding_bits = min(8, self.bits * 2)
        if self.output_bits is None:
            self.output_bits = min(8, self.bits * 2)


class BackendSelector:
    """Automatically select the best quantization backend."""
    
    @staticmethod
    def detect_hardware() -> dict[str, Any]:
        """Detect hardware capabilities."""
        info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
        
        # Check for Apple Silicon
        info["is_apple_silicon"] = (
            info["platform"] == "Darwin" and 
            info["machine"] == "arm64"
        )
        
        # Check PyTorch backends
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = torch.backends.mps.is_available()
        
        # Check MLX availability
        try:
            import mlx  # noqa: F401
            info["mlx_available"] = True
        except ImportError:
            info["mlx_available"] = False
        
        # Check Quanto availability
        try:
            import optimum.quanto  # noqa: F401
            info["quanto_available"] = True
        except ImportError:
            info["quanto_available"] = False
        
        return info
    
    @staticmethod
    def select_backend(
        config: UnifiedQuantizationConfig,
        model_size_gb: Optional[float] = None,
    ) -> QuantizationBackend:
        """Select the best backend based on hardware and requirements."""
        if config.backend != QuantizationBackend.AUTO:
            return config.backend
        
        hardware = BackendSelector.detect_hardware()
        
        # Decision logic
        if hardware["is_apple_silicon"]:
            # On Apple Silicon, prefer MLX
            if hardware["mlx_available"]:
                logger.info("Selected MLX backend (optimal for Apple Silicon)")
                return QuantizationBackend.MLX
            elif hardware["quanto_available"]:
                logger.info("Selected Quanto backend (MLX not available)")
                return QuantizationBackend.QUANTO
            else:
                logger.warning(
                    "Neither MLX nor Quanto available, "
                    "falling back to custom MPS implementation"
                )
                return QuantizationBackend.MPS_CUSTOM
        
        elif hardware["mps_available"]:
            # On Intel Mac with MPS
            if hardware["quanto_available"]:
                logger.info("Selected Quanto backend for MPS")
                return QuantizationBackend.QUANTO
            else:
                logger.info("Selected custom MPS backend")
                return QuantizationBackend.MPS_CUSTOM
        
        else:
            # Non-Mac systems
            raise ValueError(
                "This module is designed for Apple Silicon/MPS. "
                "For CUDA, use standard bitsandbytes or HQQ."
            )


class UnifiedQuantizer:
    """Unified quantization interface."""
    
    def __init__(self, config: UnifiedQuantizationConfig):
        self.config = config
        self.backend = BackendSelector.select_backend(config)
        self._adapter = None
        
        # Update config backend if it was auto
        if config.backend == QuantizationBackend.AUTO:
            config.backend = self.backend
        
        # Initialize the appropriate adapter
        self._initialize_adapter()
    
    def _initialize_adapter(self):
        """Initialize the backend-specific adapter."""
        if self.backend == QuantizationBackend.MLX:
            self._initialize_mlx_adapter()
        elif self.backend == QuantizationBackend.QUANTO:
            self._initialize_quanto_adapter()
        elif self.backend == QuantizationBackend.MPS_CUSTOM:
            self._initialize_mps_adapter()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _initialize_mlx_adapter(self):
        """Initialize MLX adapter."""
        try:
            from src.backends.mlx.mlx_quantization import MLXQuantizer, MLXQuantizationConfig
            
            # Convert config to MLX format
            mlx_config = MLXQuantizationConfig(
                bits=self.config.bits,
                group_size=self.config.group_size,
                layer_bits=self.config.layer_bits,
                skip_modules=self.config.skip_modules,
                default_bits=self.config.bits,
                embedding_bits=self.config.embedding_bits,
                output_bits=self.config.output_bits,
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_target_modules=self.config.lora_target_modules,
                **self.config.backend_config.get("mlx", {}),
            )
            
            self._adapter = MLXQuantizer(mlx_config)
            self._framework = "mlx"
            
        except ImportError:
            raise ImportError(
                "MLX backend selected but not available. "
                "Install with: pip install mlx mlx-lm"
            )
    
    def _initialize_quanto_adapter(self):
        """Initialize Quanto adapter."""
        try:
            from src.backends.mps.mps_quantization_quanto import (
                QuantoQuantizationAdapter,
                QuantoConfig,
                QuantoQuantizationMethod,
            )
            
            # Map bits to Quanto method
            bit_map = {
                2: QuantoQuantizationMethod.INT2,
                4: QuantoQuantizationMethod.INT4,
                8: QuantoQuantizationMethod.INT8,
            }
            
            # Convert config to Quanto format
            quanto_config = QuantoConfig(
                quanto_method=bit_map.get(self.config.bits, QuantoQuantizationMethod.INT4),
                calibration_samples=self.config.calibration_samples,
                calibration_dataset=self.config.calibration_data,
                skip_modules=self.config.skip_modules,
                **self.config.backend_config.get("quanto", {}),
            )
            
            self._adapter = QuantoQuantizationAdapter(
                Backend.MPS,
                quanto_config,
            )
            self._framework = "pytorch"
            
        except ImportError:
            raise ImportError(
                "Quanto backend selected but not available. "
                "Install with: pip install optimum-quanto"
            )
    
    def _initialize_mps_adapter(self):
        """Initialize custom MPS adapter."""
        from src.backends.mps.mps_quantization import (
            MPSQuantizationAdapter,
            MPSQuantizationConfig,
            MPSQuantizationMethod,
        )
        
        # Convert config to MPS format
        mps_config = MPSQuantizationConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            skip_modules=self.config.skip_modules,
            mps_method=MPSQuantizationMethod.PYTORCH_DYNAMIC,
            memory_efficient=self.config.memory_efficient,
            **self.config.backend_config.get("mps", {}),
        )
        
        self._adapter = MPSQuantizationAdapter(
            Backend.MPS,
            mps_config,
        )
        self._framework = "pytorch"
    
    def quantize_model(
        self,
        model: Union[nn.Module, Any],
        device: Optional[str] = None,
        **kwargs,
    ) -> Union[nn.Module, Any]:
        """Quantize a model using the selected backend."""
        # Handle framework conversion if needed
        if self._framework == "mlx" and isinstance(model, nn.Module):
            # Convert PyTorch model to MLX
            logger.info("Converting PyTorch model to MLX...")
            model = self._convert_pytorch_to_mlx(model)
        
        # Quantize using backend adapter
        if hasattr(self._adapter, 'quantize_model'):
            quantized_model = self._adapter.quantize_model(
                model,
                device=device,
                **kwargs,
            )
        else:
            # Implement generic quantization fallback
            logger.warning(f"Using fallback quantization for {self.backend}")
            quantized_model = self._fallback_quantize_model(model, device)
        
        # Add LoRA if requested
        if self.config.enable_lora:
            quantized_model = self._add_lora_adapters(quantized_model)
        
        return quantized_model
    
    def quantize(
        self,
        tensor: torch.Tensor,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
        symmetric: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Quantize a tensor to specified bit width.
        
        Args:
            tensor: Input tensor to quantize
            bits: Quantization bits (defaults to config.bits)
            group_size: Group size for quantization (defaults to config.group_size)
            symmetric: Use symmetric quantization
            
        Returns:
            Tuple of (quantized_tensor, quantization_params)
        """
        bits = bits or self.config.bits
        group_size = group_size or self.config.group_size
        
        # Validate bit width
        if bits not in [4, 8, 16]:
            raise ValueError(f"Unsupported bit width: {bits}. Supported: 4, 8, 16")
        
        # Apply group-wise quantization if needed
        if group_size is not None and group_size > 0:
            return self._group_wise_quantize(tensor, bits, group_size, symmetric)
        else:
            return self._tensor_wise_quantize(tensor, bits, symmetric)
    
    def dequantize(
        self,
        quantized_tensor: torch.Tensor,
        quantization_params: dict[str, torch.Tensor],
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Dequantize a tensor back to floating point.
        
        Args:
            quantized_tensor: Quantized tensor
            quantization_params: Quantization parameters (scales, zero_points, etc.)
            dtype: Output dtype (defaults to float32)
            
        Returns:
            Dequantized tensor
        """
        dtype = dtype or torch.float32
        
        # Extract parameters
        scales = quantization_params.get("scales")
        zero_points = quantization_params.get("zero_points")
        group_size = quantization_params.get("group_size")
        
        if scales is None:
            raise ValueError("Missing scales in quantization parameters")
        
        # Handle group-wise dequantization
        if group_size is not None and scales.numel() > 1:
            # Group-wise dequantization
            original_shape = quantization_params.get("original_shape", quantized_tensor.shape)
            quantized_flat = quantized_tensor.flatten()
            
            # Pad if necessary
            n_groups = scales.numel()
            expected_size = n_groups * group_size
            if quantized_flat.numel() < expected_size:
                pad_size = expected_size - quantized_flat.numel()
                quantized_flat = torch.nn.functional.pad(quantized_flat, (0, pad_size))
            
            # Reshape to groups
            quantized_grouped = quantized_flat.reshape(n_groups, group_size)
            
            # Dequantize each group
            dequantized_groups = []
            for i, group in enumerate(quantized_grouped):
                group_scale = scales[i]
                if zero_points is not None:
                    group_zero_point = zero_points[i]
                    dequantized_group = (group.to(dtype) - group_zero_point.to(dtype)) * group_scale.to(dtype)
                else:
                    dequantized_group = group.to(dtype) * group_scale.to(dtype)
                dequantized_groups.append(dequantized_group)
            
            # Combine and reshape
            dequantized = torch.cat(dequantized_groups)
            dequantized = dequantized[:torch.prod(torch.tensor(original_shape)).item()]
            dequantized = dequantized.reshape(original_shape)
        else:
            # Tensor-wise dequantization
            if zero_points is not None:
                # Asymmetric quantization
                dequantized = (quantized_tensor.to(dtype) - zero_points.to(dtype)) * scales.to(dtype)
            else:
                # Symmetric quantization
                dequantized = quantized_tensor.to(dtype) * scales.to(dtype)
        
        return dequantized
    
    def _tensor_wise_quantize(
        self,
        tensor: torch.Tensor,
        bits: int,
        symmetric: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Quantize entire tensor with single scale/zero-point."""
        # Calculate quantization range
        if symmetric:
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** bits - 1
        
        # Calculate scale and zero-point
        if symmetric:
            max_val = torch.max(torch.abs(tensor))
            # Avoid division by zero
            if max_val == 0:
                scale = torch.ones(1, device=tensor.device, dtype=tensor.dtype)
            else:
                scale = max_val / max(abs(qmin), abs(qmax))
            zero_point = torch.zeros(1, device=tensor.device, dtype=tensor.dtype)
        else:
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - torch.round(min_val / scale)
        
        # Quantize
        if symmetric:
            quantized = torch.round(tensor / scale)
        else:
            quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Convert to appropriate dtype
        if bits == 8:
            quantized = quantized.to(torch.int8 if symmetric else torch.uint8)
        elif bits == 4:
            # Pack 4-bit values (simplified - real implementation would pack properly)
            quantized = quantized.to(torch.int8)
        else:  # 16-bit
            quantized = quantized.to(torch.int16)
        
        params = {
            "scales": scale,
            "zero_points": zero_point if not symmetric else None,
            "bits": bits,
            "symmetric": symmetric,
        }
        
        return quantized, params
    
    def _group_wise_quantize(
        self,
        tensor: torch.Tensor,
        bits: int,
        group_size: int,
        symmetric: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Quantize tensor with group-wise quantization."""
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        n_groups = (tensor_flat.numel() + group_size - 1) // group_size
        
        # Pad if necessary
        if tensor_flat.numel() % group_size != 0:
            pad_size = n_groups * group_size - tensor_flat.numel()
            tensor_flat = torch.nn.functional.pad(tensor_flat, (0, pad_size))
        
        # Reshape into groups
        tensor_grouped = tensor_flat.reshape(n_groups, group_size)
        
        # Quantize each group
        quantized_groups = []
        scales = []
        zero_points = []
        
        for group in tensor_grouped:
            q_group, params = self._tensor_wise_quantize(group, bits, symmetric)
            quantized_groups.append(q_group)
            scales.append(params["scales"])
            if not symmetric and params["zero_points"] is not None:
                zero_points.append(params["zero_points"])
        
        # Combine results
        quantized = torch.cat(quantized_groups)
        
        # Remove padding and reshape
        if tensor_flat.numel() != tensor.numel():
            quantized = quantized[:tensor.numel()]
        quantized = quantized.reshape(original_shape)
        
        params = {
            "scales": torch.stack(scales),
            "zero_points": torch.stack(zero_points) if zero_points and not symmetric else None,
            "bits": bits,
            "symmetric": symmetric,
            "group_size": group_size,
            "original_shape": original_shape,
        }
        
        return quantized, params
    
    def _fallback_quantize_model(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ) -> nn.Module:
        """Fallback quantization implementation."""
        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)
        
        # Find and quantize linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Skip if in skip_modules
                if any(skip in name for skip in self.config.skip_modules):
                    continue
                
                # Get bit width for this layer
                bits = self.config.bits
                for pattern, layer_bits in self.config.layer_bits.items():
                    if pattern in name:
                        bits = layer_bits
                        break
                
                # Quantize weights
                with torch.no_grad():
                    quantized_weight, quant_params = self.quantize(
                        module.weight.data,
                        bits=bits,
                        group_size=self.config.group_size,
                    )
                    
                    # Store quantization info in module
                    module.register_buffer("quantized_weight", quantized_weight)
                    for key, value in quant_params.items():
                        if value is not None:
                            # Convert non-tensor values to tensors
                            if not isinstance(value, torch.Tensor):
                                if isinstance(value, (int, float)):
                                    value = torch.tensor(value)
                                elif isinstance(value, bool):
                                    value = torch.tensor(value, dtype=torch.bool)
                                elif isinstance(value, (list, tuple)):
                                    value = torch.tensor(value)
                            module.register_buffer(f"quant_{key}", value)
                    
                    # Create a custom forward method with access to quantizer
                    quantizer = self
                    
                    def make_quantized_forward(original_module):
                        def quantized_forward(x):
                            # Dequantize weights
                            quant_params = {}
                            for key in ["scales", "zero_points", "bits", "symmetric", "group_size", "original_shape"]:
                                if hasattr(original_module, f"quant_{key}"):
                                    quant_params[key] = getattr(original_module, f"quant_{key}")
                            
                            weight = quantizer.dequantize(
                                original_module.quantized_weight,
                                quant_params,
                            )
                            return torch.nn.functional.linear(x, weight, original_module.bias)
                        return quantized_forward
                    
                    # Replace forward method
                    module.forward = make_quantized_forward(module)
        
        return model
    
    def _convert_pytorch_to_mlx(self, model: nn.Module) -> Any:
        """Convert PyTorch model to MLX."""
        try:
            from pytorch_mlx_bridge import convert_huggingface_to_mlx
            
            # Save model temporarily
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                # This is simplified - real implementation would handle
                # different model types
                mlx_model, _ = convert_huggingface_to_mlx(
                    model,
                    tmpdir,
                    quantize=False,  # We'll quantize separately
                )
            
            return mlx_model
            
        except Exception as e:
            logger.error(f"Failed to convert model to MLX: {e}")
            raise
    
    def _add_lora_adapters(self, model: Any) -> Any:
        """Add LoRA adapters for fine-tuning."""
        if self._framework == "mlx":
            # MLX LoRA
            if hasattr(self._adapter, 'add_lora_adapters'):
                return self._adapter.add_lora_adapters(
                    model,
                    target_modules=self.config.lora_target_modules,
                )
        else:
            # PyTorch LoRA (would need PEFT integration)
            logger.warning(
                "LoRA not yet implemented for PyTorch backends. "
                "Use PEFT library separately."
            )
        
        return model
    
    def save_model(self, model: Any, save_path: str, **kwargs):
        """Save quantized model."""
        if hasattr(self._adapter, 'save_quantized_model'):
            self._adapter.save_quantized_model(model, save_path, **kwargs)
        else:
            # Generic save with quantization info
            os.makedirs(save_path, exist_ok=True)
            
            if self._framework == "pytorch":
                # Save model state and quantization config
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "quantization_config": self.config,
                    "backend": self.backend.value,
                    "framework": self._framework,
                }
                
                # Add metadata
                checkpoint["metadata"] = {
                    "bits": self.config.bits,
                    "group_size": self.config.group_size,
                    "skip_modules": self.config.skip_modules,
                    "layer_bits": self.config.layer_bits,
                }
                
                torch.save(
                    checkpoint,
                    os.path.join(save_path, "quantized_model.pt"),
                )
                logger.info(f"Saved quantized model to {save_path}")
            else:
                logger.warning(f"Generic save not fully implemented for {self._framework}")
    
    def load_model(
        self,
        load_path: str,
        model_class: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Load quantized model."""
        if hasattr(self._adapter, 'load_quantized_model'):
            return self._adapter.load_quantized_model(
                load_path,
                model_class,
                **kwargs,
            )
        else:
            # Implement fallback loading
            logger.warning(f"Using fallback model loading for {self.backend}")
            return self._fallback_load_model(load_path, model_class, **kwargs)
    
    def _fallback_load_model(
        self,
        load_path: str,
        model_class: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Fallback model loading implementation."""
        if self._framework == "pytorch":
            # Load PyTorch model
            import os
            
            if model_class is None:
                raise ValueError("model_class is required for fallback loading")
            
            # Initialize model
            model = model_class(**kwargs)
            
            # Load state dict
            state_path = os.path.join(load_path, "model.pt")
            if os.path.exists(state_path):
                state_dict = torch.load(state_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
            else:
                # Try loading quantized checkpoint
                quant_path = os.path.join(load_path, "quantized_model.pt")
                if os.path.exists(quant_path):
                    checkpoint = torch.load(quant_path, map_location="cpu")
                    
                    # Load model state
                    if "model_state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    
                    # Load quantization parameters
                    if "quantization_config" in checkpoint:
                        self.config = checkpoint["quantization_config"]
                else:
                    raise FileNotFoundError(f"No model files found in {load_path}")
            
            # Re-quantize if needed
            if hasattr(model, "quantized_weight"):
                # Model is already quantized
                return model
            else:
                # Quantize the loaded model
                return self.quantize_model(model)
        else:
            # For unsupported frameworks, provide a helpful error message
            supported_frameworks = ["pytorch", "tensorflow", "jax"]
            raise ValueError(
                f"Framework '{self._framework}' is not supported for model loading. "
                f"Supported frameworks: {', '.join(supported_frameworks)}. "
                f"Please use the appropriate framework-specific loading method."
            )
    
    def benchmark(
        self,
        model: Any,
        input_shape: tuple[int, ...],
        num_runs: int = 100,
    ) -> dict[str, float]:
        """Benchmark quantized model performance."""
        import time
        import numpy as np
        
        device = "mps" if self._framework == "pytorch" else None
        
        # Warmup
        for _ in range(10):
            if self._framework == "pytorch":
                dummy_input = torch.randn(input_shape).to(device)
                with torch.no_grad():
                    _ = model(dummy_input)
            else:
                # MLX
                import mlx.core as mx
                dummy_input = mx.random.normal(input_shape)
                _ = model(dummy_input)
                mx.eval(model)
        
        # Timing
        times = []
        for _ in range(num_runs):
            start = time.time()
            
            if self._framework == "pytorch":
                dummy_input = torch.randn(input_shape).to(device)
                with torch.no_grad():
                    _ = model(dummy_input)
                if device == "mps":
                    torch.mps.synchronize()
            else:
                # MLX
                import mlx.core as mx
                dummy_input = mx.random.normal(input_shape)
                _ = model(dummy_input)
                mx.eval(model)
            
            end = time.time()
            times.append(end - start)
        
        # Calculate stats
        times = np.array(times[10:])  # Remove first 10 for stability
        
        return {
            "backend": self.backend.value,
            "framework": self._framework,
            "avg_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
            "throughput_samples_per_sec": 1.0 / np.mean(times),
        }


def quantize_model(
    model: Union[nn.Module, str],
    bits: int = 4,
    backend: Union[str, QuantizationBackend] = QuantizationBackend.AUTO,
    enable_lora: bool = False,
    **kwargs,
) -> tuple[Any, UnifiedQuantizer]:
    """
    Quantize a model with automatic backend selection.
    
    Args:
        model: PyTorch model or HuggingFace model ID
        bits: Quantization bits (2, 4, or 8)
        backend: Backend to use (auto selects best)
        enable_lora: Enable LoRA adapters for fine-tuning
        **kwargs: Additional configuration
        
    Returns:
        Tuple of (quantized_model, quantizer)
    """
    # Create configuration
    if isinstance(backend, str):
        backend = QuantizationBackend(backend)
    
    config = UnifiedQuantizationConfig(
        backend=backend,
        bits=bits,
        enable_lora=enable_lora,
        **kwargs,
    )
    
    # Create quantizer
    quantizer = UnifiedQuantizer(config)
    
    # Load model if needed
    if isinstance(model, str):
        # Load from HuggingFace
        if quantizer._framework == "mlx":
            from mlx_lm import load
            model, tokenizer = load(model)
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model)
    
    # Quantize
    quantized_model = quantizer.quantize_model(model)
    
    return quantized_model, quantizer


def compare_backends(
    model: Union[nn.Module, str],
    input_shape: tuple[int, ...],
    bits: int = 4,
) -> dict[str, dict[str, float]]:
    """
    Compare different quantization backends.
    
    Args:
        model: Model to quantize
        input_shape: Input shape for benchmarking
        bits: Quantization bits
        
    Returns:
        Benchmark results for each backend
    """
    results = {}
    hardware = BackendSelector.detect_hardware()
    
    # Test available backends
    backends_to_test = []
    
    if hardware["mlx_available"]:
        backends_to_test.append(QuantizationBackend.MLX)
    if hardware["quanto_available"]:
        backends_to_test.append(QuantizationBackend.QUANTO)
    if hardware["mps_available"]:
        backends_to_test.append(QuantizationBackend.MPS_CUSTOM)
    
    for backend in backends_to_test:
        try:
            logger.info(f"\nTesting {backend.value} backend...")
            
            # Quantize with this backend
            quantized_model, quantizer = quantize_model(
                model,
                bits=bits,
                backend=backend,
            )
            
            # Benchmark
            results[backend.value] = quantizer.benchmark(
                quantized_model,
                input_shape,
            )
            
        except Exception as e:
            logger.error(f"Failed to test {backend.value}: {e}")
            results[backend.value] = {"error": str(e)}
    
    return results


__all__ = [
    'QuantizationBackend',
    'UnifiedQuantizationConfig',
    'UnifiedQuantizer',
    'quantize_model',
    'compare_backends',
]