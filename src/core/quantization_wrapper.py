"""
Quantization Abstraction Layer for FSDP QLoRA

This module provides a unified interface for quantization across different backends
(bitsandbytes, HQQ, MLX). It abstracts the differences between quantization libraries
and provides a consistent API for quantizing models.
"""

import enum
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.core.backend_manager import Backend
from src.core.imports import get_module, check_import_availability


class QuantizationMethod(enum.Enum):
    """Supported quantization methods."""
    BNB_NF4 = "bnb_nf4"       # bitsandbytes Normal Float 4-bit
    BNB_INT8 = "bnb_int8"     # bitsandbytes INT8
    HQQ = "hqq"               # Half-Quadratic Quantization
    MLX_INT4 = "mlx_int4"     # MLX 4-bit integer
    MLX_INT8 = "mlx_int8"     # MLX 8-bit integer
    QUANTO_INT2 = "quanto_int2"  # Quanto 2-bit integer
    QUANTO_INT4 = "quanto_int4"  # Quanto 4-bit integer
    QUANTO_INT8 = "quanto_int8"  # Quanto 8-bit integer
    NONE = "none"             # No quantization


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    
    method: QuantizationMethod = QuantizationMethod.BNB_NF4
    bits: int = 4
    group_size: int = 64
    compute_dtype: Optional[torch.dtype] = None
    storage_dtype: Optional[torch.dtype] = None
    double_quant: bool = True
    quant_type: str = "nf4"  # nf4 or fp4 for bitsandbytes
    block_size: int = 64     # For HQQ
    quant_zero: bool = True  # For HQQ
    quant_scale: bool = False  # For HQQ
    
    # Layer-specific configurations
    layer_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    skip_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    
    # Backend-specific options
    backend_options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.bits not in [2, 4, 8, 16]:
            raise ValueError(f"Unsupported bit width: {self.bits}. Supported: 2, 4, 8, 16")
        
        if self.compute_dtype is None:
            self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        if self.storage_dtype is None:
            self.storage_dtype = torch.uint8 if self.bits <= 8 else torch.float16


class QuantizedLinear(nn.Module, ABC):
    """Base class for quantized linear layers."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, config: QuantizationConfig = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantized layer."""
        pass
    
    @abstractmethod
    def quantize_weights(self, weights: torch.Tensor) -> Any:
        """Quantize the given weights."""
        pass
    
    @abstractmethod
    def dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights back to full precision."""
        pass


class QuantizationAdapter(ABC):
    """Abstract base class for backend-specific quantization adapters."""
    
    def __init__(self, backend: Backend, config: QuantizationConfig):
        self.backend = backend
        self.config = config
        self._validate_backend_support()
    
    @abstractmethod
    def _validate_backend_support(self):
        """Validate that the backend supports the requested quantization."""
        pass
    
    @abstractmethod
    def create_quantized_linear(self, in_features: int, out_features: int, bias: bool = True) -> QuantizedLinear:
        """Create a quantized linear layer."""
        pass
    
    @abstractmethod
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Quantize an entire model."""
        pass
    
    @abstractmethod
    def prepare_model_for_training(self, model: nn.Module) -> nn.Module:
        """Prepare a quantized model for training (e.g., QLoRA)."""
        pass
    
    @abstractmethod
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save a quantized model."""
        pass
    
    @abstractmethod
    def load_quantized_model(self, model_path: str, model_config: Any) -> nn.Module:
        """Load a quantized model."""
        pass


class BitsAndBytesAdapter(QuantizationAdapter):
    """Adapter for bitsandbytes quantization (CUDA only)."""
    
    def _validate_backend_support(self):
        """Validate that bitsandbytes is available and backend is CUDA."""
        if self.backend != Backend.CUDA:
            raise ValueError("bitsandbytes quantization is only supported on CUDA backend")
        
        if not check_import_availability('bitsandbytes', str(self.backend)):
            raise ImportError("bitsandbytes is not available")
    
    def create_quantized_linear(self, in_features: int, out_features: int, bias: bool = True) -> QuantizedLinear:
        """Create a bitsandbytes quantized linear layer."""
        bnb = get_module('bitsandbytes', str(self.backend))
        
        if self.config.bits == 4:
            return BNBLinear4bit(in_features, out_features, bias, self.config, bnb)
        elif self.config.bits == 8:
            return BNBLinear8bit(in_features, out_features, bias, self.config, bnb)
        else:
            raise ValueError(f"bitsandbytes doesn't support {self.config.bits}-bit quantization")
    
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Quantize a model using bitsandbytes."""
        from train import replace_linear
        
        bnb = get_module('bitsandbytes', str(self.backend))
        Linear4bit = bnb.Linear4bit
        
        # Replace linear layers with quantized versions
        model = replace_linear(
            model,
            Linear4bit,
            compute_dtype=self.config.compute_dtype,
            quant_type=self.config.quant_type,
            use_double_quant=self.config.double_quant,
            skip=self.config.skip_modules
        )
        
        return model
    
    def prepare_model_for_training(self, model: nn.Module) -> nn.Module:
        """Prepare bitsandbytes model for QLoRA training."""
        
        # These functions handle the specific requirements for PEFT with quantized models
        if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
            # Setup for PEFT compatibility
            return model
        
        return model
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save a bitsandbytes quantized model."""
        # bitsandbytes models are typically saved using the standard PyTorch method
        # with special handling for quantization state
        torch.save(model.state_dict(), save_path)
    
    def load_quantized_model(self, model_path: str, model_config: Any) -> nn.Module:
        """Load a bitsandbytes quantized model.
        
        Args:
            model_path: Path to the saved model checkpoint
            model_config: Model configuration (can be AutoConfig or dict)
            
        Returns:
            Loaded quantized model
        """
        import json
        from pathlib import Path
        
        # Get bitsandbytes module
        bnb = get_module('bitsandbytes', str(self.backend))
        
        # Handle different model config types
        if hasattr(model_config, 'to_dict'):
            config_dict = model_config.to_dict()
        else:
            config_dict = model_config if isinstance(model_config, dict) else {}
            
        # Check if model_path is a directory or file
        model_path = Path(model_path)
        if model_path.is_dir():
            # Look for standard checkpoint files
            checkpoint_path = None
            for fname in ['pytorch_model.bin', 'model.safetensors', 'model.pth']:
                if (model_path / fname).exists():
                    checkpoint_path = model_path / fname
                    break
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found in {model_path}")
                
            # Look for quantization config
            quant_config_path = model_path / 'quantization_config.json'
            if quant_config_path.exists():
                with open(quant_config_path) as f:
                    quant_config = json.load(f)
            else:
                # Use default config from self.config
                quant_config = {
                    'bits': self.config.bits,
                    'quant_type': self.config.quant_type,
                    'compute_dtype': str(self.config.compute_dtype) if self.config.compute_dtype else 'float16'
                }
        else:
            checkpoint_path = model_path
            quant_config = {
                'bits': self.config.bits,
                'quant_type': self.config.quant_type,
                'compute_dtype': str(self.config.compute_dtype) if self.config.compute_dtype else 'float16'
            }
            
        # Create model architecture based on config
        # This is a simplified version - in practice, you'd use AutoModel or similar
        try:
            from transformers import AutoModelForCausalLM
            
            # Create model with empty weights first
            from accelerate import init_empty_weights
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(model_config)
                
            # Replace linear layers with quantized versions
            from train import replace_linear
            model = replace_linear(
                model,
                bnb.Linear4bit if quant_config.get('bits', 4) == 4 else bnb.Linear8bitLt,
                compute_dtype=getattr(torch, quant_config.get('compute_dtype', 'float16')),
                quant_type=quant_config.get('quant_type', 'nf4')
            )
            
            # Load the state dict
            if checkpoint_path.suffix == '.safetensors':
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                
            # Handle potential state dict wrapper
            if 'model' in state_dict:
                state_dict = state_dict['model']
                
            # Load weights into the quantized model
            model.load_state_dict(state_dict, strict=False)
            
            return model
            
        except ImportError:
            # Fallback: create a simple quantized model structure
            warnings.warn("Transformers not available, using basic model loading")
            
            # Load state dict
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Create a container for the quantized model
            class QuantizedModel(nn.Module):
                def __init__(self, state_dict, bnb_module, bits=4):
                    super().__init__()
                    
                    # Recreate layers from state dict
                    for name, param in state_dict.items():
                        if 'weight' in name and len(param.shape) == 2:
                            # This is likely a linear layer
                            layer_name = name.replace('.weight', '')
                            in_features, out_features = param.shape
                            
                            if bits == 4:
                                layer = bnb_module.Linear4bit(
                                    in_features, out_features,
                                    compute_dtype=torch.float16,
                                    quant_type='nf4'
                                )
                            else:
                                layer = bnb_module.Linear8bitLt(
                                    in_features, out_features
                                )
                                
                            # Set the layer in the model
                            setattr(self, layer_name.replace('.', '_'), layer)
                            
            model = QuantizedModel(state_dict, bnb, quant_config.get('bits', 4))
            model.load_state_dict(state_dict, strict=False)
            
            return model


class HQQAdapter(QuantizationAdapter):
    """Adapter for HQQ quantization."""
    
    def _validate_backend_support(self):
        """Validate that HQQ is available."""
        if not check_import_availability('hqq', str(self.backend)):
            warnings.warn("HQQ is not available, using fallback")
    
    def create_quantized_linear(self, in_features: int, out_features: int, bias: bool = True) -> QuantizedLinear:
        """Create an HQQ quantized linear layer."""
        try:
            hqq = get_module('hqq', str(self.backend))
            return HQQLinearWrapper(in_features, out_features, bias, self.config, hqq)
        except ImportError:
            # Fallback to standard linear layer
            warnings.warn("HQQ not available, falling back to standard linear layer")
            return nn.Linear(in_features, out_features, bias)
    
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Quantize a model using HQQ."""
        try:
            hqq = get_module('hqq', str(self.backend))
            from train import replace_linear
            
            # Create HQQ quantization config
            quant_config = hqq.BaseQuantizeConfig(
                nbits=self.config.bits,
                group_size=self.config.group_size,
                quant_zero=self.config.quant_zero,
                quant_scale=self.config.quant_scale,
            )
            
            # Replace linear layers
            model = replace_linear(
                model,
                hqq.HQQLinear,
                compute_dtype=self.config.compute_dtype,
                quant_config=quant_config,
                del_orig=True,
                initialize=True,
                skip=self.config.skip_modules
            )
            
            return model
            
        except ImportError:
            warnings.warn("HQQ not available, returning original model")
            return model
    
    def prepare_model_for_training(self, model: nn.Module) -> nn.Module:
        """Prepare HQQ model for training."""
        # HQQ models can be trained directly after quantization
        return model
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save an HQQ quantized model."""
        torch.save(model.state_dict(), save_path)
    
    def load_quantized_model(self, model_path: str, model_config: Any) -> nn.Module:
        """Load an HQQ quantized model.
        
        Args:
            model_path: Path to the saved model checkpoint
            model_config: Model configuration (can be AutoConfig or dict)
            
        Returns:
            Loaded HQQ quantized model
        """
        import json
        from pathlib import Path
        
        try:
            # Get HQQ module
            hqq = get_module('hqq', str(self.backend))
            from hqq.models.base import HQQModel
            from hqq.core.quantize import Quantizer
        except ImportError as e:
            raise ImportError(f"HQQ not available: {e}")
            
        # Handle path
        model_path = Path(model_path)
        
        # Load HQQ-specific configuration
        if model_path.is_dir():
            # Look for HQQ config
            hqq_config_path = model_path / 'hqq_config.json'
            if hqq_config_path.exists():
                with open(hqq_config_path) as f:
                    hqq_config = json.load(f)
            else:
                # Use defaults from self.config
                hqq_config = {
                    'weight_quant_params': {
                        'nbits': self.config.bits,
                        'group_size': self.config.group_size,
                        'quant_zero': self.config.quant_zero,
                        'quant_scale': self.config.quant_scale,
                    },
                    'scale_quant_params': {
                        'nbits': 8,  # Default scale quantization
                        'group_size': 128,
                    },
                    'zero_quant_params': {
                        'nbits': 8,
                        'group_size': 128,
                    } if self.config.quant_zero else None
                }
                
            # Find checkpoint
            checkpoint_path = None
            for fname in ['hqq_model.pt', 'pytorch_model.bin', 'model.safetensors']:
                if (model_path / fname).exists():
                    checkpoint_path = model_path / fname
                    break
                    
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found in {model_path}")
        else:
            checkpoint_path = model_path
            # Use default HQQ config
            hqq_config = {
                'weight_quant_params': {
                    'nbits': self.config.bits,
                    'group_size': self.config.group_size,
                    'quant_zero': self.config.quant_zero,
                    'quant_scale': self.config.quant_scale,
                },
            }
            
        # Try to load as HQQ model first
        try:
            # Load HQQ model directly if saved in HQQ format
            model = HQQModel.load(str(checkpoint_path))
            return model
        except:
            # Fallback: reconstruct from state dict
            pass
            
        # Load state dict and reconstruct
        if checkpoint_path.suffix == '.safetensors':
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
        # Create model architecture
        try:
            from transformers import AutoModelForCausalLM
            from accelerate import init_empty_weights
            
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(model_config)
                
            # Apply HQQ quantization with loaded config
            from train import replace_linear
            
            # Create custom linear replacement function for HQQ
            def create_hqq_linear(in_features, out_features, bias=True):
                linear = nn.Linear(in_features, out_features, bias=bias)
                # Quantize the layer
                quantizer = Quantizer()
                quantizer.configure(
                    bits=hqq_config['weight_quant_params']['nbits'],
                    group_size=hqq_config['weight_quant_params']['group_size'],
                    quant_zero=hqq_config['weight_quant_params'].get('quant_zero', True),
                    quant_scale=hqq_config['weight_quant_params'].get('quant_scale', False)
                )
                return quantizer.quantize_module(linear)
                
            # Replace linear layers
            model = replace_linear(model, create_hqq_linear)
            
            # Load weights
            # HQQ stores quantized weights in a special format
            # Need to handle the mapping correctly
            loaded_keys = []
            for name, module in model.named_modules():
                if hasattr(module, 'W_q'):  # HQQ quantized layer
                    # Map quantized weight keys
                    if f"{name}.W_q" in state_dict:
                        module.W_q = state_dict[f"{name}.W_q"]
                        loaded_keys.append(f"{name}.W_q")
                    if f"{name}.meta" in state_dict:
                        module.meta = state_dict[f"{name}.meta"]
                        loaded_keys.append(f"{name}.meta")
                    if f"{name}.bias" in state_dict and hasattr(module, 'bias'):
                        module.bias = state_dict[f"{name}.bias"]
                        loaded_keys.append(f"{name}.bias")
                        
            # Load any remaining non-quantized parameters
            remaining_state_dict = {k: v for k, v in state_dict.items() if k not in loaded_keys}
            if remaining_state_dict:
                model.load_state_dict(remaining_state_dict, strict=False)
                
            return model
            
        except ImportError:
            warnings.warn("Transformers not available, using basic HQQ model loading")
            
            # Basic fallback implementation
            class HQQModel(nn.Module):
                def __init__(self, state_dict, hqq_config):
                    super().__init__()
                    self.hqq_config = hqq_config
                    
                    # Reconstruct layers from state dict
                    for name, param in state_dict.items():
                        if name.endswith('.W_q'):
                            # This is an HQQ quantized weight
                            layer_name = name.replace('.W_q', '')
                            setattr(self, layer_name.replace('.', '_'), param)
                            
            model = HQQModel(state_dict, hqq_config)
            return model


class MLXAdapter(QuantizationAdapter):
    """Adapter for MLX quantization (Apple Silicon only)."""
    
    def _validate_backend_support(self):
        """Validate that MLX is available and backend is appropriate."""
        if self.backend not in [Backend.MLX, Backend.MPS]:
            raise ValueError("MLX quantization is only supported on Apple Silicon (MLX/MPS backends)")
        
        if not check_import_availability('mlx', str(self.backend)):
            raise ImportError("MLX is not available")
    
    def create_quantized_linear(self, in_features: int, out_features: int, bias: bool = True) -> QuantizedLinear:
        """Create an MLX quantized linear layer."""
        # MLX quantization works differently - it's applied to the entire model
        # This returns a wrapper that will be handled by quantize_model
        return MLXLinearWrapper(in_features, out_features, bias, self.config)
    
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Quantize a model using MLX."""
        try:
            import mlx.core as mx
            import mlx.nn as nn_mlx
            from mlx.utils import tree_unflatten
            
            # Convert PyTorch model to MLX format first
            # This is a simplified version - actual implementation would need proper conversion
            warnings.warn("MLX quantization requires model conversion from PyTorch to MLX format")
            
            # Apply quantization using MLX's built-in methods
            # MLX supports per-layer bit configuration
            layer_bits = {}
            for name in self.config.layer_configs:
                layer_bits[name] = self.config.layer_configs[name].get('bits', self.config.bits)
            
            # Return a wrapped model that can interface with PyTorch
            return MLXModelWrapper(model, self.config)
            
        except ImportError:
            warnings.warn("MLX not available, returning original model")
            return model
    
    def prepare_model_for_training(self, model: nn.Module) -> nn.Module:
        """Prepare MLX model for training."""
        # MLX supports training on quantized models natively
        return model
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save an MLX quantized model."""
        # MLX models have their own serialization format
        if hasattr(model, 'save_mlx'):
            model.save_mlx(save_path)
        else:
            warnings.warn("Model doesn't support MLX save format, using PyTorch save")
            torch.save(model.state_dict(), save_path)
    
    def load_quantized_model(self, model_path: str, model_config: Any) -> nn.Module:
        """Load an MLX quantized model.
        
        Args:
            model_path: Path to the saved model checkpoint
            model_config: Model configuration (can be AutoConfig or dict)
            
        Returns:
            Loaded MLX quantized model wrapped for PyTorch compatibility
        """
        import json
        from pathlib import Path
        
        # Validate MLX availability
        try:
            import mlx.core as mx
        except ImportError as e:
            raise ImportError(f"MLX not available: {e}")
            
        # Also check for MLX quantization module
        try:
            from src.backends.mlx.mlx_quantization import (
                load_quantized_model as mlx_load_model
            )
        except ImportError:
            warnings.warn("MLX quantization module not available, using basic loading")
            mlx_load_model = None
            
        model_path = Path(model_path)
        
        # Load MLX-specific configuration
        if model_path.is_dir():
            # Look for MLX config
            mlx_config_path = model_path / 'mlx_config.json'
            if mlx_config_path.exists():
                with open(mlx_config_path) as f:
                    mlx_config = json.load(f)
            else:
                # Use defaults
                mlx_config = {
                    'bits': self.config.bits,
                    'group_size': self.config.group_size,
                }
                
            # Find MLX checkpoint
            checkpoint_path = None
            for fname in ['model.safetensors', 'mlx_model.npz', 'weights.npz']:
                if (model_path / fname).exists():
                    checkpoint_path = model_path / fname
                    break
                    
            if checkpoint_path is None:
                # Check for PyTorch checkpoint to convert
                for fname in ['pytorch_model.bin', 'model.pth']:
                    if (model_path / fname).exists():
                        checkpoint_path = model_path / fname
                        break
                        
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found in {model_path}")
        else:
            checkpoint_path = model_path
            mlx_config = {
                'bits': self.config.bits,
                'group_size': self.config.group_size,
            }
            
        # Try to use the MLX quantization module's loader if available
        if mlx_load_model is not None and checkpoint_path.suffix in ['.npz', '.safetensors']:
            try:
                # Load using MLX quantization module
                mlx_model = mlx_load_model(
                    str(checkpoint_path),
                    bits=mlx_config['bits'],
                    group_size=mlx_config['group_size']
                )
                
                # Wrap in PyTorch compatibility layer
                return MLXModelWrapper(mlx_model, self.config)
                
            except Exception as e:
                warnings.warn(f"Failed to load with MLX module: {e}, trying fallback")
                
        # Fallback: Load and convert
        if checkpoint_path.suffix in ['.bin', '.pth']:
            # Load PyTorch checkpoint and convert to MLX
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Create model from config
            try:
                from transformers import AutoModelForCausalLM
                from accelerate import init_empty_weights
                
                # Create PyTorch model structure
                with init_empty_weights():
                    pytorch_model = AutoModelForCausalLM.from_config(model_config)
                    
                # Convert to MLX format
                mlx_weights = {}
                for name, param in state_dict.items():
                    # Convert torch tensor to numpy then to MLX
                    if isinstance(param, torch.Tensor):
                        np_array = param.detach().cpu().numpy()
                        mlx_weights[name] = mx.array(np_array)
                    else:
                        mlx_weights[name] = param
                        
                # Create a simple MLX model wrapper
                class MLXModel:
                    def __init__(self, weights, config):
                        self.weights = weights
                        self.config = config
                        self.quantized = True
                        
                    def __call__(self, x):
                        # This is a placeholder - actual implementation would
                        # need the full model architecture in MLX
                        warnings.warn("MLX model inference not fully implemented")
                        return x
                        
                mlx_model = MLXModel(mlx_weights, mlx_config)
                
                # Wrap for PyTorch compatibility
                wrapped_model = MLXModelWrapper(pytorch_model, self.config)
                wrapped_model.mlx_model = mlx_model
                
                return wrapped_model
                
            except ImportError:
                warnings.warn("Transformers not available, returning basic MLX model")
                
        elif checkpoint_path.suffix == '.npz':
            # Load native MLX format
            weights = mx.load(str(checkpoint_path))
            
            # Create a basic MLX model
            class MLXModel:
                def __init__(self, weights):
                    self.weights = weights
                    
            mlx_model = MLXModel(weights)
            
            # Need to create a PyTorch wrapper
            # This is simplified - real implementation would need proper architecture
            dummy_pytorch_model = nn.Module()
            wrapped_model = MLXModelWrapper(dummy_pytorch_model, self.config)
            wrapped_model.mlx_model = mlx_model
            
            return wrapped_model
            
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path.suffix}")


class FallbackAdapter(QuantizationAdapter):
    """Fallback adapter when no quantization backend is available."""
    
    def _validate_backend_support(self):
        """No validation needed for fallback."""
        warnings.warn(f"No quantization support for {self.backend}, using fallback (no quantization)")
    
    def create_quantized_linear(self, in_features: int, out_features: int, bias: bool = True) -> nn.Module:
        """Create a standard linear layer as fallback."""
        return nn.Linear(in_features, out_features, bias)
    
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Return model unchanged."""
        warnings.warn("No quantization applied (fallback mode)")
        return model
    
    def prepare_model_for_training(self, model: nn.Module) -> nn.Module:
        """Return model unchanged."""
        return model
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save model using standard PyTorch method."""
        torch.save(model.state_dict(), save_path)
    
    def load_quantized_model(self, model_path: str, model_config: Any) -> nn.Module:
        """Load model using standard PyTorch method."""
        raise NotImplementedError("Standard model loading should be used")


class QuantoAdapter(QuantizationAdapter):
    """Adapter for Quanto quantization (Hugging Face)."""
    
    def _validate_backend_support(self):
        """Validate that Quanto is available."""
        if not check_import_availability('optimum.quanto', str(self.backend)):
            warnings.warn("Quanto is not available. Install with: pip install optimum-quanto")
    
    def create_quantized_linear(self, in_features: int, out_features: int, bias: bool = True) -> QuantizedLinear:
        """Create a Quanto quantized linear layer."""
        try:
            from optimum.quanto import QLinear, qint2, qint4, qint8
            
            # Select quantization type based on bits
            if self.config.bits == 2:
                qtype = qint2
            elif self.config.bits == 4:
                qtype = qint4
            elif self.config.bits == 8:
                qtype = qint8
            else:
                warnings.warn(f"Quanto doesn't support {self.config.bits}-bit quantization, using 8-bit")
                qtype = qint8
                
            # Create quantized layer
            layer = QLinear(in_features, out_features, bias=bias, weights=qtype)
            return layer
            
        except ImportError:
            warnings.warn("Quanto not available, falling back to standard linear layer")
            return nn.Linear(in_features, out_features, bias)
    
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Quantize a model using Quanto."""
        try:
            from optimum.quanto import quantize, freeze, qint2, qint4, qint8
            
            # Select quantization type
            if self.config.bits == 2:
                weights_qtype = qint2
            elif self.config.bits == 4:
                weights_qtype = qint4
            elif self.config.bits == 8:
                weights_qtype = qint8
            else:
                weights_qtype = qint8
                
            # Apply quantization
            quantize(model, weights=weights_qtype)
            
            # Freeze to make model inference-only (optional)
            if kwargs.get('freeze', True):
                freeze(model)
                
            return model
            
        except ImportError:
            warnings.warn("Quanto not available, returning original model")
            return model
    
    def prepare_model_for_training(self, model: nn.Module) -> nn.Module:
        """Prepare Quanto model for training."""
        # Quanto models can be trained after quantization
        # Just ensure gradients are enabled for trainable parameters
        for param in model.parameters():
            param.requires_grad = True
        return model
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save a Quanto quantized model."""
        import json
        from pathlib import Path
        
        save_path = Path(save_path)
        
        # Create directory if saving to directory
        if not save_path.suffix:
            save_path.mkdir(parents=True, exist_ok=True)
            
        # Save the state dict
        if save_path.is_dir():
            # Save model weights
            torch.save(model.state_dict(), save_path / 'pytorch_model.bin')
            
            # Save Quanto config
            quanto_config = {
                'quantization_config': {
                    'quant_method': 'quanto',
                    'bits': self.config.bits,
                    'group_size': self.config.group_size,
                }
            }
            with open(save_path / 'config.json', 'w') as f:
                json.dump(quanto_config, f, indent=2)
        else:
            # Save as single file
            torch.save({
                'state_dict': model.state_dict(),
                'quanto_config': {
                    'bits': self.config.bits,
                    'group_size': self.config.group_size,
                }
            }, save_path)
    
    def load_quantized_model(self, model_path: str, model_config: Any) -> nn.Module:
        """Load a Quanto quantized model.
        
        Args:
            model_path: Path to the saved model checkpoint
            model_config: Model configuration (can be AutoConfig or dict)
            
        Returns:
            Loaded Quanto quantized model
        """
        import json
        from pathlib import Path
        
        try:
            from optimum.quanto import quantize, freeze, qint2, qint4, qint8
        except ImportError:
            raise ImportError("Quanto is required for loading Quanto models. Install with: pip install optimum-quanto")
            
        model_path = Path(model_path)
        
        # Load checkpoint and config
        if model_path.is_dir():
            # Look for model file
            checkpoint_path = None
            for fname in ['pytorch_model.bin', 'model.safetensors', 'model.pth']:
                if (model_path / fname).exists():
                    checkpoint_path = model_path / fname
                    break
                    
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found in {model_path}")
                
            # Load Quanto config if available
            config_path = model_path / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    saved_config = json.load(f)
                    quanto_config = saved_config.get('quantization_config', {})
            else:
                quanto_config = {'bits': self.config.bits}
                
        else:
            checkpoint_path = model_path
            
            # Try to load as a single file with embedded config
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'quanto_config' in checkpoint:
                quanto_config = checkpoint['quanto_config']
                state_dict = checkpoint.get('state_dict', checkpoint)
            else:
                state_dict = checkpoint
                quanto_config = {'bits': self.config.bits}
                
        # Create model architecture
        try:
            from transformers import AutoModelForCausalLM
            from accelerate import init_empty_weights
            
            # Create model with empty weights
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(model_config)
                
            # Apply Quanto quantization to the model structure
            bits = quanto_config.get('bits', 8)
            if bits == 2:
                weights_qtype = qint2
            elif bits == 4:
                weights_qtype = qint4
            else:
                weights_qtype = qint8
                
            # Quantize the model architecture
            quantize(model, weights=weights_qtype)
            
            # Load the quantized weights
            if checkpoint_path.suffix == '.safetensors':
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            else:
                if 'state_dict' not in locals():
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    
            # Handle state dict wrappers
            if 'model' in state_dict and isinstance(state_dict['model'], dict):
                state_dict = state_dict['model']
                
            # Load the weights
            model.load_state_dict(state_dict, strict=False)
            
            # Freeze the model to finalize quantization
            freeze(model)
            
            return model
            
        except ImportError:
            warnings.warn("Transformers not available, using basic Quanto model loading")
            
            # Basic fallback
            class QuantoModel(nn.Module):
                def __init__(self, state_dict, quanto_config):
                    super().__init__()
                    self.quanto_config = quanto_config
                    
                    # Recreate structure from state dict
                    # This is simplified - real implementation would need proper architecture
                    for name, param in state_dict.items():
                        if 'weight' in name:
                            setattr(self, name.replace('.', '_'), nn.Parameter(param))
                            
            model = QuantoModel(state_dict, quanto_config)
            return model


# Concrete implementations of quantized linear layers

class BNBLinear4bit(QuantizedLinear):
    """bitsandbytes 4-bit quantized linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool, config: QuantizationConfig, bnb_module):
        super().__init__(in_features, out_features, bias, config)
        self.bnb = bnb_module
        
        # Create the actual bitsandbytes layer
        self.layer = self.bnb.Linear4bit(
            in_features,
            out_features,
            bias=bias,
            compute_dtype=config.compute_dtype,
            compress_statistics=config.double_quant,
            quant_type=config.quant_type
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bitsandbytes 4-bit layer."""
        return self.layer(x)
    
    def quantize_weights(self, weights: torch.Tensor) -> Any:
        """Quantize weights using bitsandbytes."""
        # bitsandbytes handles quantization internally
        self.layer.weight = self.bnb.Params4bit(
            weights.to(self.config.compute_dtype),
            requires_grad=False,
            compress_statistics=self.config.double_quant,
            quant_type=self.config.quant_type
        )
        return self.layer.weight
    
    def dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights back to full precision."""
        # bitsandbytes provides dequantization
        return self.layer.weight.dequantize()


class BNBLinear8bit(QuantizedLinear):
    """bitsandbytes 8-bit quantized linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool, config: QuantizationConfig, bnb_module):
        super().__init__(in_features, out_features, bias, config)
        self.bnb = bnb_module
        
        # Create the actual bitsandbytes layer
        self.layer = self.bnb.Linear8bitLt(
            in_features,
            out_features,
            bias=bias,
            has_fp16_weights=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through bitsandbytes 8-bit layer."""
        return self.layer(x)
    
    def quantize_weights(self, weights: torch.Tensor) -> Any:
        """Quantize weights using bitsandbytes 8-bit."""
        self.layer.weight = nn.Parameter(weights)
        # Trigger quantization
        self.layer.cuda()
        return self.layer.weight
    
    def dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights back to full precision."""
        return self.layer.weight.float()


class HQQLinearWrapper(QuantizedLinear):
    """HQQ quantized linear layer wrapper."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool, config: QuantizationConfig, hqq_module):
        super().__init__(in_features, out_features, bias, config)
        self.hqq = hqq_module
        
        # Create HQQ configuration
        quant_config = self.hqq.BaseQuantizeConfig(
            nbits=config.bits,
            group_size=config.group_size,
            quant_zero=config.quant_zero,
            quant_scale=config.quant_scale,
        )
        
        # Create HQQ linear layer
        self.layer = self.hqq.HQQLinear(
            None,  # Will be set during quantization
            quant_config,
            compute_dtype=config.compute_dtype,
            device='cuda' if config.compute_dtype else 'cpu'
        )
        
        # Initialize with dummy linear layer
        self.linear_layer = nn.Linear(in_features, out_features, bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through HQQ layer."""
        # Use linear_layer for forward pass
        # HQQ applies quantization internally
        return self.linear_layer(x)
    
    def quantize_weights(self, weights: torch.Tensor) -> Any:
        """Quantize weights using HQQ."""
        self.linear_layer.weight.data = weights
        # HQQ quantization happens in-place
        self.layer.linear_layer = self.linear_layer
        self.layer.initialize()
        return self.layer
    
    def dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights back to full precision."""
        if hasattr(self.layer, 'dequantize'):
            return self.layer.dequantize()
        return self.linear_layer.weight.data


class MLXLinearWrapper(QuantizedLinear):
    """MLX quantized linear layer wrapper."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool, config: QuantizationConfig):
        super().__init__(in_features, out_features, bias, config)
        
        # MLX quantization is handled differently
        # This is a placeholder that will be replaced during model conversion
        self.layer = nn.Linear(in_features, out_features, bias)
        self.quantized = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through layer."""
        return self.layer(x)
    
    def quantize_weights(self, weights: torch.Tensor) -> Any:
        """Mark for MLX quantization."""
        self.layer.weight.data = weights
        self.quantized = True
        return self.layer.weight
    
    def dequantize_weights(self) -> torch.Tensor:
        """Return weights (MLX handles dequantization internally)."""
        return self.layer.weight.data


class MLXModelWrapper(nn.Module):
    """Wrapper for MLX quantized models to interface with PyTorch."""
    
    def __init__(self, pytorch_model: nn.Module, config: QuantizationConfig):
        super().__init__()
        self.pytorch_model = pytorch_model
        self.config = config
        self.mlx_model = None  # Will be set during conversion
        
    def forward(self, *args, **kwargs):
        """Forward pass through MLX model."""
        if self.mlx_model is not None:
            # Convert inputs to MLX format, run inference, convert back
            warnings.warn("MLX model inference requires format conversion")
            return self.pytorch_model(*args, **kwargs)
        return self.pytorch_model(*args, **kwargs)
    
    def save_mlx(self, path: str):
        """Save in MLX format."""
        if self.mlx_model is not None:
            # Save using MLX utilities
            pass
        else:
            warnings.warn("No MLX model to save")


# Factory function to create appropriate adapter

def create_quantization_adapter(
    backend: Union[str, Backend],
    config: Optional[QuantizationConfig] = None
) -> QuantizationAdapter:
    """
    Create a quantization adapter for the specified backend.
    
    Args:
        backend: Backend to use (cuda, mps, mlx, cpu, or Backend enum)
        config: Quantization configuration
        
    Returns:
        QuantizationAdapter instance
    """
    if isinstance(backend, str):
        backend = Backend(backend)
    
    if config is None:
        config = QuantizationConfig()
    
    # Select adapter based on backend and quantization method
    if backend == Backend.CUDA and config.method in [QuantizationMethod.BNB_NF4, QuantizationMethod.BNB_INT8]:
        return BitsAndBytesAdapter(backend, config)
    elif config.method == QuantizationMethod.HQQ:
        return HQQAdapter(backend, config)
    elif backend in [Backend.MLX, Backend.MPS] and config.method in [QuantizationMethod.MLX_INT4, QuantizationMethod.MLX_INT8]:
        return MLXAdapter(backend, config)
    elif config.method in [QuantizationMethod.QUANTO_INT2, QuantizationMethod.QUANTO_INT4, QuantizationMethod.QUANTO_INT8]:
        return QuantoAdapter(backend, config)
    else:
        # Fallback for unsupported combinations
        return FallbackAdapter(backend, config)


# Utility functions for configuration validation

def validate_quantization_config(config: QuantizationConfig, backend: Backend) -> List[str]:
    """
    Validate a quantization configuration for a specific backend.
    
    Returns:
        List of validation warnings/errors (empty if valid)
    """
    issues = []
    
    # Check method compatibility
    if config.method in [QuantizationMethod.BNB_NF4, QuantizationMethod.BNB_INT8] and backend != Backend.CUDA:
        issues.append(f"bitsandbytes quantization is only supported on CUDA, not {backend}")
    
    if config.method in [QuantizationMethod.MLX_INT4, QuantizationMethod.MLX_INT8] and backend not in [Backend.MLX, Backend.MPS]:
        issues.append(f"MLX quantization is only supported on Apple Silicon, not {backend}")
    
    # Check bit width support
    if backend == Backend.MPS and config.bits == 4 and config.method not in [QuantizationMethod.MLX_INT4]:
        issues.append("MPS backend has limited 4-bit quantization support")
    
    # Check compute dtype
    if backend == Backend.MPS and config.compute_dtype == torch.bfloat16:
        issues.append("MPS backend doesn't support bfloat16")
    
    return issues


def get_recommended_config(
    backend: Backend,
    model_size_b: float = 7.0,
    available_memory_gb: float = 16.0
) -> QuantizationConfig:
    """
    Get recommended quantization configuration based on backend and constraints.
    
    Args:
        backend: Target backend
        model_size_b: Model size in billions of parameters
        available_memory_gb: Available memory in GB
        
    Returns:
        Recommended QuantizationConfig
    """
    # Estimate memory requirements
    fp16_memory_gb = model_size_b * 2  # 2 bytes per parameter
    int8_memory_gb = model_size_b * 1   # 1 byte per parameter
    # int4_memory_gb = model_size_b * 0.5 # 0.5 bytes per parameter
    
    # Select configuration based on backend and memory
    if backend == Backend.CUDA:
        if available_memory_gb < fp16_memory_gb * 1.2:  # Need quantization
            if available_memory_gb < int8_memory_gb * 1.5:  # Very limited memory
                return QuantizationConfig(
                    method=QuantizationMethod.BNB_NF4,
                    bits=4,
                    compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:  # Moderate memory
                return QuantizationConfig(
                    method=QuantizationMethod.BNB_INT8,
                    bits=8,
                    compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
        else:
            return QuantizationConfig(method=QuantizationMethod.NONE)
    
    elif backend in [Backend.MLX, Backend.MPS]:
        if available_memory_gb < fp16_memory_gb * 1.2:  # Need quantization
            if available_memory_gb < int8_memory_gb * 1.5:  # Very limited memory
                return QuantizationConfig(
                    method=QuantizationMethod.MLX_INT4,
                    bits=4,
                    compute_dtype=torch.float16  # MPS doesn't support bfloat16
                )
            else:  # Moderate memory
                return QuantizationConfig(
                    method=QuantizationMethod.MLX_INT8,
                    bits=8,
                    compute_dtype=torch.float16
                )
        else:
            return QuantizationConfig(method=QuantizationMethod.NONE)
    
    else:
        # CPU or other backends - use HQQ if available
        if available_memory_gb < int8_memory_gb * 2:  # Higher overhead on CPU
            return QuantizationConfig(
                method=QuantizationMethod.HQQ,
                bits=8,
                compute_dtype=torch.float32
            )
        else:
            return QuantizationConfig(method=QuantizationMethod.NONE)


# Integration helpers for train.py

def replace_linear_with_quantized(
    model: nn.Module,
    adapter: QuantizationAdapter,
    skip_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Replace linear layers in a model with quantized versions.
    
    Args:
        model: Model to quantize
        adapter: Quantization adapter to use
        skip_modules: List of module names to skip
        
    Returns:
        Model with quantized linear layers
    """
    skip_modules = skip_modules or adapter.config.skip_modules
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if we should skip this module
            if any(skip in name for skip in skip_modules):
                continue
            
            # Get parent module and attribute name
            *parent_names, attr_name = name.split('.')
            parent = model
            for parent_name in parent_names:
                parent = getattr(parent, parent_name)
            
            # Create quantized replacement
            quantized_layer = adapter.create_quantized_linear(
                module.in_features,
                module.out_features,
                module.bias is not None
            )
            
            # Copy weights if available
            if hasattr(module, 'weight') and module.weight is not None:
                if hasattr(quantized_layer, 'quantize_weights'):
                    quantized_layer.quantize_weights(module.weight.data)
                else:
                    # Fallback: just copy weights
                    if hasattr(quantized_layer, 'weight'):
                        quantized_layer.weight.data = module.weight.data
            
            # Replace the module
            setattr(parent, attr_name, quantized_layer)
    
    return model


__all__ = [
    'QuantizationMethod',
    'QuantizationConfig',
    'QuantizedLinear',
    'QuantizationAdapter',
    'BitsAndBytesAdapter',
    'HQQAdapter',
    'MLXAdapter',
    'FallbackAdapter',
    'create_quantization_adapter',
    'validate_quantization_config',
    'get_recommended_config',
    'replace_linear_with_quantized'
]