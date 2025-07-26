"""
PyTorch-MLX Bridge

Provides conversion utilities between PyTorch and MLX models,
enabling seamless migration and interoperability.
"""

import os
import json
import warnings
import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn_mlx
    from mlx.utils import tree_map
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    warnings.warn("MLX not available. Install with: pip install mlx")

from transformers import AutoTokenizer, AutoConfig

logger = logging.getLogger(__name__)


class TensorConverter:
    """Convert between PyTorch and MLX tensors."""
    
    @staticmethod
    def torch_to_mlx(tensor: torch.Tensor) -> '"mx.array"':
        """Convert PyTorch tensor to MLX array."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX not available")
        
        # Convert to numpy first (handles device transfer)
        numpy_array = tensor.detach().cpu().numpy()
        
        # Convert to MLX
        mlx_array = mx.array(numpy_array)
        
        return mlx_array
    
    @staticmethod
    def mlx_to_torch(
        array: '"mx.array"',
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Convert MLX array to PyTorch tensor."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX not available")
        
        # Convert to numpy
        numpy_array = np.array(array)
        
        # Convert to PyTorch
        tensor = torch.from_numpy(numpy_array)
        
        # Set dtype if specified
        if dtype is not None:
            tensor = tensor.to(dtype)
        
        # Move to device if specified
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor
    
    @staticmethod
    def convert_state_dict(
        state_dict: Dict[str, Any],
        to_framework: str = "mlx",
    ) -> Dict[str, Any]:
        """Convert entire state dict between frameworks."""
        converted = {}
        
        for key, value in state_dict.items():
            if to_framework == "mlx":
                if isinstance(value, torch.Tensor):
                    converted[key] = TensorConverter.torch_to_mlx(value)
                else:
                    converted[key] = value
            elif to_framework == "torch":
                if MLX_AVAILABLE and isinstance(value, mx.array):
                    converted[key] = TensorConverter.mlx_to_torch(value)
                else:
                    converted[key] = value
            else:
                raise ValueError(f"Unknown framework: {to_framework}")
        
        return converted


class LayerConverter:
    """Convert between PyTorch and MLX layers."""
    
    @staticmethod
    def convert_linear(
        torch_linear: nn.Linear,
        to_mlx: bool = True,
    ) -> Union['nn_mlx.Linear', nn.Linear]:
        """Convert Linear layer between frameworks."""
        if to_mlx:
            if not MLX_AVAILABLE:
                raise ImportError("MLX not available")
            
            # Create MLX linear layer
            mlx_linear = nn_mlx.Linear(
                torch_linear.in_features,
                torch_linear.out_features,
                bias=torch_linear.bias is not None,
            )
            
            # Copy weights
            mlx_linear.weight = TensorConverter.torch_to_mlx(
                torch_linear.weight.T  # MLX uses transposed weight
            )
            
            if torch_linear.bias is not None:
                mlx_linear.bias = TensorConverter.torch_to_mlx(torch_linear.bias)
            
            return mlx_linear
        else:
            # Convert from MLX to PyTorch
            if not hasattr(to_mlx, 'weight'):
                raise ValueError("Invalid MLX linear layer")
            
            # Create PyTorch linear layer
            in_features = to_mlx.weight.shape[1]
            out_features = to_mlx.weight.shape[0]
            
            torch_linear = nn.Linear(
                in_features,
                out_features,
                bias=hasattr(to_mlx, 'bias') and to_mlx.bias is not None,
            )
            
            # Copy weights (transpose back)
            torch_linear.weight.data = TensorConverter.mlx_to_torch(
                to_mlx.weight.T
            )
            
            if hasattr(to_mlx, 'bias') and to_mlx.bias is not None:
                torch_linear.bias.data = TensorConverter.mlx_to_torch(to_mlx.bias)
            
            return torch_linear
    
    @staticmethod
    def convert_embedding(
        torch_embedding: nn.Embedding,
        to_mlx: bool = True,
    ) -> Union['nn_mlx.Embedding', nn.Embedding]:
        """Convert Embedding layer between frameworks."""
        if to_mlx:
            if not MLX_AVAILABLE:
                raise ImportError("MLX not available")
            
            # Create MLX embedding
            mlx_embedding = nn_mlx.Embedding(
                torch_embedding.num_embeddings,
                torch_embedding.embedding_dim,
            )
            
            # Copy weights
            mlx_embedding.weight = TensorConverter.torch_to_mlx(
                torch_embedding.weight
            )
            
            return mlx_embedding
        else:
            # Convert from MLX to PyTorch
            torch_embedding = nn.Embedding(
                to_mlx.num_embeddings,
                to_mlx.embedding_dim,
            )
            
            torch_embedding.weight.data = TensorConverter.mlx_to_torch(
                to_mlx.weight
            )
            
            return torch_embedding
    
    @staticmethod
    def convert_layernorm(
        torch_ln: nn.LayerNorm,
        to_mlx: bool = True,
    ) -> Union['nn_mlx.LayerNorm', nn.LayerNorm]:
        """Convert LayerNorm between frameworks."""
        if to_mlx:
            if not MLX_AVAILABLE:
                raise ImportError("MLX not available")
            
            # Create MLX LayerNorm
            mlx_ln = nn_mlx.LayerNorm(
                torch_ln.normalized_shape[0],
                eps=torch_ln.eps,
                affine=torch_ln.elementwise_affine,
            )
            
            # Copy parameters
            if torch_ln.elementwise_affine:
                mlx_ln.weight = TensorConverter.torch_to_mlx(torch_ln.weight)
                mlx_ln.bias = TensorConverter.torch_to_mlx(torch_ln.bias)
            
            return mlx_ln
        else:
            # Convert from MLX to PyTorch
            torch_ln = nn.LayerNorm(
                to_mlx.dims,
                eps=to_mlx.eps,
                elementwise_affine=to_mlx.affine,
            )
            
            if to_mlx.affine:
                torch_ln.weight.data = TensorConverter.mlx_to_torch(to_mlx.weight)
                torch_ln.bias.data = TensorConverter.mlx_to_torch(to_mlx.bias)
            
            return torch_ln


class ModelConverter:
    """Convert entire models between PyTorch and MLX."""
    
    def __init__(self):
        self.layer_converters = {
            nn.Linear: LayerConverter.convert_linear,
            nn.Embedding: LayerConverter.convert_embedding,
            nn.LayerNorm: LayerConverter.convert_layernorm,
        }
        
        if MLX_AVAILABLE:
            self.mlx_layer_types = {
                nn_mlx.Linear: LayerConverter.convert_linear,
                nn_mlx.Embedding: LayerConverter.convert_embedding,
                nn_mlx.LayerNorm: LayerConverter.convert_layernorm,
            }
    
    def convert_module(
        self,
        module: nn.Module,
        to_mlx: bool = True,
        module_map: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Convert a single module between frameworks."""
        module_type = type(module)
        
        # Check if we have a converter for this type
        if to_mlx and module_type in self.layer_converters:
            converter = self.layer_converters[module_type]
            return converter(module, to_mlx=True)
        elif not to_mlx and MLX_AVAILABLE and module_type in self.mlx_layer_types:
            converter = self.mlx_layer_types[module_type]
            return converter(module, to_mlx=False)
        else:
            # For unsupported layers, try to convert recursively
            return self._convert_container(module, to_mlx, module_map)
    
    def _convert_container(
        self,
        module: nn.Module,
        to_mlx: bool,
        module_map: Optional[Dict[str, Any]],
    ) -> Any:
        """Convert container modules (Sequential, ModuleList, etc.)."""
        if isinstance(module, nn.Sequential):
            if to_mlx and MLX_AVAILABLE:
                # Convert to MLX Sequential equivalent
                layers = []
                for layer in module:
                    converted = self.convert_module(layer, to_mlx, module_map)
                    layers.append(converted)
                return nn_mlx.Sequential(*layers)
            else:
                # Convert from MLX to PyTorch Sequential
                layers = []
                for layer in module:
                    converted = self.convert_module(layer, to_mlx, module_map)
                    layers.append(converted)
                return nn.Sequential(*layers)
        
        elif isinstance(module, nn.ModuleList):
            # Convert ModuleList
            converted_list = []
            for sub_module in module:
                converted = self.convert_module(sub_module, to_mlx, module_map)
                converted_list.append(converted)
            
            if to_mlx:
                # MLX doesn't have ModuleList, use list
                return converted_list
            else:
                return nn.ModuleList(converted_list)
        
        else:
            # For custom modules, create a new instance with converted sub-modules
            if module_map and module.__class__.__name__ in module_map:
                # Use custom conversion logic
                return module_map[module.__class__.__name__](module, to_mlx)
            else:
                warnings.warn(
                    f"No converter for {module.__class__.__name__}, "
                    "returning as-is"
                )
                return module
    
    def convert_model(
        self,
        model: Union[nn.Module, Any],
        to_mlx: bool = True,
        module_map: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Convert entire model between frameworks."""
        if to_mlx:
            if not isinstance(model, nn.Module):
                raise ValueError("Input must be a PyTorch model")
            
            # Convert each module
            converted_modules = {}
            
            for name, module in model.named_children():
                converted = self.convert_module(module, to_mlx, module_map)
                converted_modules[name] = converted
            
            # Create MLX model structure
            # This is simplified - real implementation would need model-specific logic
            if MLX_AVAILABLE:
                class MLXModel(nn_mlx.Module):
                    def __init__(self, modules):
                        super().__init__()
                        for name, module in modules.items():
                            setattr(self, name, module)
                    
                    def __call__(self, *args, **kwargs):
                        # Forward pass through converted modules
                        x = args[0] if args else kwargs.get('input_ids')
                        
                        # Simple forward pass - real implementation would be model-specific
                        for name, module in self.__dict__.items():
                            if isinstance(module, nn_mlx.Module) and name != '__class__':
                                if hasattr(module, '__call__'):
                                    x = module(x)
                        
                        return x
                
                return MLXModel(converted_modules)
            else:
                raise ImportError("MLX not available")
        else:
            # Convert from MLX to PyTorch
            if not MLX_AVAILABLE:
                raise ImportError("MLX not available")
                
            # Create PyTorch model structure
            converted_modules = {}
            
            # Get all modules from MLX model
            for name in dir(model):
                attr = getattr(model, name)
                if isinstance(attr, nn_mlx.Module):
                    # Convert MLX module to PyTorch
                    converted = self.convert_module(attr, to_mlx=False, module_map=module_map)
                    converted_modules[name] = converted
            
            # Create PyTorch model
            class PyTorchModel(nn.Module):
                def __init__(self, modules):
                    super().__init__()
                    for name, module in modules.items():
                        setattr(self, name, module)
                        
                def forward(self, *args, **kwargs):
                    # This would need model-specific implementation
                    x = args[0] if args else kwargs.get('input_ids')
                    
                    # Process through layers (simplified)
                    if hasattr(self, 'embeddings'):
                        x = self.embeddings(x)
                    
                    if hasattr(self, 'layers'):
                        for layer in self.layers:
                            x = layer(x)
                    
                    if hasattr(self, 'lm_head'):
                        x = self.lm_head(x)
                        
                    return x
            
            return PyTorchModel(converted_modules)
    
    def save_converted_model(
        self,
        model: Any,
        save_path: str,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """Save converted model with metadata."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights
        if MLX_AVAILABLE and isinstance(model, nn_mlx.Module):
            # Save MLX model
            weights = tree_map(lambda x: x, model.parameters())
            mx.save(os.path.join(save_path, "model.npz"), weights)
        else:
            # Save PyTorch model
            torch.save(
                model.state_dict(),
                os.path.join(save_path, "model.pt")
            )
        
        # Save metadata
        metadata = {
            "framework": "mlx" if MLX_AVAILABLE and isinstance(model, nn_mlx.Module) else "pytorch",
            "conversion_info": {
                "source": "pytorch-mlx-bridge",
                "version": "1.0.0",
            },
        }
        
        if model_config:
            metadata["model_config"] = model_config
        
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)


def convert_huggingface_to_mlx(
    model_id: str,
    save_path: str,
    quantize: bool = True,
    quantization_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """
    Convert HuggingFace model to MLX format.
    
    Args:
        model_id: HuggingFace model ID
        save_path: Path to save converted model
        quantize: Whether to quantize the model
        quantization_config: Quantization configuration
        
    Returns:
        Tuple of (mlx_model, tokenizer)
    """
    try:
        from transformers import AutoModel
        from mlx_lm.convert import convert_model, save_model
        
        # Load HuggingFace model
        logger.info(f"Loading {model_id} from HuggingFace...")
        torch_model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        config = AutoConfig.from_pretrained(model_id)
        
        # Convert to MLX format
        logger.info("Converting to MLX format...")
        mlx_model = convert_model(torch_model, config)
        
        # Quantize if requested
        if quantize:
            from src.backends.mlx.mlx_quantization import MLXQuantizer, MLXQuantizationConfig
            
            quant_config = MLXQuantizationConfig(**quantization_config) \
                if quantization_config else MLXQuantizationConfig()
            
            quantizer = MLXQuantizer(quant_config)
            mlx_model = quantizer.quantize_model(mlx_model)
        
        # Save model
        save_model(mlx_model, save_path)
        tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
        
        return mlx_model, tokenizer
        
    except ImportError as e:
        logger.error(f"Required dependencies not available: {e}")
        logger.info("Install with: pip install mlx-lm transformers")
        raise


def convert_checkpoint(
    checkpoint_path: str,
    output_path: str,
    from_framework: str = "pytorch",
    to_framework: str = "mlx",
    model_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Convert a checkpoint between frameworks.
    
    Args:
        checkpoint_path: Path to input checkpoint
        output_path: Path to save converted checkpoint
        from_framework: Source framework (pytorch/mlx)
        to_framework: Target framework (pytorch/mlx)
        model_config: Model configuration
        
    Returns:
        Path to converted checkpoint
    """
    os.makedirs(output_path, exist_ok=True)
    
    if from_framework == "pytorch" and to_framework == "mlx":
        # Load PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Convert state dict
        mlx_state_dict = TensorConverter.convert_state_dict(
            checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint
            else checkpoint.get('state_dict', checkpoint),
            to_framework="mlx"
        )
        
        # Save MLX checkpoint
        mx.save(os.path.join(output_path, "weights.npz"), mlx_state_dict)
        
    elif from_framework == "mlx" and to_framework == "pytorch":
        # Load MLX checkpoint
        mlx_weights = mx.load(os.path.join(checkpoint_path, "weights.npz"))
        
        # Convert to PyTorch
        torch_state_dict = TensorConverter.convert_state_dict(
            mlx_weights,
            to_framework="torch"
        )
        
        # Save PyTorch checkpoint
        torch.save(
            torch_state_dict,
            os.path.join(output_path, "model.pt")
        )
    
    else:
        raise ValueError(
            f"Unsupported conversion: {from_framework} -> {to_framework}"
        )
    
    # Save metadata
    metadata = {
        "conversion": {
            "from": from_framework,
            "to": to_framework,
            "model_config": model_config,
        }
    }
    
    with open(os.path.join(output_path, "conversion_info.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Converted checkpoint saved to {output_path}")
    
    return output_path


__all__ = [
    'TensorConverter',
    'LayerConverter',
    'ModelConverter',
    'convert_huggingface_to_mlx',
    'convert_checkpoint',
]