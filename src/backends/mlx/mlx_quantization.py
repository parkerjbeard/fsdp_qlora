"""
MLX Quantization Module

Provides native quantization support for Apple Silicon using MLX framework.
This is the optimal solution for quantization on Mac, offering:
- Native 1-8 bit quantization with Metal acceleration
- Group-wise quantization for better accuracy
- Mixed-precision support per layer
- QLoRA-style fine-tuning on quantized models
- Unified memory optimization
"""

import os
import json
import warnings
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map, tree_flatten
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    warnings.warn(
        "MLX not available. Install with: pip install mlx"
    )

try:
    from mlx_lm import load, generate, quantize as mlx_quantize
    from mlx_lm.models.base import KVCache
    from mlx_lm.lora import LoRALinear, fuse_lora
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False
    if MLX_AVAILABLE:
        warnings.warn(
            "mlx_lm not available. Install with: pip install mlx-lm"
        )

logger = logging.getLogger(__name__)


@dataclass
class MLXQuantizationConfig:
    """Configuration for MLX quantization."""
    
    # Quantization settings
    bits: int = 4  # 1-8 bits supported
    group_size: int = 64  # Group size for quantization
    
    # Per-layer configuration
    layer_bits: Dict[str, int] = field(default_factory=dict)
    skip_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    
    # Mixed precision
    default_bits: int = 4
    embedding_bits: int = 8  # Higher precision for embeddings
    output_bits: int = 8     # Higher precision for output layer
    
    # LoRA settings for fine-tuning
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Performance settings
    lazy_evaluation: bool = True
    unified_memory: bool = True
    stream_device: str = "gpu"  # gpu, cpu, or gpu,cpu
    
    def get_bits_for_layer(self, layer_name: str) -> int:
        """Get quantization bits for a specific layer."""
        # Check layer-specific config
        for pattern, bits in self.layer_bits.items():
            if pattern in layer_name:
                return bits
        
        # Check common patterns
        if any(skip in layer_name for skip in self.skip_modules):
            return 16  # No quantization for skip modules
        elif "embed" in layer_name.lower():
            return self.embedding_bits
        elif "lm_head" in layer_name.lower() or "output" in layer_name.lower():
            return self.output_bits
        else:
            return self.default_bits


class MLXQuantizedLinear(nn.Module):
    """MLX quantized linear layer with group-wise quantization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 4,
        group_size: int = 64,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        
        # Quantization parameters
        self.scales = None
        self.biases = None
        self.quantized_weight = None
        
        # Original weight shape
        self.weight_shape = (out_features, in_features)
        
        # Bias parameter
        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None
    
    def quantize_weights(self, weight: "mx.array") -> None:
        """Quantize weights using MLX native quantization."""
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available")
        
        # Ensure correct shape
        if weight.shape != self.weight_shape:
            weight = weight.reshape(self.weight_shape)
        
        # Quantize using MLX's native method
        if hasattr(mx, 'quantize'):
            # Use MLX's built-in quantization
            self.quantized_weight, self.scales, self.biases = mx.quantize(
                weight,
                bits=self.bits,
                group_size=self.group_size,
            )
        else:
            # Manual quantization fallback
            self.quantized_weight, self.scales, self.biases = self._manual_quantize(
                weight
            )
    
    def _manual_quantize(
        self, 
        weight: "mx.array"
    ) -> Tuple["mx.array", "mx.array", "mx.array"]:
        """Manual quantization implementation."""
        # Reshape weight for group-wise quantization
        out_features, in_features = weight.shape
        
        # Pad if necessary
        if in_features % self.group_size != 0:
            pad_size = self.group_size - (in_features % self.group_size)
            weight = mx.pad(weight, [(0, 0), (0, pad_size)])
        
        # Reshape into groups
        grouped = weight.reshape(out_features, -1, self.group_size)
        
        # Compute scales and biases per group
        w_max = mx.max(grouped, axis=2, keepdims=True)
        w_min = mx.min(grouped, axis=2, keepdims=True)
        
        # Symmetric quantization
        scales = (w_max - w_min) / (2**self.bits - 1)
        scales = mx.maximum(scales, 1e-8)  # Avoid division by zero
        biases = w_min
        
        # Quantize
        quantized = mx.round((grouped - biases) / scales)
        quantized = mx.clip(quantized, 0, 2**self.bits - 1)
        
        # Pack if bits < 8
        if self.bits < 8:
            quantized = self._pack_bits(quantized.astype(mx.uint8))
        else:
            quantized = quantized.astype(mx.int8 if self.bits == 8 else mx.int16)
        
        return quantized, scales.squeeze(axis=2), biases.squeeze(axis=2)
    
    def _pack_bits(self, quantized: "mx.array") -> "mx.array":
        """Pack low-bit values for storage efficiency."""
        # Pack multiple values into single bytes
        # This is a simplified version - MLX handles this internally
        return quantized
    
    def __call__(self, x: "mx.array") -> "mx.array":
        """Forward pass with dequantization."""
        if self.quantized_weight is None:
            raise ValueError("Weights not quantized")
        
        # Dequantize on the fly
        if hasattr(mx, 'dequantize'):
            weight = mx.dequantize(
                self.quantized_weight,
                self.scales,
                self.biases,
                group_size=self.group_size,
                bits=self.bits,
            )
        else:
            # Manual dequantization
            weight = self._manual_dequantize()
        
        # Reshape weight back to original shape
        weight = weight.reshape(self.weight_shape)
        
        # Matrix multiplication
        output = x @ weight.T
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _manual_dequantize(self) -> "mx.array":
        """Manual dequantization implementation."""
        # Dequantize
        dequantized = self.quantized_weight.astype(mx.float32)
        
        # Reshape to groups
        out_features = self.weight_shape[0]
        dequantized = dequantized.reshape(
            out_features, -1, self.group_size
        )
        
        # Apply scales and biases
        scales = self.scales.reshape(out_features, -1, 1)
        biases = self.biases.reshape(out_features, -1, 1)
        
        dequantized = dequantized * scales + biases
        
        # Reshape back
        dequantized = dequantized.reshape(out_features, -1)
        
        # Remove padding if necessary
        if dequantized.shape[1] > self.weight_shape[1]:
            dequantized = dequantized[:, :self.weight_shape[1]]
        
        return dequantized


class MLXQLoRAAdapter(nn.Module):
    """QLoRA adapter for fine-tuning quantized models."""
    
    def __init__(
        self,
        base_layer: MLXQuantizedLinear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices (not quantized)
        self.lora_a = mx.random.normal(
            (rank, base_layer.in_features),
            scale=1.0 / np.sqrt(rank),
        )
        self.lora_b = mx.zeros((base_layer.out_features, rank))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def __call__(self, x: "mx.array") -> "mx.array":
        """Forward pass with LoRA adaptation."""
        # Base layer forward (quantized)
        base_output = self.base_layer(x)
        
        # LoRA forward
        if self.dropout:
            x = self.dropout(x)
        
        lora_output = (x @ self.lora_a.T @ self.lora_b.T) * self.scaling
        
        return base_output + lora_output


class MLXQuantizer:
    """Main quantizer for MLX models."""
    
    def __init__(self, config: MLXQuantizationConfig):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required. Install with: pip install mlx")
        
        self.config = config
        
        # Set MLX device/stream
        if self.config.stream_device:
            devices = self.config.stream_device.split(',')
            for device in devices:
                if device == "gpu":
                    mx.set_default_device(mx.gpu)
                elif device == "cpu":
                    mx.set_default_device(mx.cpu)
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[List["mx.array"]] = None,
    ) -> nn.Module:
        """Quantize an MLX model."""
        # Get all linear layers
        linear_layers = self._find_linear_layers(model)
        
        for name, layer in linear_layers:
            if any(skip in name for skip in self.config.skip_modules):
                continue
            
            # Get bits for this layer
            bits = self.config.get_bits_for_layer(name)
            
            if bits < 16:  # Only quantize if < 16 bits
                # Create quantized layer
                quantized_layer = MLXQuantizedLinear(
                    layer.weight.shape[1],  # in_features
                    layer.weight.shape[0],  # out_features
                    bias=layer.bias is not None,
                    bits=bits,
                    group_size=self.config.group_size,
                )
                
                # Quantize weights
                quantized_layer.quantize_weights(layer.weight)
                
                # Copy bias
                if layer.bias is not None:
                    quantized_layer.bias = layer.bias
                
                # Replace layer
                self._replace_layer(model, name, quantized_layer)
        
        logger.info(f"Quantized {len(linear_layers)} layers")
        
        return model
    
    def add_lora_adapters(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
    ) -> nn.Module:
        """Add LoRA adapters to quantized model for fine-tuning."""
        target_modules = target_modules or self.config.lora_target_modules
        
        quantized_layers = self._find_quantized_layers(model)
        adapters_added = 0
        
        for name, layer in quantized_layers:
            # Check if this module should have LoRA
            if any(target in name for target in target_modules):
                # Create QLoRA adapter
                adapter = MLXQLoRAAdapter(
                    layer,
                    rank=self.config.lora_rank,
                    alpha=self.config.lora_alpha,
                    dropout=self.config.lora_dropout,
                )
                
                # Replace layer with adapter
                self._replace_layer(model, name, adapter)
                adapters_added += 1
        
        logger.info(f"Added {adapters_added} LoRA adapters")
        
        return model
    
    def _find_linear_layers(
        self, 
        model: nn.Module,
        prefix: str = "",
    ) -> List[Tuple[str, nn.Linear]]:
        """Find all linear layers in model."""
        layers = []
        
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(module, nn.Linear):
                layers.append((full_name, module))
            else:
                # Recursively search
                layers.extend(self._find_linear_layers(module, full_name))
        
        return layers
    
    def _find_quantized_layers(
        self,
        model: nn.Module,
        prefix: str = "",
    ) -> List[Tuple[str, MLXQuantizedLinear]]:
        """Find all quantized linear layers in model."""
        layers = []
        
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(module, MLXQuantizedLinear):
                layers.append((full_name, module))
            else:
                # Recursively search
                layers.extend(self._find_quantized_layers(module, full_name))
        
        return layers
    
    def _replace_layer(self, model: nn.Module, name: str, new_layer: nn.Module):
        """Replace a layer in the model."""
        parts = name.split('.')
        parent = model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_layer)
    
    def save_quantized_model(self, model: nn.Module, path: str):
        """Save quantized model in MLX format."""
        os.makedirs(path, exist_ok=True)
        
        # Save weights
        weights = tree_flatten(model.parameters())
        mx.save(os.path.join(path, "weights.npz"), weights)
        
        # Save config
        config_dict = {
            "quantization": {
                "bits": self.config.bits,
                "group_size": self.config.group_size,
                "layer_bits": self.config.layer_bits,
            }
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved quantized model to {path}")
    
    def load_quantized_model(
        self,
        model: nn.Module,
        path: str,
    ) -> nn.Module:
        """Load quantized model from MLX format."""
        # Load weights
        weights = mx.load(os.path.join(path, "weights.npz"))
        
        # Update model parameters
        model.update(tree_map("mx.array", weights))
        
        return model


def create_mlx_quantized_model(
    model_path: str,
    config: Optional[MLXQuantizationConfig] = None,
    load_from_hub: bool = True,
) -> Tuple[nn.Module, Any]:
    """
    Create or load an MLX quantized model.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        config: Quantization configuration
        load_from_hub: Load from HuggingFace Hub
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if not MLX_LM_AVAILABLE:
        raise ImportError("mlx_lm is required. Install with: pip install mlx-lm")
    
    config = config or MLXQuantizationConfig()
    
    if load_from_hub:
        # Load from HuggingFace (potentially pre-quantized)
        model, tokenizer = load(model_path)
        
        # Check if already quantized
        if not any(isinstance(m, MLXQuantizedLinear) 
                   for _, m in model.named_modules()):
            # Quantize the model
            quantizer = MLXQuantizer(config)
            model = quantizer.quantize_model(model)
    else:
        # Load local model
        import json
        from pathlib import Path
        
        model_path = Path(model_name_or_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
        # Look for MLX model files
        weights_path = None
        config_path = None
        
        if model_path.is_dir():
            # Check for weights file
            for fname in ['weights.npz', 'model.npz', 'mlx_model.npz']:
                if (model_path / fname).exists():
                    weights_path = model_path / fname
                    break
                    
            # Check for config
            for fname in ['config.json', 'mlx_config.json']:
                if (model_path / fname).exists():
                    config_path = model_path / fname
                    break
                    
            # Check for tokenizer
            tokenizer_path = model_path / 'tokenizer.json'
            if not tokenizer_path.exists():
                tokenizer_path = None
        else:
            # Single file provided
            if model_path.suffix == '.npz':
                weights_path = model_path
                # Look for config in same directory
                config_path = model_path.with_suffix('.json')
                if not config_path.exists():
                    config_path = None
            else:
                raise ValueError(f"Unsupported file format: {model_path.suffix}")
                
        if weights_path is None:
            raise FileNotFoundError(f"No weights file found in {model_path}")
            
        # Load weights
        weights = mx.load(str(weights_path))
        
        # Load config if available
        if config_path and config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)
        else:
            # Try to infer config from weights
            model_config = {}
            
        # Create model based on architecture
        # This is simplified - real implementation would use proper model classes
        if 'model_type' in model_config:
            model_type = model_config['model_type']
            
            if model_type == 'llama':
                # Create Llama model
                from mlx.models.llama import Llama
                model = Llama(model_config)
            elif model_type == 'mistral':
                # Create Mistral model
                from mlx.models.mistral import Mistral
                model = Mistral(model_config)
            else:
                # Generic model
                class GenericMLXModel(nn.Module):
                    def __init__(self, config):
                        super().__init__()
                        self.config = config
                        
                model = GenericMLXModel(model_config)
        else:
            # Create a generic container
            class MLXModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    
            model = MLXModel()
            
        # Load weights into model
        model.load_weights(weights)
        
        # Apply quantization if model is not already quantized
        if quantize and not any('quantized' in name for name in weights.keys()):
            quantizer = MLXQuantizer(config)
            model = quantizer.quantize_model(model)
            
        # Load tokenizer if available
        tokenizer = None
        if tokenizer_path and tokenizer_path.exists():
            try:
                from mlx.tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
            except ImportError:
                warnings.warn("MLX tokenizer not available")
    
    return model, tokenizer


def fine_tune_quantized_model(
    model: nn.Module,
    train_data: List[Dict[str, "mx.array"]],
    config: MLXQuantizationConfig,
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 1,
) -> nn.Module:
    """
    Fine-tune a quantized model using QLoRA.
    
    Args:
        model: Quantized MLX model
        train_data: Training data
        config: Quantization config with LoRA settings
        learning_rate: Learning rate
        num_epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Fine-tuned model
    """
    # Add LoRA adapters
    quantizer = MLXQuantizer(config)
    model = quantizer.add_lora_adapters(model)
    
    # Get trainable parameters (LoRA only)
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            trainable_params.append(param)
    
    # Create optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            
            # Forward pass
            loss = 0
            for sample in batch:
                output = model(sample["input_ids"])
                target = sample["labels"]
                
                # Cross entropy loss
                loss += mx.mean(
                    nn.losses.cross_entropy(
                        output.reshape(-1, output.shape[-1]),
                        target.reshape(-1),
                        reduction="none",
                    )
                )
            
            loss = loss / len(batch)
            
            # Backward pass
            loss_grad_fn = mx.value_and_grad(lambda: loss)
            loss_value, grads = loss_grad_fn()
            
            # Update weights
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            
            total_loss += loss_value.item()
        
        avg_loss = total_loss / (len(train_data) / batch_size)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model


__all__ = [
    'MLXQuantizationConfig',
    'MLXQuantizedLinear',
    'MLXQLoRAAdapter',
    'MLXQuantizer',
    'create_mlx_quantized_model',
    'fine_tune_quantized_model',
]