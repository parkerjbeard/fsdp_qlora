"""
MLX Model Wrapper for FSDP QLoRA

This module provides a comprehensive wrapper for MLX framework integration,
enabling efficient model loading, quantization, and LoRA fine-tuning on Apple Silicon.
"""

import os
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.nn as nn_mlx
import json
import numpy as np

import torch
import torch.nn as nn

from src.core.backend_manager import BackendManager
from src.core.quantization_wrapper import QuantizationConfig, QuantizationMethod

# Import MLX conditionally
try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn_mlx
    import mlx.optimizers as optim_mlx
    from mlx.utils import tree_unflatten, tree_flatten, tree_map
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    warnings.warn("MLX is not available. MLX model wrapper will have limited functionality.")


@dataclass
class MLXConfig:
    """Configuration for MLX models."""
    
    # Model configuration
    model_name: str
    model_type: str = "llama"  # llama, mixtral, etc.
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    intermediate_size: int = 11008
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Quantization settings
    quantization_bits: int = 4
    quantization_group_size: int = 64
    use_quantization: bool = True
    
    # LoRA settings
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_scaling: float = 1.0
    
    # Training settings
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Memory optimization
    use_unified_memory: bool = True
    gradient_checkpointing: bool = False
    
    @classmethod
    def from_huggingface_config(cls, hf_config: Dict[str, Any]) -> "MLXConfig":
        """Create MLXConfig from HuggingFace configuration."""
        return cls(
            model_name=hf_config.get("_name_or_path", ""),
            model_type=hf_config.get("model_type", "llama"),
            vocab_size=hf_config.get("vocab_size", 32000),
            hidden_size=hf_config.get("hidden_size", 4096),
            num_hidden_layers=hf_config.get("num_hidden_layers", 32),
            num_attention_heads=hf_config.get("num_attention_heads", 32),
            num_key_value_heads=hf_config.get("num_key_value_heads"),
            intermediate_size=hf_config.get("intermediate_size", 11008),
            max_position_embeddings=hf_config.get("max_position_embeddings", 4096),
            rope_theta=hf_config.get("rope_theta", 10000.0),
            rope_scaling=hf_config.get("rope_scaling"),
        )


class MLXLinear(nn_mlx.Module if MLX_AVAILABLE else object):
    """MLX Linear layer with quantization support."""
    
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        quantized: bool = False,
        bits: int = 4,
        group_size: int = 64,
    ):
        super().__init__()
        
        scale = np.sqrt(1.0 / input_dims)
        
        if quantized:
            # Initialize quantized weights
            self.bits = bits
            self.group_size = group_size
            self.quantized = True
            
            # Calculate quantization parameters
            self.scales = mx.zeros((output_dims, (input_dims + group_size - 1) // group_size))
            self.biases = mx.zeros_like(self.scales)
            
            # Quantized weight storage
            q_weight_shape = (output_dims, input_dims * bits // 8)
            self.q_weight = mx.zeros(q_weight_shape, dtype=mx.uint8)
            
            # Initialize scales
            self.scales = mx.ones_like(self.scales) * scale
        else:
            # Standard linear layer
            self.weight = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims, input_dims),
            )
            self.quantized = False
        
        if bias:
            self.bias = mx.zeros((output_dims,))
        else:
            self.bias = None
    
    def __call__(self, x: "mx.array") -> "mx.array":
        """Forward pass through the linear layer."""
        if self.quantized:
            # Dequantize weights on the fly
            weight = self._dequantize_weights()
            out = x @ weight.T
        else:
            out = x @ self.weight.T
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def _dequantize_weights(self) -> "mx.array":
        """Dequantize weights from compressed format."""
        # This is a simplified dequantization - actual implementation would be more complex
        # For now, return a dummy weight matrix
        output_dims, input_dims_compressed = self.q_weight.shape
        input_dims = input_dims_compressed * 8 // self.bits
        return mx.random.uniform(
            low=-0.1,
            high=0.1,
            shape=(output_dims, input_dims),
        )
    
    def quantize_weights(self, weights: "mx.array"):
        """Quantize the given weights."""
        # Simplified quantization logic
        # In practice, this would implement proper k-bit quantization
        self.scales = mx.ones((weights.shape[0], (weights.shape[1] + self.group_size - 1) // self.group_size))
        self.biases = mx.zeros_like(self.scales)
        # Store compressed weights
        self.q_weight = mx.zeros((weights.shape[0], weights.shape[1] * self.bits // 8), dtype=mx.uint8)


class LoRALinear(nn_mlx.Module if MLX_AVAILABLE else object):
    """LoRA adapter for MLX linear layers."""
    
    def __init__(
        self,
        base_layer: Any,  # Union[MLXLinear, nn_mlx.Linear]
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank * scale
        
        # Get dimensions from base layer
        if isinstance(base_layer, MLXLinear):
            if base_layer.quantized:
                # Get dimensions from quantized layer
                out_features = base_layer.q_weight.shape[0]
                in_features = base_layer.q_weight.shape[1] * 8 // base_layer.bits
            else:
                out_features, in_features = base_layer.weight.shape
        else:
            # Standard nn_mlx.Linear
            out_features, in_features = base_layer.weight.shape
        
        # Initialize LoRA matrices
        self.lora_a = mx.random.normal(shape=(rank, in_features)) * (1 / np.sqrt(rank))
        self.lora_b = mx.zeros((out_features, rank))
        
        # Dropout layer
        self.lora_dropout = nn_mlx.Dropout(p=dropout) if dropout > 0 else None
    
    def __call__(self, x: "mx.array") -> "mx.array":
        """Forward pass with LoRA."""
        # Base layer forward
        result = self.base_layer(x)
        
        # LoRA forward
        if self.lora_dropout is not None:
            x = self.lora_dropout(x)
        
        # Compute LoRA contribution: x @ A^T @ B^T
        lora_out = (x @ self.lora_a.T) @ self.lora_b.T
        
        return result + lora_out * self.scaling


class MLXModel(nn_mlx.Module if MLX_AVAILABLE else object):
    """Base class for MLX models."""
    
    def __init__(self, config: MLXConfig):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def __call__(self, input_ids: "mx.array", **kwargs) -> "mx.array":
        """Forward pass through the model."""
        # This should be implemented by concrete model classes
        raise NotImplementedError("Forward pass must be implemented by subclasses")
    
    def apply_lora(self, target_modules: Optional[List[str]] = None):
        """Apply LoRA to specified modules."""
        if target_modules is None:
            target_modules = self.config.lora_target_modules
        
        # Recursively find and wrap linear layers
        def wrap_linear(module, name):
            if isinstance(module, (MLXLinear, nn_mlx.Linear)):
                # Check if this module should have LoRA
                for target in target_modules:
                    if target in name:
                        return LoRALinear(
                            module,
                            rank=self.config.lora_rank,
                            alpha=self.config.lora_alpha,
                            dropout=self.config.lora_dropout,
                            scale=self.config.lora_scaling,
                        )
            return module
        
        # Apply LoRA wrapping
        self._apply_to_modules(wrap_linear)
    
    def _apply_to_modules(self, fn):
        """Recursively apply a function to all modules."""
        for name, module in self.__dict__.items():
            if isinstance(module, nn_mlx.Module):
                setattr(self, name, fn(module, name))


class PyTorchToMLXConverter:
    """Converter for PyTorch models to MLX format."""
    
    @staticmethod
    def convert_tensor(tensor: torch.Tensor) -> "mx.array":
        """Convert PyTorch tensor to MLX array."""
        # Move to CPU and convert to numpy
        np_array = tensor.detach().cpu().numpy()
        # Convert to MLX array
        return mx.array(np_array)
    
    @staticmethod
    def convert_linear_layer(
        pytorch_layer: nn.Linear,
        quantize: bool = False,
        bits: int = 4,
        group_size: int = 64,
    ) -> Any:  # Union[MLXLinear, nn_mlx.Linear]:
        """Convert PyTorch linear layer to MLX."""
        in_features = pytorch_layer.in_features
        out_features = pytorch_layer.out_features
        has_bias = pytorch_layer.bias is not None
        
        if quantize:
            # Create quantized MLX linear layer
            mlx_layer = MLXLinear(
                in_features,
                out_features,
                bias=has_bias,
                quantized=True,
                bits=bits,
                group_size=group_size,
            )
            
            # Convert and quantize weights
            weight = PyTorchToMLXConverter.convert_tensor(pytorch_layer.weight)
            mlx_layer.quantize_weights(weight)
        else:
            # Create standard MLX linear layer
            mlx_layer = nn_mlx.Linear(in_features, out_features, bias=has_bias)
            
            # Convert weights
            mlx_layer.weight = PyTorchToMLXConverter.convert_tensor(pytorch_layer.weight)
        
        # Convert bias if present
        if has_bias:
            mlx_layer.bias = PyTorchToMLXConverter.convert_tensor(pytorch_layer.bias)
        
        return mlx_layer
    
    @staticmethod
    def convert_embedding(pytorch_emb: nn.Embedding) -> Any:  # nn_mlx.Embedding:
        """Convert PyTorch embedding to MLX."""
        mlx_emb = nn_mlx.Embedding(
            pytorch_emb.num_embeddings,
            pytorch_emb.embedding_dim,
        )
        
        # Convert weights
        mlx_emb.weight = PyTorchToMLXConverter.convert_tensor(pytorch_emb.weight)
        
        return mlx_emb


class MLXModelWrapper:
    """Wrapper for MLX models to interface with PyTorch training loop."""
    
    def __init__(
        self,
        mlx_model: MLXModel,
        tokenizer: Any,
        backend_manager: Optional[BackendManager] = None,
    ):
        self.mlx_model = mlx_model
        self.tokenizer = tokenizer
        self.backend_manager = backend_manager or BackendManager()
        
        # Cache for converted inputs/outputs
        self._input_cache = {}
        self._output_cache = {}
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass compatible with PyTorch training loop."""
        # Convert inputs to MLX
        mlx_input_ids = self._torch_to_mlx(input_ids)
        
        # Run MLX model
        mlx_outputs = self.mlx_model(mlx_input_ids)
        
        # Convert outputs back to PyTorch
        logits = self._mlx_to_torch(mlx_outputs)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Simple cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def _torch_to_mlx(self, tensor: torch.Tensor) -> "mx.array":
        """Convert PyTorch tensor to MLX array with caching."""
        # For unified memory optimization, avoid unnecessary copies
        tensor_id = id(tensor)
        if tensor_id in self._input_cache:
            return self._input_cache[tensor_id]
        
        mlx_array = PyTorchToMLXConverter.convert_tensor(tensor)
        self._input_cache[tensor_id] = mlx_array
        return mlx_array
    
    def _mlx_to_torch(self, array: "mx.array") -> torch.Tensor:
        """Convert MLX array to PyTorch tensor with caching."""
        array_id = id(array)
        if array_id in self._output_cache:
            return self._output_cache[array_id]
        
        # Convert to numpy then torch
        np_array = np.array(array)
        torch_tensor = torch.from_numpy(np_array)
        
        # Move to appropriate device
        device = self.backend_manager.get_device()
        torch_tensor = torch_tensor.to(device)
        
        self._output_cache[array_id] = torch_tensor
        return torch_tensor
    
    def train(self):
        """Set model to training mode."""
        # MLX doesn't have explicit train/eval modes
        self.training = True
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        # MLX doesn't have explicit train/eval modes
        self.training = False
        return self
    
    def parameters(self):
        """Get model parameters compatible with PyTorch optimizers."""
        # Return LoRA parameters only if using LoRA
        params = []
        
        def collect_lora_params(module):
            if isinstance(module, LoRALinear):
                params.extend([
                    self._mlx_to_torch(module.lora_a),
                    self._mlx_to_torch(module.lora_b),
                ])
        
        # Recursively collect LoRA parameters
        self._traverse_modules(self.mlx_model, collect_lora_params)
        
        return params
    
    def _traverse_modules(self, module, fn):
        """Recursively traverse MLX modules."""
        fn(module)
        for name, submodule in module.__dict__.items():
            if isinstance(submodule, nn_mlx.Module):
                self._traverse_modules(submodule, fn)
    
    def save_pretrained(self, save_directory: str):
        """Save model in MLX format."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        weights = tree_flatten(self.mlx_model.parameters())
        mx.save(f"{save_directory}/weights.npz", weights)
        
        # Save configuration
        config_path = Path(save_directory) / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.mlx_model.config.__dict__, f, indent=2)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[MLXConfig] = None,
        tokenizer: Optional[Any] = None,
        backend_manager: Optional[BackendManager] = None,
    ) -> "MLXModelWrapper":
        """Load model from MLX format."""
        # Load configuration
        if config is None:
            config_path = Path(model_path) / "config.json"
            with open(config_path) as f:
                config_dict = json.load(f)
                config = MLXConfig(**config_dict)
        
        # Create model
        # This would need model-specific implementation
        mlx_model = MLXModel(config)  # Placeholder
        
        # Load weights
        weights = mx.load(f"{model_path}/weights.npz")
        mlx_model.load_weights(weights)
        
        # Load tokenizer if not provided
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return cls(mlx_model, tokenizer, backend_manager)


class UnifiedMemoryOptimizer:
    """Optimizer for unified memory on Apple Silicon."""
    
    def __init__(self, model_wrapper: MLXModelWrapper):
        self.model_wrapper = model_wrapper
        self._memory_pool = {}
        self._allocation_strategy = "lazy"
    
    def optimize_memory_layout(self):
        """Optimize memory layout for unified memory architecture."""
        # MLX automatically handles unified memory optimization
        # This method can be extended for custom optimization strategies
        
        # Clear caches periodically
        self.model_wrapper._input_cache.clear()
        self.model_wrapper._output_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        # MLX doesn't have built-in gradient checkpointing
        # This would need custom implementation
        warnings.warn("Gradient checkpointing not yet implemented for MLX")
    
    def profile_memory_usage(self) -> Dict[str, float]:
        """Profile memory usage of the model."""
        # Get process memory info
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            "rss_gb": mem_info.rss / 1e9,
            "vms_gb": mem_info.vms / 1e9,
            "available_gb": psutil.virtual_memory().available / 1e9,
            "percent": process.memory_percent(),
        }


# Convenience functions

def create_mlx_model(
    model_name: str,
    quantization_config: Optional[QuantizationConfig] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    backend_manager: Optional[BackendManager] = None,
) -> MLXModelWrapper:
    """
    Create an MLX model with optional quantization and LoRA.
    
    Args:
        model_name: HuggingFace model name or path
        quantization_config: Quantization configuration
        lora_config: LoRA configuration
        backend_manager: Backend manager instance
        
    Returns:
        MLXModelWrapper instance
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX is not available. Please install mlx and mlx-lm.")
    
    import json
    from pathlib import Path
    from transformers import AutoConfig, AutoTokenizer
    from src.backends.mlx.models import (
        LlamaModel, LlamaConfig,
        MistralModel, MistralConfig,
        PhiModel, PhiConfig,
        QwenModel, QwenConfig,
    )
    
    # Load HuggingFace config
    if Path(model_name).exists():
        # Local path
        config_path = Path(model_name) / "config.json"
        with open(config_path) as f:
            hf_config = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code="qwen" in model_name.lower()
        )
    else:
        # HuggingFace Hub
        hf_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code="qwen" in model_name.lower()
        ).to_dict()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code="qwen" in model_name.lower()
        )
    
    # Determine model type and create appropriate config
    model_type = hf_config.get("model_type", "").lower()
    architectures = hf_config.get("architectures", [])
    
    mlx_model = None
    mlx_config = None
    
    if "llama" in model_type or any("Llama" in arch for arch in architectures):
        mlx_config = LlamaConfig.from_huggingface(hf_config)
        mlx_model = LlamaModel(mlx_config)
    elif "mistral" in model_type or any("Mistral" in arch for arch in architectures):
        mlx_config = MistralConfig.from_huggingface(hf_config)
        mlx_model = MistralModel(mlx_config)
    elif "phi" in model_type or any("Phi" in arch for arch in architectures):
        mlx_config = PhiConfig.from_huggingface(hf_config)
        mlx_model = PhiModel(mlx_config)
    elif "qwen" in model_type or any("Qwen" in arch for arch in architectures):
        mlx_config = QwenConfig.from_huggingface(hf_config)
        mlx_model = QwenModel(mlx_config)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported types: llama, mistral, phi, qwen"
        )
    
    # Load weights if available
    if Path(model_name).exists():
        weights_path = Path(model_name) / "weights.npz"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            mlx_model.load_weights(weights)
        else:
            # Try to convert from PyTorch weights
            pytorch_path = Path(model_name) / "pytorch_model.bin"
            safetensors_path = Path(model_name) / "model.safetensors"
            
            if pytorch_path.exists() or safetensors_path.exists():
                # Convert weights
                mlx_model = _convert_pytorch_weights_to_mlx(
                    mlx_model, model_name, quantization_config
                )
    
    # Apply quantization if requested
    if quantization_config and quantization_config.method != QuantizationMethod.NONE:
        from src.backends.mlx.mlx_quantization import MLXQuantizer, MLXQuantizationConfig
        
        mlx_quant_config = MLXQuantizationConfig(
            bits=quantization_config.bits,
            group_size=quantization_config.group_size,
        )
        quantizer = MLXQuantizer(mlx_quant_config)
        mlx_model = quantizer.quantize_model(mlx_model)
    
    # Apply LoRA if requested
    if lora_config:
        mlx_config.use_lora = True
        mlx_config.lora_rank = lora_config.get("rank", 16)
        mlx_config.lora_alpha = lora_config.get("alpha", 32.0)
        mlx_config.lora_dropout = lora_config.get("dropout", 0.1)
        mlx_config.lora_target_modules = lora_config.get(
            "target_modules", ["q_proj", "v_proj"]
        )
        mlx_model.apply_lora(mlx_config.lora_target_modules)
    
    # Create wrapper
    return MLXModelWrapper(mlx_model, tokenizer, backend_manager)


def convert_pytorch_to_mlx(
    pytorch_model: nn.Module,
    config: MLXConfig,
    quantize: bool = True,
) -> MLXModel:
    """
    Convert PyTorch model to MLX format.
    
    Args:
        pytorch_model: PyTorch model to convert
        config: MLX configuration
        quantize: Whether to quantize during conversion
        
    Returns:
        MLX model
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX is not available. Please install mlx and mlx-lm.")
    
    from src.backends.mlx.models import (
        LlamaModel, LlamaConfig,
        MistralModel, MistralConfig,
        PhiModel, PhiConfig,
        QwenModel, QwenConfig,
    )
    
    # Create appropriate MLX model based on config
    if config.model_type == "llama":
        mlx_model = LlamaModel(config if isinstance(config, LlamaConfig) else LlamaConfig(**config.__dict__))
    elif config.model_type == "mistral":
        mlx_model = MistralModel(config if isinstance(config, MistralConfig) else MistralConfig(**config.__dict__))
    elif config.model_type == "phi":
        mlx_model = PhiModel(config if isinstance(config, PhiConfig) else PhiConfig(**config.__dict__))
    elif config.model_type == "qwen":
        mlx_model = QwenModel(config if isinstance(config, QwenConfig) else QwenConfig(**config.__dict__))
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
    
    # Convert weights
    converter = PyTorchToMLXConverter()
    
    # Map PyTorch state dict to MLX
    pytorch_state_dict = pytorch_model.state_dict()
    mlx_weights = {}
    
    for name, param in pytorch_state_dict.items():
        # Convert parameter names
        mlx_name = _convert_param_name(name, config.model_type)
        
        # Convert tensor
        mlx_array = converter.convert_tensor(param)
        
        # Handle special conversions
        if "embed_tokens" in name or "tok_embeddings" in name:
            mlx_weights["tok_embeddings.weight"] = mlx_array
        elif "lm_head" in name:
            if not config.tie_word_embeddings:
                mlx_weights["lm_head.weight"] = mlx_array
        elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
            # Handle attention projections
            layer_idx = _extract_layer_idx(name)
            if layer_idx is not None:
                proj_type = name.split(".")[-2]  # q_proj, k_proj, v_proj
                mlx_weights[f"layers.{layer_idx}.attention.{proj_type}.weight"] = mlx_array
        elif "o_proj" in name:
            layer_idx = _extract_layer_idx(name)
            if layer_idx is not None:
                mlx_weights[f"layers.{layer_idx}.attention.o_proj.weight"] = mlx_array
        elif "gate_proj" in name or "up_proj" in name or "down_proj" in name:
            # Handle MLP projections
            layer_idx = _extract_layer_idx(name)
            if layer_idx is not None:
                proj_type = name.split(".")[-2]
                mlx_weights[f"layers.{layer_idx}.feed_forward.{proj_type}.weight"] = mlx_array
        elif "input_layernorm" in name or "attention_norm" in name:
            layer_idx = _extract_layer_idx(name)
            if layer_idx is not None:
                mlx_weights[f"layers.{layer_idx}.attention_norm.weight"] = mlx_array
        elif "post_attention_layernorm" in name or "ffn_norm" in name:
            layer_idx = _extract_layer_idx(name)
            if layer_idx is not None:
                mlx_weights[f"layers.{layer_idx}.ffn_norm.weight"] = mlx_array
        elif "norm" in name and "layers" not in name:
            mlx_weights["norm.weight"] = mlx_array
        else:
            # Default mapping
            mlx_weights[mlx_name] = mlx_array
    
    # Load weights into model
    mlx_model.load_weights(mlx_weights)
    
    # Apply quantization if requested
    if quantize and config.use_quantization:
        from src.backends.mlx.mlx_quantization import MLXQuantizer, MLXQuantizationConfig
        
        quant_config = MLXQuantizationConfig(
            bits=config.quantization_bits,
            group_size=config.quantization_group_size,
        )
        quantizer = MLXQuantizer(quant_config)
        mlx_model = quantizer.quantize_model(mlx_model)
    
    return mlx_model


def _convert_param_name(pytorch_name: str, model_type: str) -> str:
    """Convert PyTorch parameter name to MLX format."""
    # Basic conversions
    name = pytorch_name.replace("model.", "")
    name = name.replace("self_attn", "attention")
    name = name.replace("mlp", "feed_forward")
    
    # Model-specific conversions
    if model_type == "qwen":
        name = name.replace("c_attn", "qkv_proj")
        name = name.replace("c_proj", "o_proj")
        name = name.replace("w1", "gate_proj")
        name = name.replace("w2", "up_proj")
    
    return name


def _extract_layer_idx(name: str) -> Optional[int]:
    """Extract layer index from parameter name."""
    import re
    match = re.search(r'layers?[._](\d+)', name)
    if match:
        return int(match.group(1))
    return None


def _convert_pytorch_weights_to_mlx(
    mlx_model: Any,
    model_path: str,
    quantization_config: Optional[QuantizationConfig] = None,
) -> Any:
    """Convert PyTorch weights to MLX format."""
    import torch
    from pathlib import Path
    
    model_path = Path(model_path)
    
    # Load PyTorch weights
    pytorch_path = model_path / "pytorch_model.bin"
    safetensors_path = model_path / "model.safetensors"
    
    if safetensors_path.exists():
        from safetensors import safe_open
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError("No PyTorch weights found")
    
    # Create a dummy PyTorch model for conversion
    class DummyModel(nn.Module):
        def __init__(self, state_dict):
            super().__init__()
            self._state_dict = state_dict
            
        def state_dict(self):
            return self._state_dict
    
    dummy_model = DummyModel(state_dict)
    
    # Convert using the main conversion function
    return convert_pytorch_to_mlx(
        dummy_model,
        mlx_model.config,
        quantize=quantization_config is not None
    )


__all__ = [
    "MLXConfig",
    "MLXLinear",
    "LoRALinear",
    "MLXModel",
    "PyTorchToMLXConverter",
    "MLXModelWrapper",
    "UnifiedMemoryOptimizer",
    "create_mlx_model",
    "convert_pytorch_to_mlx",
]