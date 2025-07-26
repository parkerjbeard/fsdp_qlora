"""
LLaMA model implementation for MLX

Implements the LLaMA architecture optimized for Apple Silicon using MLX framework.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from .base import BaseModelConfig, MLXModelBase

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


@dataclass
class LlamaConfig(BaseModelConfig):
    """Configuration for LLaMA models."""
    model_type: str = "llama"
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    use_bias: bool = False
    
    # LLaMA specific
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    
    @classmethod
    def from_huggingface(cls, hf_config: Dict[str, Any]) -> "LlamaConfig":
        """Create config from HuggingFace format."""
        return cls(
            vocab_size=hf_config.get("vocab_size", 32000),
            hidden_size=hf_config.get("hidden_size", 4096),
            intermediate_size=hf_config.get("intermediate_size", 11008),
            num_hidden_layers=hf_config.get("num_hidden_layers", 32),
            num_attention_heads=hf_config.get("num_attention_heads", 32),
            num_key_value_heads=hf_config.get("num_key_value_heads"),
            max_position_embeddings=hf_config.get("max_position_embeddings", 4096),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-5),
            rope_theta=hf_config.get("rope_theta", 10000.0),
            rope_scaling=hf_config.get("rope_scaling"),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
            use_bias=hf_config.get("use_bias", False),
            attention_bias=hf_config.get("attention_bias", False),
            attention_dropout=hf_config.get("attention_dropout", 0.0),
            mlp_bias=hf_config.get("mlp_bias", False),
        )


class LlamaModel(MLXModelBase):
    """LLaMA model implementation for MLX."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
    @staticmethod
    def from_pretrained(
        model_path: str,
        config: Optional[LlamaConfig] = None,
    ) -> "LlamaModel":
        """Load pretrained LLaMA model."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for LlamaModel")
            
        import json
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # Load config if not provided
        if config is None:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    hf_config = json.load(f)
                config = LlamaConfig.from_huggingface(hf_config)
            else:
                raise FileNotFoundError(f"Config file not found at {config_path}")
                
        # Create model
        model = LlamaModel(config)
        
        # Load weights
        weights_path = model_path / "weights.npz"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            model.load_weights(weights)
        else:
            # Try loading from safetensors
            safetensors_path = model_path / "model.safetensors"
            if safetensors_path.exists():
                # Safetensors loading is not yet supported for MLX
                raise ValueError(
                    f"Safetensors format is not yet supported for MLX models. "
                    f"Please convert the model to MLX format first using the MLX conversion tools. "
                    f"Found safetensors at: {safetensors_path}"
                )
            else:
                raise FileNotFoundError(
                    f"Model weights not found at {model_path}. "
                    f"Expected either .npz or .safetensors files."
                )
                
        return model
        
    def sanitize(self) -> None:
        """Sanitize model weights for optimization."""
        # Custom optimizations for LLaMA models
        pass