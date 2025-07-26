"""
Mistral model implementation for MLX

Implements the Mistral architecture with sliding window attention.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from .base import BaseModelConfig, MLXModelBase, TransformerBlock, RMSNorm, FeedForward

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None


@dataclass
class MistralConfig(BaseModelConfig):
    """Configuration for Mistral models."""
    model_type: str = "mistral"
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    use_bias: bool = False
    
    # Mistral specific
    sliding_window: int = 4096
    attention_dropout: float = 0.0
    
    @classmethod
    def from_huggingface(cls, hf_config: Dict[str, Any]) -> "MistralConfig":
        """Create config from HuggingFace format."""
        return cls(
            vocab_size=hf_config.get("vocab_size", 32000),
            hidden_size=hf_config.get("hidden_size", 4096),
            intermediate_size=hf_config.get("intermediate_size", 14336),
            num_hidden_layers=hf_config.get("num_hidden_layers", 32),
            num_attention_heads=hf_config.get("num_attention_heads", 32),
            num_key_value_heads=hf_config.get("num_key_value_heads", 8),
            max_position_embeddings=hf_config.get("max_position_embeddings", 32768),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-5),
            rope_theta=hf_config.get("rope_theta", 10000.0),
            rope_scaling=hf_config.get("rope_scaling"),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
            use_bias=hf_config.get("use_bias", False),
            sliding_window=hf_config.get("sliding_window", 4096),
            attention_dropout=hf_config.get("attention_dropout", 0.0),
        )


class MistralAttention(nn.Module if MLX_AVAILABLE else object):
    """Mistral attention with sliding window support."""
    
    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        bias: bool = False,
        rope_base: float = 10000.0,
        sliding_window: int = 4096,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        if head_dim is None:
            self.head_dim = dims // num_heads
        else:
            self.head_dim = head_dim
            
        self.scale = self.head_dim ** -0.5
        self.sliding_window = sliding_window
        
        # Linear projections
        self.q_proj = nn.Linear(dims, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dims, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dims, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dims, bias=bias)
        
        # RoPE
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=rope_base)
        
    def __call__(
        self,
        x: "mx.array",
        mask: Optional["mx.array"] = None,
        cache: Optional[Tuple["mx.array", "mx.array"]] = None,
    ) -> Tuple["mx.array", Optional[Tuple["mx.array", "mx.array"]]]:
        B, L, _ = x.shape
        
        # Compute queries, keys, values
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        if cache is not None:
            key_cache, value_cache = cache
            offset = key_cache.shape[2]
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            
            # Apply sliding window to cache
            if key_cache.shape[2] > self.sliding_window:
                key_cache = key_cache[:, :, -self.sliding_window:, :]
                value_cache = value_cache[:, :, -self.sliding_window:, :]
                
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
            
        # Repeat k/v heads if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            keys = mx.repeat(keys, self.num_heads // self.num_kv_heads, axis=1)
            values = mx.repeat(values, self.num_heads // self.num_kv_heads, axis=1)
            
        # Scaled dot-product attention with sliding window
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply sliding window mask
        if mask is None and L > 1:
            mask = self.create_sliding_window_mask(L)
            mask = mask.astype(scores.dtype)
            
        if mask is not None:
            scores = scores + mask
            
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        output = self.o_proj(output)
        
        return output, (keys, values) if cache is not None else None
        
    def create_sliding_window_mask(self, seq_len: int) -> "mx.array":
        """Create sliding window attention mask."""
        mask = mx.full((seq_len, seq_len), -mx.inf)
        for i in range(seq_len):
            start = max(0, i - self.sliding_window + 1)
            mask[i, start:i+1] = 0
        return mask


class MistralBlock(TransformerBlock):
    """Mistral transformer block with sliding window attention."""
    
    def __init__(self, config: MistralConfig):
        # Skip parent init to use custom attention
        nn.Module.__init__(self)
        
        self.attention = MistralAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.use_bias,
            rope_base=config.rope_theta,
            sliding_window=config.sliding_window,
        )
        
        self.feed_forward = FeedForward(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )
        
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class MistralModel(MLXModelBase):
    """Mistral model implementation for MLX."""
    
    def __init__(self, config: MistralConfig):
        # Initialize base components
        nn.Module.__init__(self)
        self.config = config
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers with Mistral blocks
        self.layers = [
            MistralBlock(config)
            for _ in range(config.num_hidden_layers)
        ]
        
        # Output norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Language model head
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False
            )
            
    @staticmethod
    def from_pretrained(
        model_path: str,
        config: Optional[MistralConfig] = None,
    ) -> "MistralModel":
        """Load pretrained Mistral model."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for MistralModel")
            
        import json
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # Load config if not provided
        if config is None:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    hf_config = json.load(f)
                config = MistralConfig.from_huggingface(hf_config)
            else:
                raise FileNotFoundError(f"Config file not found at {config_path}")
                
        # Create model
        model = MistralModel(config)
        
        # Load weights
        weights_path = model_path / "weights.npz"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            model.load_weights(weights)
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")
                
        return model