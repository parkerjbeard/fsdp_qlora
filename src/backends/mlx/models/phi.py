"""
Phi model implementation for MLX

Implements Microsoft's Phi architecture for MLX framework.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

from .base import BaseModelConfig

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
class PhiConfig(BaseModelConfig):
    """Configuration for Phi models."""
    model_type: str = "phi"
    vocab_size: int = 51200
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = 32
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    use_bias: bool = True
    
    # Phi specific
    partial_rotary_factor: float = 0.4
    qk_layernorm: bool = False
    use_flash_attn: bool = True
    
    @classmethod
    def from_huggingface(cls, hf_config: Dict[str, Any]) -> "PhiConfig":
        """Create config from HuggingFace format."""
        # Handle Phi-2 and Phi-3 differences
        model_type = hf_config.get("model_type", "phi")
        
        if "phi3" in model_type or "Phi-3" in hf_config.get("_name_or_path", ""):
            # Phi-3 configuration
            return cls(
                model_type="phi3",
                vocab_size=hf_config.get("vocab_size", 32064),
                hidden_size=hf_config.get("hidden_size", 3072),
                intermediate_size=hf_config.get("intermediate_size", 8192),
                num_hidden_layers=hf_config.get("num_hidden_layers", 32),
                num_attention_heads=hf_config.get("num_attention_heads", 32),
                num_key_value_heads=hf_config.get("num_key_value_heads", 32),
                max_position_embeddings=hf_config.get("max_position_embeddings", 4096),
                rms_norm_eps=hf_config.get("rms_norm_eps", 1e-5),
                rope_theta=hf_config.get("rope_theta", 10000.0),
                rope_scaling=hf_config.get("rope_scaling"),
                tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
                use_bias=hf_config.get("use_bias", False),
                partial_rotary_factor=hf_config.get("partial_rotary_factor", 0.4),
                qk_layernorm=hf_config.get("qk_layernorm", False),
            )
        else:
            # Phi-2 configuration
            return cls(
                model_type="phi2",
                vocab_size=hf_config.get("vocab_size", 51200),
                hidden_size=hf_config.get("hidden_size", 2560),
                intermediate_size=hf_config.get("intermediate_size", 10240),
                num_hidden_layers=hf_config.get("num_hidden_layers", 32),
                num_attention_heads=hf_config.get("num_attention_heads", 32),
                num_key_value_heads=hf_config.get("num_key_value_heads"),
                max_position_embeddings=hf_config.get("max_position_embeddings", 2048),
                layer_norm_eps=hf_config.get("layer_norm_eps", 1e-5),
                rope_theta=hf_config.get("rope_theta", 10000.0),
                rope_scaling=hf_config.get("rope_scaling"),
                tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
                use_bias=hf_config.get("use_bias", True),
                partial_rotary_factor=hf_config.get("partial_rotary_factor", 0.4),
                qk_layernorm=hf_config.get("qk_layernorm", False),
            )


class LayerNorm(nn.Module if MLX_AVAILABLE else object):
    """Layer normalization for Phi models."""
    
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))
        self.eps = eps
        
    def __call__(self, x: "mx.array") -> "mx.array":
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        return self.weight * x + self.bias


class PhiAttention(nn.Module if MLX_AVAILABLE else object):
    """Phi attention with partial rotary embeddings."""
    
    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = True,
        rope_base: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        partial_rotary_factor: float = 0.4,
        qk_layernorm: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        if num_kv_heads is None:
            num_kv_heads = num_heads
            
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        if head_dim is None:
            self.head_dim = dims // num_heads
        else:
            self.head_dim = head_dim
            
        self.scale = self.head_dim ** -0.5
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        
        # Linear projections
        self.q_proj = nn.Linear(dims, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dims, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dims, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dims, bias=bias)
        
        # Optional QK LayerNorm
        if qk_layernorm:
            self.q_layernorm = LayerNorm(self.head_dim, eps=layer_norm_eps)
            self.k_layernorm = LayerNorm(self.head_dim, eps=layer_norm_eps)
        else:
            self.q_layernorm = None
            self.k_layernorm = None
        
        # RoPE for partial dimensions
        rope_scale = 1.0
        if rope_scaling:
            if rope_scaling.get("type") == "linear":
                rope_scale = rope_scaling.get("factor", 1.0)
                
        self.rope = nn.RoPE(
            self.rotary_dim,
            traditional=False,
            base=rope_base,
            scale=rope_scale
        )
        
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
        
        # Apply QK LayerNorm if enabled
        if self.q_layernorm is not None:
            queries = self.q_layernorm(queries)
        if self.k_layernorm is not None:
            keys = self.k_layernorm(keys)
        
        # Split for partial rotary
        queries_rot = queries[..., :self.rotary_dim]
        queries_pass = queries[..., self.rotary_dim:]
        keys_rot = keys[..., :self.rotary_dim]
        keys_pass = keys[..., self.rotary_dim:]
        
        # Apply RoPE to partial dimensions
        if cache is not None:
            key_cache, value_cache = cache
            offset = key_cache.shape[2]
            queries_rot = self.rope(queries_rot, offset=offset)
            keys_rot = self.rope(keys_rot, offset=offset)
            
            # Concatenate rotary and pass-through
            queries = mx.concatenate([queries_rot, queries_pass], axis=-1)
            keys = mx.concatenate([keys_rot, keys_pass], axis=-1)
            
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries_rot = self.rope(queries_rot)
            keys_rot = self.rope(keys_rot)
            
            # Concatenate rotary and pass-through
            queries = mx.concatenate([queries_rot, queries_pass], axis=-1)
            keys = mx.concatenate([keys_rot, keys_pass], axis=-1)
            
        # Repeat k/v heads if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            keys = mx.repeat(keys, self.num_heads // self.num_kv_heads, axis=1)
            values = mx.repeat(values, self.num_heads // self.num_kv_heads, axis=1)
            
        # Scaled dot-product attention
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
            
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        output = self.o_proj(output)
        
        return output, (keys, values) if cache is not None else None


class PhiMLP(nn.Module if MLX_AVAILABLE else object):
    """Phi MLP layer."""
    
    def __init__(
        self,
        dims: int,
        hidden_dims: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = dims * 4
            
        self.fc1 = nn.Linear(dims, hidden_dims, bias=bias)
        self.fc2 = nn.Linear(hidden_dims, dims, bias=bias)
        self.act = nn.GELU()
        
    def __call__(self, x: "mx.array") -> "mx.array":
        return self.fc2(self.act(self.fc1(x)))


class PhiBlock(nn.Module if MLX_AVAILABLE else object):
    """Phi transformer block."""
    
    def __init__(self, config: PhiConfig):
        super().__init__()
        
        self.attention = PhiAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.use_bias,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
            partial_rotary_factor=config.partial_rotary_factor,
            qk_layernorm=config.qk_layernorm,
            layer_norm_eps=config.layer_norm_eps,
        )
        
        self.mlp = PhiMLP(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )
        
        # Use LayerNorm for Phi instead of RMSNorm
        self.attention_norm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.mlp_norm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        
    def __call__(
        self,
        x: "mx.array",
        mask: Optional["mx.array"] = None,
        cache: Optional[Tuple["mx.array", "mx.array"]] = None,
    ) -> Tuple["mx.array", Optional[Tuple["mx.array", "mx.array"]]]:
        # Self-attention with residual
        r = x
        x = self.attention_norm(x)
        x, cache = self.attention(x, mask, cache)
        x = r + x
        
        # MLP with residual
        r = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = r + x
        
        return x, cache


class PhiModel(nn.Module if MLX_AVAILABLE else object):
    """Phi model implementation for MLX."""
    
    def __init__(self, config: PhiConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = [
            PhiBlock(config)
            for _ in range(config.num_hidden_layers)
        ]
        
        # Output norm - use LayerNorm for Phi
        self.norm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        
        # Language model head
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=config.use_bias
            )
            
    def __call__(
        self,
        input_ids: "mx.array",
        cache: Optional[List[Tuple["mx.array", "mx.array"]]] = None,
    ) -> Tuple["mx.array", Optional[List[Tuple["mx.array", "mx.array"]]]]:
        # Token embeddings
        x = self.tok_embeddings(input_ids)
        
        # Create causal mask
        mask = None
        T = x.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(x.dtype)
            
        # Pass through transformer layers
        if cache is None:
            cache = [None] * len(self.layers)
            
        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            x, c = layer(x, mask, c)
            if c is not None:
                cache[i] = c
                
        # Output normalization
        x = self.norm(x)
        
        # Language model head
        if self.lm_head is None:
            # Tied embeddings
            logits = x @ self.tok_embeddings.weight.T
        else:
            logits = self.lm_head(x)
            
        return logits, cache
        
    @staticmethod
    def from_pretrained(
        model_path: str,
        config: Optional[PhiConfig] = None,
    ) -> "PhiModel":
        """Load pretrained Phi model."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for PhiModel")
            
        import json
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # Load config if not provided
        if config is None:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    hf_config = json.load(f)
                config = PhiConfig.from_huggingface(hf_config)
            else:
                raise FileNotFoundError(f"Config file not found at {config_path}")
                
        # Create model
        model = PhiModel(config)
        
        # Load weights
        weights_path = model_path / "weights.npz"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            model.load_weights(weights)
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")
                
        return model