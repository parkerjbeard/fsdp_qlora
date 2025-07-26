"""
Qwen model implementation for MLX

Implements Alibaba's Qwen architecture for MLX framework.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

from .base import BaseModelConfig, RMSNorm

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
class QwenConfig(BaseModelConfig):
    """Configuration for Qwen models."""
    model_type: str = "qwen"
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = 32
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    use_bias: bool = True
    
    # Qwen specific
    use_sliding_window: bool = False
    sliding_window_size: int = 32768
    use_flash_attn: bool = True
    attention_dropout: float = 0.0
    use_cache_quantization: bool = False
    use_cache_kernel: bool = False
    softmax_in_fp32: bool = True
    
    @classmethod
    def from_huggingface(cls, hf_config: Dict[str, Any]) -> "QwenConfig":
        """Create config from HuggingFace format."""
        # Handle different Qwen versions
        model_type = hf_config.get("model_type", "qwen")
        
        # Extract RoPE configuration
        rope_scaling = hf_config.get("rope_scaling")
        if rope_scaling is None and "rope" in hf_config:
            # Qwen2 format
            rope_config = hf_config["rope"]
            rope_theta = rope_config.get("base", 10000.0)
            if rope_config.get("scaling_factor"):
                rope_scaling = {
                    "type": "linear",
                    "factor": rope_config["scaling_factor"]
                }
        else:
            rope_theta = hf_config.get("rope_theta", 10000.0)
            
        return cls(
            model_type=model_type,
            vocab_size=hf_config.get("vocab_size", 151936),
            hidden_size=hf_config.get("hidden_size", 4096),
            intermediate_size=hf_config.get("intermediate_size", 22016),
            num_hidden_layers=hf_config.get("num_hidden_layers", 32),
            num_attention_heads=hf_config.get("num_attention_heads", 32),
            num_key_value_heads=hf_config.get("num_key_value_heads"),
            max_position_embeddings=hf_config.get("max_position_embeddings", 32768),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
            use_bias=hf_config.get("use_bias", True),
            use_sliding_window=hf_config.get("use_sliding_window", False),
            sliding_window_size=hf_config.get("sliding_window_size", 32768),
            use_flash_attn=hf_config.get("use_flash_attn", True),
            attention_dropout=hf_config.get("attention_dropout", 0.0),
            use_cache_quantization=hf_config.get("use_cache_quantization", False),
            use_cache_kernel=hf_config.get("use_cache_kernel", False),
            softmax_in_fp32=hf_config.get("softmax_in_fp32", True),
        )


class QwenAttention(nn.Module if MLX_AVAILABLE else object):
    """Qwen attention implementation."""
    
    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = True,
        rope_base: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        use_sliding_window: bool = False,
        sliding_window_size: int = 32768,
        softmax_in_fp32: bool = True,
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
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.softmax_in_fp32 = softmax_in_fp32
        
        # Qwen uses concatenated QKV projection
        self.qkv_proj = nn.Linear(
            dims,
            (num_heads + 2 * num_kv_heads) * self.head_dim,
            bias=bias
        )
        self.o_proj = nn.Linear(num_heads * self.head_dim, dims, bias=False)
        
        # RoPE with potential scaling
        rope_scale = 1.0
        if rope_scaling:
            if rope_scaling.get("type") == "linear":
                rope_scale = rope_scaling.get("factor", 1.0)
            elif rope_scaling.get("type") == "dynamic":
                # Dynamic scaling for extended context
                rope_scale = rope_scaling.get("factor", 1.0)
                
        self.rope = nn.RoPE(
            self.head_dim,
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
        
        # Compute QKV in one projection
        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V
        splits = [
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        ]
        queries, keys, values = mx.split(qkv, splits[:-1], axis=-1)
        
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
            
            # Apply sliding window if enabled
            if self.use_sliding_window and key_cache.shape[2] > self.sliding_window_size:
                key_cache = key_cache[:, :, -self.sliding_window_size:, :]
                value_cache = value_cache[:, :, -self.sliding_window_size:, :]
                
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
            
        # Repeat k/v heads if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            keys = mx.repeat(keys, repeat_factor, axis=1)
            values = mx.repeat(values, repeat_factor, axis=1)
            
        # Scaled dot-product attention
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores + mask
        elif self.use_sliding_window and L > 1:
            # Create sliding window mask
            mask = self.create_sliding_window_mask(L)
            scores = scores + mask
            
        # Softmax (optionally in fp32 for stability)
        if self.softmax_in_fp32:
            scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(queries.dtype)
        else:
            scores = mx.softmax(scores, axis=-1)
            
        # Attention output
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)
        
        return output, (keys, values) if cache is not None else None
        
    def create_sliding_window_mask(self, seq_len: int) -> "mx.array":
        """Create sliding window attention mask."""
        mask = mx.full((seq_len, seq_len), -mx.inf)
        for i in range(seq_len):
            start = max(0, i - self.sliding_window_size + 1)
            mask[i, start:i+1] = 0
        return mask


class QwenMLP(nn.Module if MLX_AVAILABLE else object):
    """Qwen MLP with gated activation."""
    
    def __init__(
        self,
        dims: int,
        hidden_dims: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = dims * 4
            
        # Qwen uses split gate and up projections
        self.w1 = nn.Linear(dims, hidden_dims, bias=bias)
        self.w2 = nn.Linear(dims, hidden_dims, bias=bias)
        self.c_proj = nn.Linear(hidden_dims, dims, bias=bias)
        
    def __call__(self, x: "mx.array") -> "mx.array":
        return self.c_proj(nn.silu(self.w1(x)) * self.w2(x))


class QwenBlock(nn.Module if MLX_AVAILABLE else object):
    """Qwen transformer block."""
    
    def __init__(self, config: QwenConfig):
        super().__init__()
        
        self.attention = QwenAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.use_bias,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
            use_sliding_window=config.use_sliding_window,
            sliding_window_size=config.sliding_window_size,
            softmax_in_fp32=config.softmax_in_fp32,
        )
        
        self.mlp = QwenMLP(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )
        
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
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


class QwenModel(nn.Module if MLX_AVAILABLE else object):
    """Qwen model implementation for MLX."""
    
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = [
            QwenBlock(config)
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
        config: Optional[QwenConfig] = None,
        trust_remote_code: bool = True,
    ) -> "QwenModel":
        """Load pretrained Qwen model."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for QwenModel")
            
        import json
        from pathlib import Path
        
        model_path = Path(model_path)
        
        # Load config if not provided
        if config is None:
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    hf_config = json.load(f)
                config = QwenConfig.from_huggingface(hf_config)
            else:
                raise FileNotFoundError(f"Config file not found at {config_path}")
                
        # Create model
        model = QwenModel(config)
        
        # Load weights
        weights_path = model_path / "weights.npz"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            model.load_weights(weights)
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")
                
        return model
        
    def sanitize(self) -> None:
        """Sanitize model weights for optimization."""
        # Qwen-specific optimizations
        pass