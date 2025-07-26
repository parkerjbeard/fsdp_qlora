"""
Base classes for MLX models

Provides common components and utilities used across different model architectures.
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    import warnings
    warnings.warn("MLX not available. MLX models will not work.")
    # Create dummy classes for type hints
    class mx:
        array = Any
    class nn:
        Module = object
        RoPE = object
        
        
@dataclass
class BaseModelConfig:
    """Base configuration for MLX models."""
    model_type: str = "base"
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
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class RMSNorm(nn.Module if MLX_AVAILABLE else object):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        
    def __call__(self, x: "mx.array") -> "mx.array":
        return mx.fast.rms_norm(x, self.weight, self.eps)


class MLXAttention(nn.Module if MLX_AVAILABLE else object):
    """Multi-head attention implementation for MLX."""
    
    def __init__(
        self,
        dims: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = False,
        rope_base: float = 10000.0,
        rope_traditional: bool = False,
        rope_scaling: Optional[Dict[str, Any]] = None,
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
        
        # Linear projections
        self.q_proj = nn.Linear(dims, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dims, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dims, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dims, bias=bias)
        
        # RoPE
        rope_scale = 1.0
        if rope_scaling:
            if rope_scaling.get("type") == "linear":
                rope_scale = rope_scaling.get("factor", 1.0)
                
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=rope_traditional,
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
        
        # Apply RoPE
        if cache is not None:
            key_cache, value_cache = cache
            offset = key_cache.shape[2]
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
            
        # Repeat k/v heads if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            key_shape = keys.shape
            keys = mx.tile(
                keys[:, :, None, :, :],
                (1, 1, self.num_heads // self.num_kv_heads, 1, 1)
            )
            keys = keys.reshape(
                key_shape[0], 
                self.num_heads,
                key_shape[2],
                key_shape[3]
            )
            
            value_shape = values.shape
            values = mx.tile(
                values[:, :, None, :, :],
                (1, 1, self.num_heads // self.num_kv_heads, 1, 1)
            )
            values = values.reshape(
                value_shape[0],
                self.num_heads,
                value_shape[2],
                value_shape[3]
            )
            
        # Scaled dot-product attention
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            scores = scores + mask
            
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        output = self.o_proj(output)
        
        return output, (keys, values) if cache is not None else None


class FeedForward(nn.Module if MLX_AVAILABLE else object):
    """Feed-forward network."""
    
    def __init__(
        self,
        dims: int,
        hidden_dims: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = dims * 4
            
        self.gate_proj = nn.Linear(dims, hidden_dims, bias=bias)
        self.up_proj = nn.Linear(dims, hidden_dims, bias=bias)
        self.down_proj = nn.Linear(hidden_dims, dims, bias=bias)
        
    def __call__(self, x: "mx.array") -> "mx.array":
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module if MLX_AVAILABLE else object):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        
        self.attention = MLXAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=config.use_bias,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        
        self.feed_forward = FeedForward(
            config.hidden_size,
            config.intermediate_size,
            bias=config.use_bias,
        )
        
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
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
        
        # Feed-forward with residual
        r = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = r + x
        
        return x, cache


class MLXModelBase(nn.Module if MLX_AVAILABLE else object):
    """Base class for MLX transformer models."""
    
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = [
            TransformerBlock(config)
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
        
    def generate(
        self,
        prompt_tokens: "mx.array",
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> "mx.array":
        """Generate tokens autoregressively."""
        
        def sample(logits: "mx.array") -> "mx.array":
            if temperature == 0:
                return mx.argmax(logits, axis=-1)
            else:
                # Apply temperature
                logits = logits / temperature
                
                # Top-p sampling
                if top_p < 1.0:
                    # Sort and compute cumulative probabilities
                    probs = mx.softmax(logits, axis=-1)
                    sorted_indices = mx.argsort(probs, axis=-1)
                    sorted_probs = mx.take_along_axis(
                        probs, sorted_indices, axis=-1
                    )
                    cumsum = mx.cumsum(sorted_probs, axis=-1)
                    
                    # Find cutoff
                    cutoff_mask = cumsum < (1 - top_p)
                    cutoff_mask[:, -1] = True
                    
                    # Zero out tokens below cutoff
                    sorted_probs = mx.where(cutoff_mask, 0, sorted_probs)
                    
                    # Renormalize and sample
                    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)
                    
                    # Sample from filtered distribution
                    sampled = mx.random.categorical(mx.log(sorted_probs))
                    
                    # Map back to vocabulary indices
                    return mx.take_along_axis(
                        sorted_indices,
                        sampled[:, None],
                        axis=-1
                    ).squeeze(-1)
                else:
                    # Simple sampling
                    return mx.random.categorical(logits)
                    
        # Initialize
        tokens = prompt_tokens
        cache = None
        
        # Generate tokens
        for _ in range(max_tokens):
            # Forward pass
            logits, cache = self(tokens[:, -1:], cache)
            logits = logits[:, -1, :]
            
            # Sample next token
            next_token = sample(logits)
            
            # Append to sequence
            tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)
            
        return tokens