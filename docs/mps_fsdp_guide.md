# MPS FSDP (Fully Sharded Data Parallel) Guide

This guide provides comprehensive documentation for using FSDP with Apple Silicon's MPS (Metal Performance Shaders) backend.

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [MPS Limitations and Workarounds](#mps-limitations-and-workarounds)
4. [Installation and Setup](#installation-and-setup)
5. [Basic Usage](#basic-usage)
6. [Advanced Configuration](#advanced-configuration)
7. [Memory Optimization](#memory-optimization)
8. [Multi-Process Training](#multi-process-training)
9. [Checkpoint Management](#checkpoint-management)
10. [Integration with Quantization](#integration-with-quantization)
11. [Troubleshooting](#troubleshooting)
12. [Performance Tips](#performance-tips)
13. [API Reference](#api-reference)

## Overview

The MPS FSDP wrapper provides a fully compatible implementation of PyTorch's Fully Sharded Data Parallel training for Apple Silicon. It handles MPS-specific limitations and optimizes for the unified memory architecture of M1/M2/M3 chips.

### Why FSDP on MPS?

- **Memory Efficiency**: Train larger models by sharding parameters across processes
- **Unified Memory Benefits**: No GPU-CPU transfer overhead on Apple Silicon
- **Optimized Sharding**: Strategies tailored for unified memory architecture
- **Automatic Fallbacks**: Handles unsupported operators gracefully

## Key Features

- ✅ **MPS-Optimized**: Tailored for Apple Silicon's unified memory
- ✅ **Dtype Handling**: Automatic bfloat16 → float16 conversion
- ✅ **Gloo Backend**: Uses Gloo for distributed communication (NCCL not supported)
- ✅ **Operator Fallbacks**: Automatic CPU fallback for unsupported ops
- ✅ **Memory Profiling**: Built-in memory tracking and optimization
- ✅ **Checkpoint Support**: Full checkpoint save/load functionality
- ✅ **Quantization Compatible**: Works with HQQ and other MPS-compatible quantization

## MPS Limitations and Workarounds

### Dtype Limitations

MPS doesn't support `bfloat16`. The wrapper automatically converts to `float16`:

```python
# This will automatically use float16 on MPS
wrapper = MPSFSDPWrapper()
model = wrapper.wrap_model(model, param_dtype=torch.bfloat16)  # Converted to float16
```

### Backend Limitations

MPS only supports the Gloo distributed backend:

```python
# NCCL is not supported - will automatically use Gloo
config = MPSFSDPConfig(backend="gloo")  # Must be Gloo
```

### Operator Fallbacks

Some operators may fallback to CPU. Common fallback operators:
- `aten::_fused_adam` - Fused optimizers
- `aten::_foreach_*` - Some batched operations
- Various backward operations

The wrapper handles these automatically with performance warnings.

## Installation and Setup

### Requirements

- PyTorch 2.0+ (2.7+ recommended for best MPS support)
- macOS 13.0+
- Apple Silicon Mac (M1/M2/M3)

### Compatibility Check

```python
from mps_fsdp_wrapper import check_mps_fsdp_compatibility

# Check your system
info = check_mps_fsdp_compatibility()
print(f"MPS Available: {info['mps_available']}")
print(f"PyTorch Version: {info['pytorch_version']}")
print(f"Float16 Support: {info.get('float16_supported', False)}")
print(f"Warnings: {info.get('warnings', [])}")
```

## Basic Usage

### Simple Model Wrapping

```python
import torch
import torch.nn as nn
from mps_fsdp_wrapper import wrap_model_for_mps

# Create your model
model = nn.TransformerModel(...)

# Wrap with FSDP for MPS
fsdp_model = wrap_model_for_mps(
    model,
    min_num_params=1e6,  # Wrap layers with >1M parameters
)

# Move to MPS
fsdp_model = fsdp_model.to("mps")
```

### Custom Configuration

```python
from mps_fsdp_wrapper import MPSFSDPConfig, MPSFSDPWrapper

# Create custom configuration
config = MPSFSDPConfig(
    sharding_strategy="FULL_SHARD",  # or "SHARD_GRAD_OP", "NO_SHARD"
    min_num_params=1e6,
    use_mixed_precision=True,
    cpu_offload=False,
    backward_prefetch="BACKWARD_PRE",
)

# Create wrapper
wrapper = MPSFSDPWrapper(config)

# Wrap model
fsdp_model = wrapper.wrap_model(model)
```

### Transformer Models

For transformer architectures, use automatic layer wrapping:

```python
from mps_fsdp_wrapper import create_mps_fsdp_wrapper

# Create wrapper
wrapper = create_mps_fsdp_wrapper(
    sharding_strategy="FULL_SHARD",
    mixed_precision=True,
)

# Wrap transformer with automatic layer detection
fsdp_model = wrapper.wrap_transformer(
    model,
    transformer_layer_cls=nn.TransformerEncoderLayer,  # Your layer class
)
```

## Advanced Configuration

### Full Configuration Options

```python
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch

config = MPSFSDPConfig(
    # Sharding settings
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    min_num_params=1e6,
    
    # Mixed precision (MPS-compatible)
    use_mixed_precision=True,
    compute_dtype=torch.float16,  # Not bfloat16!
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float16,
    
    # Memory optimization
    cpu_offload=False,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    forward_prefetch=True,
    limit_all_gathers=True,
    
    # Distributed settings
    backend="gloo",  # Required for MPS
    world_size=1,
    rank=0,
    
    # MPS-specific
    sync_module_states=True,
    use_orig_params=True,  # Required for some optimizers
    
    # Unified memory optimization
    unified_memory_pool_size=None,  # Auto-detect
    aggressive_memory_optimization=False,
    
    # Debug settings
    debug_mode=False,
    profile_memory=False,
)
```

### Sharding Strategies

Choose the right strategy based on model size and memory:

```python
# Full sharding - maximum memory savings
config.sharding_strategy = ShardingStrategy.FULL_SHARD

# Shard gradients and optimizer states only
config.sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

# No sharding - for small models
config.sharding_strategy = ShardingStrategy.NO_SHARD
```

## Memory Optimization

### Automatic Strategy Selection

The wrapper can automatically choose the optimal sharding strategy:

```python
wrapper = MPSFSDPWrapper()

# Get optimal strategy based on model size
model_size = sum(p.numel() * p.element_size() for p in model.parameters())
available_memory = 32 * 1024**3  # 32GB

optimal_strategy = wrapper.memory_optimizer.optimize_sharding_strategy(
    model_size, available_memory
)

wrapper.config.sharding_strategy = optimal_strategy
```

### Memory Pool Configuration

Optimize memory allocation for Apple Silicon:

```python
config = MPSFSDPConfig(
    unified_memory_pool_size=16 * 1024**3,  # 16GB pool
    aggressive_memory_optimization=True,  # Enable aggressive cleanup
)

wrapper = MPSFSDPWrapper(config)
wrapper.memory_optimizer.setup_memory_pool()
```

### Memory Profiling

Track memory usage during training:

```python
# Enable memory profiling
wrapper.config.profile_memory = True

# Use profiling context
with wrapper.profile_memory():
    output = fsdp_model(batch)
    loss = criterion(output, target)
    loss.backward()

# Get memory statistics
stats = wrapper.get_memory_stats()
print(f"Allocated: {stats['allocated_gb']:.2f} GB")
print(f"Reserved: {stats['reserved_gb']:.2f} GB")
```

## Multi-Process Training

### Setup Distributed Training

```python
import torch.distributed as dist
from mps_fsdp_wrapper import create_mps_fsdp_wrapper

# Initialize process group (must use Gloo on MPS)
dist.init_process_group(
    backend="gloo",
    init_method="env://",
    world_size=2,
    rank=rank,
)

# Create wrapper with distributed config
wrapper = create_mps_fsdp_wrapper(
    world_size=2,
    rank=rank,
    sharding_strategy="FULL_SHARD",
)

# Wrap and train model
fsdp_model = wrapper.wrap_model(model)
```

### Launch Multiple Processes

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=2 train.py

# Or manually with environment variables
RANK=0 WORLD_SIZE=2 python train.py &
RANK=1 WORLD_SIZE=2 python train.py
```

## Checkpoint Management

### Saving Checkpoints

```python
# Save model and optimizer state
wrapper.save_checkpoint(
    fsdp_model,
    checkpoint_path="checkpoint_epoch_10.pt",
    optimizer=optimizer,
    epoch=10,
    global_step=10000,
    best_loss=0.123,
)
```

### Loading Checkpoints

```python
# Load checkpoint
checkpoint = wrapper.load_checkpoint(
    fsdp_model,
    checkpoint_path="checkpoint_epoch_10.pt",
    optimizer=optimizer,
)

# Access saved metadata
epoch = checkpoint.get("epoch", 0)
global_step = checkpoint.get("global_step", 0)
best_loss = checkpoint.get("best_loss", float('inf'))
```

### Checkpoint Configuration

FSDP uses special handling for checkpoints:

```python
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

# Checkpoint saving is handled internally with:
save_policy = FullStateDictConfig(
    offload_to_cpu=True,  # Save memory
    rank0_only=True,      # Only rank 0 saves
)
```

## Integration with Quantization

### Using with HQQ Quantization

HQQ works well with MPS:

```python
from quantization_wrapper import QuantizationConfig, QuantizationMethod

# Create quantization config
quant_config = QuantizationConfig(
    method=QuantizationMethod.HQQ_4BIT,
    bits=4,
    compute_dtype=torch.float16,  # MPS-compatible
)

# Wrap quantized model with FSDP
wrapper = MPSFSDPWrapper()
fsdp_model = wrapper.wrap_model(
    quantized_model,
    param_dtype=quant_config.compute_dtype,
)
```

### Memory Savings Calculation

```python
# Calculate memory savings with quantization + FSDP
base_memory = model_size_gb
quantized_memory = base_memory * (4/16)  # 4-bit vs 16-bit
sharded_memory = quantized_memory / world_size

print(f"Base model: {base_memory:.1f} GB")
print(f"Quantized: {quantized_memory:.1f} GB")
print(f"Sharded per GPU: {sharded_memory:.1f} GB")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "MPS backend is not available"

```python
# Check MPS availability
import torch
print(f"MPS Built: {torch.backends.mps.is_built()}")
print(f"MPS Available: {torch.backends.mps.is_available()}")

# Solution: Update PyTorch or check macOS version
```

#### 2. Dtype Errors

```python
# Error: "MPS does not support bfloat16"
# Solution: The wrapper handles this automatically, but ensure:
config = MPSFSDPConfig(
    compute_dtype=torch.float16,  # Not bfloat16
)
```

#### 3. Backend Errors

```python
# Error: "NCCL is not supported on MPS"
# Solution: Always use Gloo
config = MPSFSDPConfig(backend="gloo")
```

#### 4. Memory Errors

```python
# Enable CPU offloading for large models
config = MPSFSDPConfig(
    cpu_offload=True,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

### Debug Mode

Enable debug mode for detailed logging:

```python
config = MPSFSDPConfig(debug_mode=True)
wrapper = MPSFSDPWrapper(config)

# This will provide detailed FSDP operation logs
```

## Performance Tips

### 1. Optimal Batch Sizes

For Apple Silicon, use these guidelines:

| Chip | Memory | 7B Model | 13B Model | 70B Model |
|------|--------|----------|-----------|-----------|
| M1 | 16GB | 1-2 | - | - |
| M1 Max | 32-64GB | 4-8 | 2-4 | - |
| M2 Ultra | 128-192GB | 16-32 | 8-16 | 2-4 |

### 2. Gradient Accumulation

Use gradient accumulation to simulate larger batches:

```python
accumulation_steps = 4
optimizer.zero_grad()

for step in range(accumulation_steps):
    output = fsdp_model(batch[step])
    loss = criterion(output, target[step]) / accumulation_steps
    loss.backward()

optimizer.step()
```

### 3. Memory-Efficient Training Loop

```python
# Clear cache periodically
if step % 100 == 0:
    torch.mps.empty_cache()

# Use memory profiling to find leaks
with wrapper.profile_memory():
    # Training step
    pass
```

### 4. Optimal Sharding Configuration

```python
# For different model sizes on M2 Max (64GB)
if model_size_gb < 10:
    strategy = ShardingStrategy.NO_SHARD
elif model_size_gb < 30:
    strategy = ShardingStrategy.SHARD_GRAD_OP
else:
    strategy = ShardingStrategy.FULL_SHARD
```

## API Reference

### MPSFSDPConfig

Main configuration class:

```python
config = MPSFSDPConfig(
    sharding_strategy: ShardingStrategy
    min_num_params: int
    use_mixed_precision: bool
    compute_dtype: torch.dtype
    cpu_offload: bool
    backend: str  # Must be "gloo"
    # ... more options
)
```

### MPSFSDPWrapper

Main wrapper class:

```python
wrapper = MPSFSDPWrapper(config: Optional[MPSFSDPConfig])

# Methods
fsdp_model = wrapper.wrap_model(model, auto_wrap_policy, param_dtype)
fsdp_model = wrapper.wrap_transformer(model, transformer_layer_cls)
wrapper.save_checkpoint(model, path, optimizer, **metadata)
checkpoint = wrapper.load_checkpoint(model, path, optimizer)
stats = wrapper.get_memory_stats()
```

### Convenience Functions

```python
# Quick creation
wrapper = create_mps_fsdp_wrapper(
    world_size=1,
    rank=0,
    sharding_strategy="FULL_SHARD",
    mixed_precision=True,
    cpu_offload=False,
)

# Quick wrapping
fsdp_model = wrap_model_for_mps(
    model,
    min_num_params=1e6,
    transformer_layer_cls=None,
)

# Compatibility check
info = check_mps_fsdp_compatibility()
```

## Example: Complete Training Script

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mps_fsdp_wrapper import create_mps_fsdp_wrapper

def train_with_fsdp_mps(model, train_loader, num_epochs=10):
    # Check compatibility
    info = check_mps_fsdp_compatibility()
    if not info['mps_available']:
        raise RuntimeError("MPS not available!")
    
    # Create FSDP wrapper
    wrapper = create_mps_fsdp_wrapper(
        sharding_strategy="FULL_SHARD",
        mixed_precision=True,
        min_num_params=1e6,
    )
    
    # Wrap model
    fsdp_model = wrapper.wrap_transformer(
        model,
        transformer_layer_cls=nn.TransformerEncoderLayer,
    )
    fsdp_model = fsdp_model.to("mps")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to("mps"), target.to("mps")
            
            # Forward pass with memory profiling
            with wrapper.profile_memory():
                output = fsdp_model(data)
                loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log progress
            if batch_idx % 100 == 0:
                stats = wrapper.get_memory_stats()
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Memory: {stats['allocated_gb']:.2f} GB")
        
        # Save checkpoint
        wrapper.save_checkpoint(
            fsdp_model,
            f"checkpoint_epoch_{epoch}.pt",
            optimizer=optimizer,
            epoch=epoch,
        )
    
    return fsdp_model

# Usage
if __name__ == "__main__":
    model = YourTransformerModel()
    train_loader = DataLoader(...)
    
    trained_model = train_with_fsdp_mps(model, train_loader)
```

## Conclusion

The MPS FSDP wrapper enables efficient training of large models on Apple Silicon by:
- Handling MPS-specific limitations automatically
- Optimizing for unified memory architecture
- Providing memory-efficient sharding strategies
- Ensuring compatibility with quantization methods

For more examples, see the [examples directory](../examples/).