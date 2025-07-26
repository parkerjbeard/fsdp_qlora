# MPS Quantization Guide for Apple Silicon

This guide provides comprehensive documentation for the advanced MPS quantization implementation, including MLX integration, Quanto support, and the unified quantization API.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Backend Selection](#backend-selection)
4. [MLX Quantization](#mlx-quantization)
5. [Quanto Integration](#quanto-integration)
6. [Custom MPS Implementation](#custom-mps-implementation)
7. [Unified API](#unified-api)
8. [Migration Guide](#migration-guide)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## Overview

### Background

PyTorch's native quantization support on MPS (Metal Performance Shaders) is limited. Key limitations include:

- `quantize_per_tensor` not implemented for MPS backend
- No support for INT4/INT2 quantization
- Limited dynamic quantization capabilities
- Missing QLoRA-style fine-tuning support

### Our Solution

We provide three complementary solutions:

1. **MLX Backend**: Native Apple Silicon quantization with 1-8 bit support
2. **Quanto Backend**: HuggingFace's device-agnostic quantization with MPS support
3. **Custom MPS Backend**: Fallback implementation for basic PyTorch compatibility

## Quick Start

### Installation

```bash
# Basic requirements
pip install torch torchvision

# For MLX support (recommended for Apple Silicon)
pip install mlx mlx-lm

# For Quanto support
pip install optimum-quanto

# Install the package
pip install -e .
```

### Basic Usage

```python
from unified_quantization import quantize_model

# Automatic backend selection
model, quantizer = quantize_model(
    "meta-llama/Llama-2-7b-hf",
    bits=4,
    backend="auto",  # Automatically selects best backend
)

# Use the quantized model
output = model(input_ids)
```

## Backend Selection

### Automatic Selection

The unified API automatically selects the best backend based on:

1. **Hardware**: Apple Silicon (M1/M2/M3) vs Intel Mac
2. **Available packages**: MLX, Quanto, PyTorch
3. **Model requirements**: Size, quantization bits, training needs

```python
from unified_quantization import UnifiedQuantizationConfig, UnifiedQuantizer

# Auto-select backend
config = UnifiedQuantizationConfig(
    backend="auto",
    bits=4,
)

quantizer = UnifiedQuantizer(config)
```

### Manual Selection

```python
from unified_quantization import QuantizationBackend

# Force specific backend
config = UnifiedQuantizationConfig(
    backend=QuantizationBackend.MLX,  # or QUANTO, MPS_CUSTOM
    bits=4,
)
```

## MLX Quantization

### Features

- **Native Metal acceleration**: Optimized for Apple Silicon
- **1-8 bit quantization**: Including INT2, INT4, INT8
- **Group-wise quantization**: Better accuracy with configurable group sizes
- **QLoRA support**: Fine-tune quantized models efficiently
- **Unified memory**: Seamless CPU-GPU memory sharing

### Basic Usage

```python
from mlx_quantization import create_mlx_quantized_model, MLXQuantizationConfig

# Configure MLX quantization
config = MLXQuantizationConfig(
    bits=4,
    group_size=64,
    embedding_bits=8,  # Higher precision for embeddings
    output_bits=8,     # Higher precision for output layer
)

# Load and quantize
model, tokenizer = create_mlx_quantized_model(
    "meta-llama/Llama-2-7b-hf",
    config=config,
)
```

### Mixed Precision

```python
config = MLXQuantizationConfig(
    default_bits=4,
    layer_bits={
        "transformer.h.0": 8,    # First layer with higher precision
        "transformer.h.31": 2,   # Last layer with aggressive quantization
    },
    skip_modules=["lm_head"],    # Skip quantization for specific modules
)
```

### QLoRA Fine-tuning

```python
from mlx_quantization import fine_tune_quantized_model

# Configure LoRA
config = MLXQuantizationConfig(
    bits=4,
    lora_rank=16,
    lora_alpha=16.0,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Fine-tune
model = fine_tune_quantized_model(
    model,
    train_data,
    config,
    learning_rate=1e-5,
    num_epochs=3,
)
```

## Performance Optimization

### Memory Management

```python
# Configure for large models
config = UnifiedQuantizationConfig(
    backend="mlx",  # Best memory efficiency on Apple Silicon
    bits=4,
    memory_efficient=True,
    chunk_size=512,  # Process layers in chunks
)

# Monitor memory usage
from backend_manager import BackendManager
manager = BackendManager.get_instance()
memory_info = manager.get_memory_info()
print(f"Available: {memory_info['available_gb']:.1f}GB")
```

## Troubleshooting

### Common Issues

#### 1. MLX Import Error

```bash
# Error: ImportError: MLX not available
# Solution:
pip install mlx mlx-lm
```

#### 2. MPS Out of Memory

```python
# Solution: Use memory-efficient mode
config = UnifiedQuantizationConfig(
    memory_efficient=True,
    chunk_size=256,  # Smaller chunks
    bits=4,  # Lower bits
)
```

## License

This project is licensed under the MIT License.
EOF < /dev/null