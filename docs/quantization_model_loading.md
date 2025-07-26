# Quantization Model Loading Guide

This guide explains how to load quantized models using the FSDP+QLoRA framework's unified quantization abstraction layer.

## Overview

The framework now supports loading pre-quantized models across all quantization backends:
- **BitsAndBytes** - 4-bit NF4 and 8-bit INT8 quantization
- **HQQ** - Flexible bit-width quantization with grouped parameters  
- **MLX** - Apple Silicon optimized quantization
- **Quanto** - Hugging Face's quantization library

## Basic Usage

```python
from src.core.quantization_wrapper import (
    create_quantization_adapter,
    QuantizationConfig,
    QuantizationMethod,
    Backend
)

# Create configuration
config = QuantizationConfig(
    method=QuantizationMethod.BNB_NF4,  # Choose quantization method
    bits=4,                             # Bit width
    compute_dtype=torch.float16         # Computation dtype
)

# Create adapter
adapter = create_quantization_adapter(Backend.CUDA, config)

# Load quantized model
model = adapter.load_quantized_model(
    model_path="path/to/saved/model",
    model_config=model_config  # AutoConfig or dict
)
```

## BitsAndBytes Model Loading

### Loading from Directory

BitsAndBytes models saved in a directory structure:
```
model_dir/
├── pytorch_model.bin         # Model weights
├── quantization_config.json  # Quantization settings
└── config.json              # Model architecture config
```

```python
# 4-bit NF4 quantization
config = QuantizationConfig(
    method=QuantizationMethod.BNB_NF4,
    bits=4,
    compute_dtype=torch.float16,
    quant_type="nf4"  # or "fp4"
)

adapter = create_quantization_adapter(Backend.CUDA, config)
model = adapter.load_quantized_model("model_dir/", model_config)
```

### Loading from Single File

```python
# 8-bit quantization
config = QuantizationConfig(
    method=QuantizationMethod.BNB_INT8,
    bits=8
)

adapter = create_quantization_adapter(Backend.CUDA, config)
model = adapter.load_quantized_model("model.pth", model_config)
```

### Quantization Config Format

`quantization_config.json`:
```json
{
    "bits": 4,
    "quant_type": "nf4",
    "compute_dtype": "float16"
}
```

## HQQ Model Loading

### Native HQQ Format

HQQ models can be saved in their native format:
```python
config = QuantizationConfig(
    method=QuantizationMethod.HQQ,
    bits=4,
    group_size=64,
    quant_zero=True,
    quant_scale=False
)

adapter = create_quantization_adapter(Backend.CUDA, config)

# Load native HQQ model
model = adapter.load_quantized_model("hqq_model.pt", model_config)
```

### HQQ State Dict Format

HQQ quantized weights are stored with special keys:
```
model_dir/
├── pytorch_model.bin  # Contains W_q, meta, bias
└── hqq_config.json   # HQQ-specific settings
```

State dict structure:
- `layer.W_q` - Quantized weights
- `layer.meta` - Quantization metadata (scale, zero point)
- `layer.bias` - Bias (if present)

### HQQ Config Format

`hqq_config.json`:
```json
{
    "weight_quant_params": {
        "nbits": 4,
        "group_size": 64,
        "quant_zero": true,
        "quant_scale": false
    },
    "scale_quant_params": {
        "nbits": 8,
        "group_size": 128
    }
}
```

## MLX Model Loading

### Loading MLX NPZ Format

MLX models use NumPy's NPZ format:
```python
config = QuantizationConfig(
    method=QuantizationMethod.MLX_INT4,
    bits=4,
    group_size=32
)

# Use MLX backend on Apple Silicon
adapter = create_quantization_adapter(Backend.MLX, config)

# Load from NPZ
model = adapter.load_quantized_model("model.npz", model_config)
```

### Converting PyTorch to MLX

The adapter can automatically convert PyTorch checkpoints:
```python
# Load and convert PyTorch model
model = adapter.load_quantized_model("pytorch_model.bin", model_config)
```

### MLX Directory Structure
```
model_dir/
├── model.safetensors  # or mlx_model.npz
├── mlx_config.json    # MLX quantization config
└── config.json        # Model architecture
```

## Quanto Model Loading

### Loading Quanto Models

```python
config = QuantizationConfig(
    method=QuantizationMethod.QUANTO_INT4,
    bits=4,
    group_size=128
)

adapter = create_quantization_adapter(Backend.MPS, config)

# Load from directory
model = adapter.load_quantized_model("quanto_model_dir/", model_config)
```

### Quanto Bit Width Options
- `QUANTO_INT2` - 2-bit quantization
- `QUANTO_INT4` - 4-bit quantization  
- `QUANTO_INT8` - 8-bit quantization

### Single File with Embedded Config

```python
# Save format
checkpoint = {
    'state_dict': model.state_dict(),
    'quanto_config': {
        'bits': 4,
        'group_size': 128
    }
}
torch.save(checkpoint, 'quanto_model.pth')

# Load
model = adapter.load_quantized_model('quanto_model.pth', model_config)
```

## Backend Compatibility

| Quantization Method | CUDA | MPS | MLX | CPU |
|--------------------|------|-----|-----|-----|
| BitsAndBytes NF4   | ✓    | ✗   | ✗   | ✗   |
| BitsAndBytes INT8  | ✓    | ✗   | ✗   | ✗   |
| HQQ               | ✓    | ✓   | ✗   | ✓   |
| MLX INT4/INT8     | ✗    | ✓   | ✓   | ✗   |
| Quanto INT2/4/8   | ✓    | ✓   | ✗   | ✓   |

## Advanced Features

### Custom Model Configurations

You can pass model configurations in multiple formats:

```python
# Using transformers AutoConfig
from transformers import AutoConfig
model_config = AutoConfig.from_pretrained("model_name")

# Using dictionary
model_config = {
    'model_type': 'llama',
    'hidden_size': 4096,
    'num_hidden_layers': 32,
    'num_attention_heads': 32
}

# Load with config
model = adapter.load_quantized_model(model_path, model_config)
```

### Handling Missing Dependencies

The adapters gracefully handle missing dependencies:

```python
try:
    model = adapter.load_quantized_model(model_path, model_config)
except ImportError as e:
    print(f"Required dependency not installed: {e}")
    # Falls back to alternative implementation if available
```

### Memory-Efficient Loading

When using with `transformers`, models are loaded with empty weights first:

```python
from accelerate import init_empty_weights

with init_empty_weights():
    # Model skeleton created without weights
    model = AutoModelForCausalLM.from_config(config)

# Quantized weights loaded afterward
# This reduces peak memory usage
```

## Common Issues and Solutions

### FileNotFoundError

Ensure the model path exists and contains the expected files:
```python
from pathlib import Path

model_path = Path("path/to/model")
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")
```

### Quantization Config Mismatch

If the saved quantization config doesn't match the adapter config, the saved config takes precedence:
```python
# Adapter config (bits=4)
config = QuantizationConfig(method=QuantizationMethod.BNB_NF4, bits=4)

# If saved model has bits=8, it will load as 8-bit
```

### Backend Incompatibility

If a quantization method isn't supported on your backend:
```python
# Will automatically use FallbackAdapter
adapter = create_quantization_adapter(Backend.CPU, config)
# Returns unquantized model with warning
```

## Examples

### Complete Example: Loading and Using a Quantized Model

```python
import torch
from transformers import AutoTokenizer
from src.core.quantization_wrapper import (
    create_quantization_adapter,
    QuantizationConfig,
    QuantizationMethod,
    Backend
)

# 1. Setup configuration
config = QuantizationConfig(
    method=QuantizationMethod.BNB_NF4,
    bits=4,
    compute_dtype=torch.float16
)

# 2. Create adapter
adapter = create_quantization_adapter(Backend.CUDA, config)

# 3. Load model
model_config = {
    'model_type': 'llama',
    'vocab_size': 32000,
    'hidden_size': 4096
}
model = adapter.load_quantized_model("path/to/model", model_config)

# 4. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# 5. Use model
model.eval()
with torch.no_grad():
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = model(**inputs)
```

### Saving and Loading Round-Trip

```python
# Save quantized model
adapter.save_quantized_model(model, "saved_model/")

# Load it back
loaded_model = adapter.load_quantized_model("saved_model/", model_config)
```

## Best Practices

1. **Match Quantization Methods**: Use the same quantization method for saving and loading
2. **Preserve Configs**: Always save quantization configs with your models
3. **Check Backend Support**: Verify your backend supports the quantization method
4. **Handle Errors Gracefully**: Wrap loading in try-except for production code
5. **Validate Loaded Models**: Test loaded models match expected behavior

## Future Enhancements

- Automatic format detection
- Cross-format conversion utilities
- Streaming model loading for very large models
- Integration with model hubs