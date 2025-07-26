# Migration Guide: MPS Quantization

## Quick Migration Steps

### 1. Assess Current Implementation

Identify what quantization method you're currently using:

- PyTorch native quantization
- Bitsandbytes
- Manual quantization
- No quantization

### 2. Install Dependencies

```bash
# For Apple Silicon (recommended)
pip install mlx mlx-lm

# For general MPS support
pip install optimum-quanto

# Basic requirements
pip install torch>=2.0
```

### 3. Update Your Code

#### From PyTorch Quantization

**Before:**
```python
import torch.quantization as tq

# Dynamic quantization
model = tq.quantize_dynamic(
    model, 
    {nn.Linear}, 
    dtype=torch.qint8
)
```

**After:**
```python
from unified_quantization import quantize_model

# Automatic backend selection
model, quantizer = quantize_model(
    model,
    bits=8,
    backend="auto"
)
```

#### From Bitsandbytes

**Before:**
```python
import bitsandbytes as bnb

# Replace layers manually
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        # Create 8-bit layer
        layer = bnb.nn.Linear8bitLt(
            module.in_features,
            module.out_features,
            bias=module.bias is not None
        )
```

**After:**
```python
from unified_quantization import quantize_model

# One-line quantization
model, quantizer = quantize_model(
    model,
    bits=4,  # or 8
    backend="auto"
)
```

### 4. Handle Model Loading/Saving

**Before:**
```python
# Standard PyTorch
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

**After:**
```python
# Save quantized model
quantizer.save_model(model, "model_quantized")

# Load quantized model
model = quantizer.load_model("model_quantized")
```

### 5. Fine-tuning Support

**For QLoRA-style fine-tuning:**
```python
from unified_quantization import quantize_model

# Enable LoRA during quantization
model, quantizer = quantize_model(
    model,
    bits=4,
    backend="mlx",  # or "auto"
    enable_lora=True,
    lora_rank=16,
)

# Model is ready for fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

## Backend Selection Guide

### Use MLX when:
- Running on Apple Silicon (M1/M2/M3)
- Need 1-8 bit quantization
- Want QLoRA fine-tuning
- Require best performance

### Use Quanto when:
- Need cross-platform compatibility
- Want calibration-based quantization
- Running on Intel Mac with MPS
- Need INT2/4/8 support

### Use Custom MPS when:
- MLX/Quanto not available
- Need basic PyTorch compatibility
- Want minimal dependencies
- Only need INT4/8

## Common Migration Patterns

### Pattern 1: Research to Production

```python
# Research code (flexible but slow)
if torch.cuda.is_available():
    model = model.cuda()
    # CUDA quantization
elif torch.backends.mps.is_available():
    model = model.to("mps")
    # Manual MPS handling

# Production code (unified and optimized)
from unified_quantization import quantize_model

model, quantizer = quantize_model(
    model,
    bits=4,
    backend="auto"  # Handles device selection
)
```

### Pattern 2: Memory-Constrained Deployment

```python
# Before: OOM errors with large models
model = AutoModel.from_pretrained("large-model")  # OOM\!

# After: Automatic memory management
from unified_quantization import quantize_model, UnifiedQuantizationConfig

config = UnifiedQuantizationConfig(
    backend="mlx",
    bits=4,
    memory_efficient=True,
    chunk_size=256,  # Process in chunks
)

model, quantizer = quantize_model(
    "large-model",
    config=config
)
```

### Pattern 3: Mixed Precision

```python
# Before: All-or-nothing quantization
model = quantize_dynamic(model, {nn.Linear}, torch.qint8)

# After: Fine-grained control
config = UnifiedQuantizationConfig(
    default_bits=4,
    layer_bits={
        "embed": 8,      # Keep embeddings at higher precision
        "lm_head": 8,    # Output layer too
        "transformer.h.0": 6,  # First transformer layer
    }
)

model, quantizer = quantize_model(model, config=config)
```

## Validation Steps

After migration, validate your model:

```python
# 1. Compare outputs
original_output = original_model(input_ids)
quantized_output = quantized_model(input_ids)

# Check similarity
similarity = torch.cosine_similarity(
    original_output.flatten(),
    quantized_output.flatten(),
    dim=0
)
print(f"Output similarity: {similarity.item():.4f}")

# 2. Benchmark performance
from unified_quantization import compare_backends

results = compare_backends(
    model,
    input_shape=(1, 512),
    bits=4
)

# 3. Check memory usage
from backend_manager import BackendManager

manager = BackendManager.get_instance()
info = manager.get_memory_info()
print(f"Memory used: {info['used_gb']:.2f} GB")
```

## Troubleshooting Migration Issues

### Issue: ImportError for MLX

```bash
# Solution
pip install --upgrade mlx mlx-lm
```

### Issue: Model outputs differ significantly

```python
# Solution: Use higher bits or calibration
config = UnifiedQuantizationConfig(
    bits=8,  # Start with 8-bit
    calibration_data=your_data,
    calibration_samples=1000
)
```

### Issue: Performance regression

```python
# Solution: Profile and optimize
results = compare_backends(model, input_shape, bits=4)
# Use the fastest backend from results
```

## Next Steps

1. Read the [full documentation](mps_quantization_guide.md)
2. Try the [example notebooks](../examples/)
3. Report issues on [GitHub](https://github.com/your-repo)
4. Join our [community discussions](https://github.com/your-repo/discussions)
EOF < /dev/null