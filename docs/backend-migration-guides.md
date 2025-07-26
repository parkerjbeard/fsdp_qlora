# Backend Migration Guides

This document provides step-by-step migration guides for moving between different backends in the FSDP QLoRA implementation.

## Table of Contents
- [Migrating to CUDA](#migrating-to-cuda)
- [Migrating to MPS (Apple Silicon)](#migrating-to-mps-apple-silicon)
- [Migrating to MLX](#migrating-to-mlx)
- [Migrating to CPU](#migrating-to-cpu)
- [Cross-Backend Model Migration](#cross-backend-model-migration)

## Migrating to CUDA

### From CPU to CUDA

#### Prerequisites
- NVIDIA GPU with CUDA capability >= 7.0
- CUDA toolkit installed (11.8 or later recommended)
- PyTorch with CUDA support

#### Installation
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CUDA-specific quantization libraries
pip install bitsandbytes
pip install flash-attn --no-build-isolation  # Optional, for Flash Attention
```

#### Code Changes
```python
# Before (CPU)
python train.py \
  --backend cpu \
  --model_name meta-llama/Llama-2-7b-hf \
  --train_type qlora

# After (CUDA)
python train.py \
  --backend cuda \
  --model_name meta-llama/Llama-2-7b-hf \
  --train_type qlora \
  --precision bf16 \
  --use_flash_attention true
```

#### Performance Optimizations
- Enable Flash Attention for longer contexts
- Use BFloat16 precision for better stability
- Enable gradient checkpointing for memory efficiency

### From MPS to CUDA

#### Key Differences
- BFloat16 is supported on CUDA (not on MPS)
- Better quantization support (BitsAndBytes)
- Multi-GPU training available

#### Migration Checklist
1. ✅ Change backend from `mps` to `cuda`
2. ✅ Update precision from `fp16` to `bf16` if desired
3. ✅ Enable CUDA-specific features (Flash Attention, BitsAndBytes)
4. ✅ Adjust batch size (CUDA typically supports larger batches)

## Migrating to MPS (Apple Silicon)

### From CUDA to MPS

#### Prerequisites
- Apple Silicon Mac (M1/M2/M3)
- macOS 12.3 or later
- PyTorch 2.0+ with MPS support

#### Installation
```bash
# Install MPS-enabled PyTorch
pip install torch torchvision torchaudio

# Install MPS-compatible quantization
pip install mlx  # For MLX backend
pip install optimum-quanto  # Alternative quantization
```

#### Code Changes
```python
# Before (CUDA)
python train.py \
  --backend cuda \
  --model_name meta-llama/Llama-2-7b-hf \
  --train_type qlora \
  --precision bf16 \
  --quantization_backend bnb

# After (MPS)
python train.py \
  --backend mps \
  --model_name meta-llama/Llama-2-7b-hf \
  --train_type qlora \
  --precision fp16 \  # BF16 not supported on MPS
  --quantization_backend mlx  # or quanto
```

#### Limitations to Consider
- No BFloat16 support (use Float16)
- No BitsAndBytes (use MLX or Quanto)
- Limited distributed training
- No Flash Attention

### Memory Management on MPS
```python
# MPS-specific memory settings
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
```

## Migrating to MLX

### From PyTorch (CUDA/MPS) to MLX

#### Prerequisites
- Apple Silicon Mac
- Python 3.8+
- MLX installation

#### Installation
```bash
pip install mlx mlx-lm
```

#### Model Conversion
```python
# Convert PyTorch model to MLX format
from mlx_lm import convert

# Convert Hugging Face model
convert(
    "meta-llama/Llama-2-7b-hf",
    "models/llama-2-7b-mlx",
    q_bits=4,  # Optional quantization
)
```

#### Code Changes
```python
# Before (PyTorch)
python train.py \
  --backend cuda \
  --model_name meta-llama/Llama-2-7b-hf \
  --train_type qlora

# After (MLX)
python train.py \
  --backend mlx \
  --model_name models/llama-2-7b-mlx \
  --train_type qlora \
  --mlx_config config.json
```

#### Key Differences
- Models must be in MLX format (.npz files)
- Different optimizer implementations
- No FSDP support
- Native unified memory (no device management)

## Migrating to CPU

### From GPU (CUDA/MPS) to CPU

#### When to Use CPU Backend
- Testing and debugging
- Small model prototyping
- When GPU is unavailable
- CI/CD pipelines

#### Code Changes
```python
# Before (GPU)
python train.py \
  --backend cuda \
  --model_name meta-llama/Llama-2-7b-hf \
  --train_type qlora \
  --batch_size 4

# After (CPU)
python train.py \
  --backend cpu \
  --model_name meta-llama/Llama-2-7b-hf \
  --train_type qlora \
  --batch_size 1 \  # Smaller batch size
  --gradient_accumulation_steps 4 \  # Compensate with accumulation
  --precision fp32  # Full precision often more stable
```

#### Performance Considerations
- Expect 10-100x slower training
- Use smaller models when possible
- Enable CPU-specific optimizations:
```bash
export OMP_NUM_THREADS=8  # Adjust based on CPU cores
export MKL_NUM_THREADS=8
```

## Cross-Backend Model Migration

### Checkpoint Compatibility

#### Saving Backend-Agnostic Checkpoints
```python
# Save in a backend-agnostic format
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'backend': 'cuda',  # Original backend
    'precision': 'bf16',
    'quantization': 'bnb-4bit',
}
torch.save(checkpoint, 'checkpoint.pt')
```

#### Loading on Different Backend
```python
# Load checkpoint
checkpoint = torch.load('checkpoint.pt', map_location='cpu')

# Adapt for target backend
if target_backend == 'mps' and checkpoint['precision'] == 'bf16':
    print("Converting bf16 to fp16 for MPS")
    # Convert weights to fp16

if target_backend == 'mlx':
    print("Note: Direct loading not supported, conversion required")
    # Use MLX conversion tools
```

### Quantization Migration

#### BitsAndBytes to HQQ
```python
# Original (BitsAndBytes on CUDA)
--quantization_backend bnb --n_bits 4

# Migrated (HQQ on CPU/CUDA)
--quantization_backend hqq --n_bits 4 --group_size 128
```

#### BitsAndBytes to MLX
```python
# Requires model re-quantization
from mlx_lm import convert
convert("model_path", "output_path", q_bits=4)
```

### Best Practices for Migration

1. **Test on Small Scale First**
   - Use a tiny model or subset of data
   - Verify functionality before full migration

2. **Benchmark Performance**
   ```python
   # Add profiling to compare backends
   --profile true --profiling_output ./profile_results
   ```

3. **Adjust Hyperparameters**
   - Batch size may need adjustment
   - Learning rate might need tuning
   - Precision settings affect convergence

4. **Handle Backend-Specific Features**
   ```python
   # Example: Feature availability check
   if backend_manager.capabilities.supports_flash_attention:
       config.use_flash_attention = True
   else:
       config.context_length = min(config.context_length, 2048)
   ```

## Troubleshooting Common Migration Issues

### Issue: "Precision not supported"
```python
# Solution: Use backend-appropriate precision
precision_map = {
    'cuda': ['fp32', 'fp16', 'bf16'],
    'mps': ['fp32', 'fp16'],
    'mlx': ['fp32', 'fp16'],
    'cpu': ['fp32']
}
```

### Issue: "Quantization method not available"
```python
# Solution: Use compatible quantization
quantization_map = {
    'cuda': ['bnb', 'hqq', 'quanto'],
    'mps': ['mlx', 'quanto'],
    'mlx': ['mlx'],
    'cpu': ['hqq']
}
```

### Issue: "Out of Memory"
```python
# Solutions by backend:
# CUDA: Enable gradient checkpointing, reduce batch size
# MPS: Set environment variables, use quanto instead of mlx
# CPU: Use sequential loading, reduce model size
```

## Migration Decision Tree

```
Start: What's your hardware?
├── NVIDIA GPU available?
│   └── Yes → Use CUDA backend
│       ├── Large VRAM (>24GB) → Use full precision
│       └── Limited VRAM → Use QLoRA with BitsAndBytes
├── Apple Silicon?
│   └── Yes → MPS or MLX?
│       ├── Need PyTorch ecosystem → Use MPS
│       └── Want best Apple performance → Use MLX
└── CPU only?
    └── Yes → Use CPU backend
        ├── Testing only → Standard settings
        └── Production → Optimize with OMP settings
```

## Summary

Each backend has its strengths:
- **CUDA**: Best overall performance and feature support
- **MPS**: Good for Apple Silicon with PyTorch compatibility
- **MLX**: Optimal for Apple Silicon native performance
- **CPU**: Universal compatibility, good for testing

Choose based on your hardware, performance requirements, and ecosystem needs.