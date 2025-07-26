# Known Limitations and Unimplemented Features

This document outlines the current limitations and unimplemented features in the FSDP QLoRA codebase.

## Table of Contents
- [Backend-Specific Limitations](#backend-specific-limitations)
- [Quantization Limitations](#quantization-limitations)
- [Model Loading Limitations](#model-loading-limitations)
- [Training Limitations](#training-limitations)
- [Feature Status](#feature-status)

## Backend-Specific Limitations

### CUDA Backend
- **Availability**: Requires NVIDIA GPU with compatible CUDA drivers
- **Tested GPUs**: A100, V100, RTX 3090/4090 series
- **Memory**: Minimum 16GB VRAM recommended for 7B models with QLoRA

### MPS Backend (Apple Silicon)
- **BFloat16 Support**: Not supported, automatically falls back to Float16
- **Flash Attention**: Not available
- **4-bit Quantization**: Limited support, only through MLX or Quanto
- **Memory Tracking**: Limited memory profiling capabilities
- **Distributed Training**: Experimental support only

### MLX Backend
- **Distributed Training**: Not supported
- **Safetensors Format**: Cannot directly load safetensors files - requires conversion to .npz format first
- **PyTorch Integration**: Limited interoperability with PyTorch models
- **FSDP**: Not supported (MLX has its own parallelization strategies)

### CPU Backend
- **Performance**: Very slow for training, suitable only for testing
- **Quantization**: Limited to HQQ 8-bit quantization
- **Memory**: High RAM requirements (32GB+ for 7B models)

## Quantization Limitations

### BitsAndBytes (BnB)
- **Backend Support**: CUDA only
- **Bit Widths**: 4-bit and 8-bit only
- **Double Quantization**: Experimental feature, may cause instability

### HQQ (Half-Quadratic Quantization)
- **Model Support**: Limited to specific architectures (LLaMA, Mistral, etc.)
- **Calibration**: Requires calibration data for optimal performance
- **Backend Support**: CUDA and CPU only

### MLX Quantization
- **Backend Support**: MLX/MPS only
- **Integration**: Cannot be used with PyTorch FSDP
- **Model Formats**: Requires MLX-specific model format

### Quanto
- **Bit Widths**: 2, 4, and 8-bit support
- **Performance**: Slower than BitsAndBytes on CUDA
- **Activation Quantization**: Limited support

## Model Loading Limitations

### Safetensors Support
- **MLX Backend**: Not supported - models must be converted to .npz format
- **Sharded Models**: Limited support for models split across multiple files

### Model Size Constraints
- **70B Models**: Requires multi-GPU setup or aggressive quantization
- **Memory Estimation**: Automatic memory estimation may be inaccurate for complex models

### Checkpoint Loading
- **Mixed Precision**: Loading fp16 checkpoints into bf16 training may cause issues
- **Quantized Checkpoints**: Cross-backend loading not supported (e.g., BnB checkpoint to HQQ)

## Training Limitations

### LoRA/QLoRA
- **Target Modules**: Not all model architectures have optimal target module configurations
- **Rank Selection**: No automatic rank selection based on task/model
- **Mixed Precision LoRA**: Some precision combinations untested

### FSDP (Fully Sharded Data Parallel)
- **MLX Backend**: Not supported
- **CPU Offloading**: Limited testing, may cause performance issues
- **Gradient Checkpointing**: Not compatible with all model architectures

### DoRA (Weight-Decomposed Low-Rank Adaptation)
- **Backend Support**: Limited to specific quantization methods
- **Performance**: Slower than standard LoRA
- **Memory**: Higher memory usage than LoRA

### Block Expansion (LLaMA-Pro style)
- **Model Support**: Only tested on LLaMA architecture
- **Configuration**: Manual configuration required
- **Checkpoint Compatibility**: Cannot load standard model checkpoints

## Feature Status

### ✅ Fully Implemented
- Basic LoRA/QLoRA training
- Multi-backend support (CUDA, MPS, CPU)
- Learning rate scheduling
- Gradient accumulation
- Mixed precision training (fp16, bf16, fp32)
- Basic quantization (4-bit, 8-bit)

### ⚠️ Partially Implemented
- Distributed training (CUDA only, experimental on MPS)
- Flash Attention (CUDA only)
- Model sharding (basic support)
- Profiling and debugging tools
- Checkpoint resumption

### ❌ Not Implemented
- Automatic hyperparameter tuning
- Model parallelism (beyond FSDP)
- Inference optimization
- Serving/deployment features
- Windows native support
- ROCm (AMD GPU) support

## Workarounds and Recommendations

### For MLX Safetensors Loading
```python
# Convert safetensors to MLX format first
from mlx_lm import convert
convert("model_path", "output_path", q_bits=4)
```

### For MPS BFloat16
```python
# Automatically handled in train.py, but can be forced:
if backend == "mps" and precision == "bf16":
    precision = "fp16"
    print("MPS doesn't support bf16, using fp16 instead")
```

### For Large Model Loading
```python
# Use sequential loading for large models
model_loader = ModelLoader(
    backend_manager,
    loading_strategy="sequential",
    low_memory=True
)
```

## Error Messages and Solutions

### "Safetensors format is not yet supported for MLX models"
- **Solution**: Convert model to MLX format using `mlx_lm.convert`

### "Framework 'X' is not supported for model loading"
- **Solution**: Use a supported framework (pytorch, tensorflow, jax)

### "Backend 'X' is not available on this system"
- **Solution**: Use an available backend or install required dependencies

## Future Improvements

1. **Safetensors support for MLX**: Direct loading without conversion
2. **Windows support**: Native Windows compatibility
3. **ROCm support**: AMD GPU acceleration
4. **Automatic rank selection**: Task-aware LoRA rank optimization
5. **Inference optimization**: Deployment-ready model serving

## Reporting Issues

When reporting issues related to these limitations:
1. Specify your backend and hardware configuration
2. Include the full error message
3. Provide a minimal reproducible example
4. Check if a workaround exists in this document

For feature requests or to contribute implementations for these limitations, please open an issue on the GitHub repository.