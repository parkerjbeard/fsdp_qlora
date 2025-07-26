# Backend Support Documentation

## Overview

The FSDP QLoRA training script now supports multiple compute backends beyond CUDA, enabling training on Apple Silicon (MPS/MLX) and CPU devices. The backend manager automatically detects available hardware and configures training appropriately.

## Supported Backends

### CUDA (Default)
- **Hardware**: NVIDIA GPUs
- **Features**: Full support for all training modes
- **Quantization**: 4-bit, 8-bit, 16-bit
- **Distributed**: Yes (NCCL backend)
- **Recommended for**: Production training, best performance

### MPS (Metal Performance Shaders)
- **Hardware**: Apple Silicon Macs (M1/M2/M3)
- **Features**: PyTorch native Apple GPU support
- **Quantization**: 8-bit, 16-bit (no 4-bit yet)
- **Distributed**: Yes (Gloo backend)
- **Notes**: 
  - No bfloat16 support (uses float16 instead)
  - Limited operator coverage
  - Single GPU only

### MLX (Apple's ML Framework)
- **Hardware**: Apple Silicon Macs
- **Features**: Optimized for unified memory architecture
- **Quantization**: 4-bit, 8-bit, 16-bit
- **Distributed**: No
- **Notes**: 
  - Requires MLX library installation
  - Efficient memory usage
  - Single device only

### CPU
- **Hardware**: Any CPU
- **Features**: Fallback option
- **Quantization**: 8-bit, 16-bit
- **Distributed**: Yes (Gloo backend)
- **Notes**: Very slow, for testing only

## Usage

### Basic Usage

```bash
# Auto-detect best available backend (recommended)
python train.py --backend auto

# Explicitly select CUDA
python train.py --backend cuda

# Use Apple Silicon MPS
python train.py --backend mps

# Use MLX framework
python train.py --backend mlx

# Force CPU backend
python train.py --backend cpu
```

### Backend-Specific Examples

#### Training on Apple Silicon with MPS
```bash
python train.py \
    --backend mps \
    --model_name meta-llama/Llama-2-7b-hf \
    --train_type qlora \
    --batch_size 2 \
    --precision fp16_autocast \
    --num_epochs 1
```

#### Training with MLX
```bash
# First install MLX
pip install mlx

python train.py \
    --backend mlx \
    --model_name meta-llama/Llama-2-7b-hf \
    --train_type qlora \
    --world_size 1 \
    --batch_size 4
```

## Backend Selection Logic

1. **Default**: CUDA (for backward compatibility)
2. **Auto mode**: Selects in order: CUDA → MLX → MPS → CPU
3. **Validation**: Ensures requested backend is available
4. **Automatic adjustments**:
   - Precision changes (e.g., bf16 → fp16 for MPS)
   - World size constraints (e.g., 1 for MLX)
   - Memory optimizations

## Configuration Adjustments by Backend

The system automatically adjusts configurations based on backend capabilities:

### MPS Adjustments
- `bf16` precision → `fp16` (with warning)
- 4-bit quantization → Shows performance warning
- Distributed backend → Gloo instead of NCCL

### MLX Adjustments
- Multi-GPU (`world_size > 1`) → Error (not supported)
- Sets `backend_type="mlx"` for special handling

### CPU Adjustments
- Shows performance warning
- Quantization → Limited support warning

## Memory Management

Different backends report memory differently:

```python
# CUDA: Precise GPU memory tracking
Memory allocated: 4.5 GiB
Memory reserved: 6.2 GiB

# MPS: Limited memory info
Memory info not available through PyTorch

# MLX: System memory (unified)
Total memory: 64.0 GiB
Available memory: 32.5 GiB
```

## Performance Expectations

Relative performance for 7B model training:

| Backend | Tokens/sec | Memory Usage | Notes |
|---------|------------|--------------|-------|
| CUDA (A100) | 50-100 | 16-24 GB | Fastest |
| CUDA (4090) | 30-50 | 16-24 GB | Consumer GPU |
| MLX (M3 Ultra) | 10-20 | <64 GB | Unified memory |
| MPS (M3 Ultra) | 5-15 | <64 GB | PyTorch native |
| CPU | <1 | System RAM | Testing only |

## Troubleshooting

### Common Issues

1. **"Backend 'cuda' is not available"**
   - You're on a non-NVIDIA system
   - Use `--backend auto` or specify available backend

2. **"MPS backend doesn't support bfloat16"**
   - Automatic conversion to fp16
   - Use `--precision fp16` explicitly

3. **"MLX backend doesn't support distributed training"**
   - Set `--world_size 1`
   - MLX is single-device only

4. **Memory errors on MPS/MLX**
   - Reduce batch size
   - Enable `--use_gradient_checkpointing`
   - Use `--low_memory` mode

### Checking Available Backends

```python
from backend_manager import BackendManager

# Check what's available
manager = BackendManager(verbose=True)
# Prints detailed system info and capabilities
```

## Advanced Configuration

### Environment Variables
```bash
# Force specific backend via environment
export FSDP_BACKEND=mps
python train.py

# Combine with other settings
export FSDP_BACKEND=auto
export CUDA_VISIBLE_DEVICES=0,1
```

### Custom Backend Logic
```python
# In your training script
from backend_manager import BackendManager, Backend

# Create manager with custom settings
manager = BackendManager(backend="auto", verbose=True)

# Check capabilities
if manager.supports_quantization(4):
    print("4-bit quantization available!")

# Get optimal batch size
batch_size = manager.get_optimal_batch_size(model_size_b=7)
```

## Contributing

To add support for a new backend:

1. Add to `Backend` enum in `backend_manager.py`
2. Define capabilities in `CAPABILITIES` dict
3. Implement device detection in `_detect_available_backends()`
4. Add backend-specific logic in `train.py`
5. Write tests in `tests/test_backend_manager.py`

## Future Enhancements

- [ ] ROCm/AMD GPU support
- [ ] Intel GPU support via IPEX
- [ ] Distributed MLX training
- [ ] 4-bit quantization on MPS
- [ ] Automatic batch size tuning
- [ ] Cross-backend model conversion