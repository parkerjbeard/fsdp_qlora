# FSDP QLoRA Documentation

Welcome to the FSDP QLoRA documentation. This implementation provides multi-backend support for efficient fine-tuning of large language models using Quantized Low-Rank Adaptation (QLoRA) with Fully Sharded Data Parallel (FSDP).

## Quick Links

### Getting Started
- [Backend Usage Guide](backend_usage.md) - How to use different compute backends
- [Migration Guides](backend-migration-guides.md) - Moving between backends
- [Limitations](LIMITATIONS.md) - Known limitations and workarounds

### Core Components
- [Backend Manager](backend_usage.md) - Automatic backend detection and management
- [Import Abstraction](import_abstraction.md) - Conditional imports for optional dependencies
- [Model Loading](model_loading_abstraction.md) - Unified model loading interface
- [Quantization](quantization_abstraction.md) - Multi-method quantization support

### Backend-Specific Guides
- [MPS FSDP Guide](mps_fsdp_guide.md) - Apple Silicon with PyTorch FSDP
- [MPS Quantization](mps_quantization_guide.md) - Quantization on Apple Silicon
- [MLX Integration](mlx_integration.md) - Native Apple Silicon framework
- [MLX Utilities](mlx_utilities_guide.md) - MLX-specific tools and helpers

### Testing & Development
- [Testing Guide](testing.md) - Comprehensive test suite documentation
- [Integration Tests](integration-test-summary.md) - Integration test overview
- [Test Results](test-results-summary.md) - Latest test results

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│             User Interface (CLI)             │
├─────────────────────────────────────────────┤
│          Training Script (train.py)          │
├─────────────┬─────────┬──────────┬──────────┤
│   Backend   │  Model  │Quantiza- │ Learning │
│   Manager   │ Loading │  tion    │  Rate    │
├─────────────┴─────────┴──────────┴──────────┤
│          Backend Implementations             │
│  ┌──────┐  ┌─────┐  ┌─────┐  ┌──────────┐  │
│  │ CUDA │  │ MPS │  │ MLX │  │   CPU    │  │
│  └──────┘  └─────┘  └─────┘  └──────────┘  │
└─────────────────────────────────────────────┘
```

## Supported Features by Backend

| Feature | CUDA | MPS | MLX | CPU |
|---------|------|-----|-----|-----|
| Training | ✅ | ✅ | ✅ | ✅ |
| FSDP | ✅ | ⚠️ | ❌ | ✅ |
| BFloat16 | ✅ | ❌ | ❌ | ✅ |
| Flash Attention | ✅ | ❌ | ❌ | ❌ |
| 4-bit Quantization | ✅ | ✅ | ✅ | ❌ |
| Multi-GPU | ✅ | ⚠️ | ❌ | ✅ |

Legend: ✅ Full support | ⚠️ Experimental | ❌ Not supported

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fsdp_qlora.git
cd fsdp_qlora

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies based on your platform
python setup.py  # Automatic platform detection and installation
```

### 2. Basic Training

```bash
# Automatic backend detection
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset alpaca \
  --train_type qlora \
  --batch_size 4 \
  --num_epochs 3

# Specify backend explicitly
python train.py \
  --backend cuda \  # or mps, mlx, cpu
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset alpaca \
  --train_type qlora
```

### 3. Advanced Configuration

```bash
# Full configuration example
python train.py \
  --backend cuda \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset custom_data.json \
  --train_type qlora \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --lr_scheduler cosine \
  --warmup_ratio 0.1 \
  --context_length 2048 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --lora_target_modules all \
  --quantization_backend bnb \
  --n_bits 4 \
  --double_quant true \
  --precision bf16 \
  --use_gradient_checkpointing true \
  --use_flash_attention true \
  --save_model true \
  --output_dir ./outputs \
  --logging_steps 10 \
  --eval_steps 100 \
  --save_steps 500 \
  --max_steps -1 \
  --world_size -1 \
  --low_memory false
```

## Component Documentation

### Backend Manager
Handles automatic detection and configuration of compute backends:
- Detects available hardware (CUDA, MPS, MLX, CPU)
- Configures device placement
- Manages backend-specific capabilities
- [Full Documentation](backend_usage.md)

### Model Loading
Provides unified interface for loading models across backends:
- Memory-efficient loading strategies
- Automatic weight conversion
- Checkpoint management
- [Full Documentation](model_loading_abstraction.md)

### Quantization
Supports multiple quantization methods:
- **BitsAndBytes**: 4/8-bit quantization (CUDA only)
- **HQQ**: Hardware-agnostic quantization
- **MLX**: Apple Silicon native quantization
- **Quanto**: Cross-platform quantization
- [Full Documentation](quantization_abstraction.md)

### Learning Rate Scheduling
Comprehensive scheduler support:
- Constant, Linear, Cosine, Polynomial, Exponential
- Warmup support (steps, ratio, or epochs)
- Integration with all training configurations
- [Full Documentation](todo_fixes_changelog.md)

## Performance Tips

### CUDA (NVIDIA GPUs)
- Use BFloat16 precision for best performance/stability
- Enable Flash Attention for long contexts
- Use BitsAndBytes 4-bit quantization for memory efficiency

### MPS (Apple Silicon)
- Use Float16 (BFloat16 not supported)
- Consider MLX backend for better performance
- Use Quanto for quantization if MLX unavailable

### MLX (Apple Silicon Native)
- Convert models to MLX format first
- Use native MLX quantization
- Leverage unified memory architecture

### CPU
- Use for testing only (very slow)
- Reduce batch size significantly
- Consider HQQ 8-bit quantization

## Troubleshooting

### Common Issues

1. **"Backend X not available"**
   - Check hardware compatibility
   - Verify driver installation
   - See [Backend Migration Guide](backend-migration-guides.md)

2. **"Out of Memory"**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use more aggressive quantization
   - Enable CPU offloading (experimental)

3. **"Precision not supported"**
   - MPS doesn't support BFloat16 (use Float16)
   - Check backend capabilities in documentation

4. **"Quantization method not available"**
   - Verify backend compatibility
   - Install required dependencies
   - See [Limitations](LIMITATIONS.md)

### Getting Help

1. Check [Limitations](LIMITATIONS.md) for known issues
2. Review backend-specific guides
3. Search existing GitHub issues
4. Create new issue with:
   - System information
   - Complete error message
   - Minimal reproducible example

## Contributing

We welcome contributions! Areas of interest:
- Windows support
- ROCm (AMD GPU) support
- Additional model architectures
- Performance optimizations
- Bug fixes and documentation

See [Contributing Guide](../CONTRIBUTING.md) for details.

## Recent Updates

- ✅ Learning rate scheduler implementation
- ✅ Comprehensive test suite
- ✅ Backend migration guides
- ✅ Limitations documentation
- ✅ Code cleanup and standardization

See [Changelog](todo_fixes_changelog.md) for full history.