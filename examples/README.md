# FSDP+QLoRA Examples

This directory contains example scripts demonstrating various features of FSDP+QLoRA.

## Available Examples

### model_loading_example.py

Demonstrates the new model loading abstraction layer:

- Simple model loading with auto-detection
- Loading with quantization
- Backend-specific optimizations
- Custom configuration
- Integration with train.py workflow

Run the example:
```bash
python examples/model_loading_example.py
```

### mlx_training_example.py

Demonstrates MLX framework integration for Apple Silicon:

- MLX model configuration
- 4-bit quantization with MLX
- LoRA adapters in MLX
- Unified memory optimization
- Training loop example

Run the example:
```bash
python examples/mlx_training_example.py
```

### mlx_complete_training.py

Complete end-to-end MLX training example:

- Full training pipeline with MLX
- Dataset loading and preprocessing
- Model configuration and initialization
- Training with evaluation
- Checkpoint saving and loading

Run the example:
```bash
python examples/mlx_complete_training.py
```

### mlx_utilities_demo.py

Comprehensive demonstration of MLX utilities:

- Dataset conversion from PyTorch/HuggingFace to MLX
- Tokenizer integration with MLX arrays
- Memory profiling on Apple Silicon
- Performance monitoring and benchmarking
- Model optimization and batch size calculation

Run the example:
```bash
# Run all demos
python examples/mlx_utilities_demo.py

# Run specific demo
python examples/mlx_utilities_demo.py --demo memory
python examples/mlx_utilities_demo.py --demo performance
python examples/mlx_utilities_demo.py --demo optimization
```

### mps_fsdp_example.py

Complete example of FSDP (Fully Sharded Data Parallel) training on MPS:

- MPS compatibility checking
- Model wrapping with FSDP
- Memory-efficient sharding strategies
- Mixed precision training (float16)
- Checkpoint management
- Performance profiling

Run the example:
```bash
# Basic training with auto-configuration
python examples/mps_fsdp_example.py

# Small model with mixed precision
python examples/mps_fsdp_example.py --model-size small --mixed-precision

# Large model with full sharding and CPU offload
python examples/mps_fsdp_example.py --model-size large --sharding-strategy FULL_SHARD --cpu-offload

# With memory profiling
python examples/mps_fsdp_example.py --profile-memory --save-checkpoint
```

### mps_quantization_example.py

Comprehensive demonstration of MPS-optimized quantization:

- Dynamic quantization strategy selection based on memory
- PyTorch native quantization (INT8) for MPS
- HQQ integration with MPS compatibility
- Performance benchmarking and optimization
- Memory-aware training with quantization
- Model-specific optimization strategies

Run the example:
```bash
# Run all demos
python examples/mps_quantization_example.py --all

# Demonstrate dynamic strategy selection
python examples/mps_quantization_example.py --demo-strategy

# Train with auto-selected quantization
python examples/mps_quantization_example.py --train --auto-select --epochs 3

# Benchmark different quantization methods
python examples/mps_quantization_example.py --benchmark --batch-size 8

# Train with specific quantization
python examples/mps_quantization_example.py --train --method PYTORCH_DYNAMIC --bits 8 --memory-efficient

# Use with HuggingFace models
python examples/mps_quantization_example.py --model-type huggingface --model-name bert-base-uncased --train
```

## Documentation

### MLX Training Guide

See [mlx_training_guide.md](mlx_training_guide.md) for comprehensive documentation on:

- Hardware requirements and recommendations
- Configuration options
- Performance optimization tips
- Troubleshooting common issues
- Complete API reference

### MLX Utilities Guide

See [../docs/mlx_utilities_guide.md](../docs/mlx_utilities_guide.md) for detailed documentation on:

- Dataset conversion utilities
- Tokenizer integration
- Memory profiling tools
- Performance monitoring
- Helper functions and best practices

## Testing

The MLX trainer includes comprehensive test coverage:

- Unit tests: `tests/test_mlx_trainer.py`
- Integration tests: `tests/test_mlx_trainer_integration.py`

Run tests:
```bash
# Run all MLX tests
python -m pytest tests/test_mlx_trainer*.py -v

# Run specific test class
python -m pytest tests/test_mlx_trainer_integration.py::TestMLXTrainerQuantizationIntegration -v
```

## Coming Soon

- Multi-backend benchmarking scripts
- Memory profiling tools
- FSDP configuration examples
- Model conversion utilities