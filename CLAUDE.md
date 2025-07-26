# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup and Installation
```bash
# Platform-aware automatic setup (detects CUDA/MPS/MLX/CPU)
python setup.py

# Login to Hugging Face (required for model access)
huggingface-cli login
```

### Running Training
```bash
# Basic QLoRA training
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --batch_size 2 \
  --context_length 512 \
  --precision bf16 \
  --train_type qlora \
  --use_gradient_checkpointing true \
  --dataset alpaca

# Multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1
python train.py --world_size 2 --train_type qlora --model_name meta-llama/Llama-2-70b-hf

# Apple Silicon (MPS) training
python train.py --backend mps --model_name meta-llama/Llama-2-7b-hf --train_type qlora --precision fp16
```

### Testing
```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_backend_manager.py -v

# Run integration tests
pytest tests/test_train_integration.py -v

# Platform-specific tests
pytest tests/test_mlx_integration.py      # MLX tests
pytest tests/test_mps_fsdp_integration.py  # MPS tests
```

### Linting and Type Checking
```bash
# Run linting
ruff check .

# Run type checking
mypy .
```

## Architecture Overview

This codebase implements a multi-backend training system for large language models with quantized LoRA/QLoRA support. The architecture is built around several key abstractions:

### Core Components

1. **Backend Manager** (`src/core/backend_manager.py`): Central hub for backend selection and configuration. Automatically detects available backends (CUDA, MPS, MLX, CPU) and manages device placement.

2. **Model Loader** (`src/core/model_loader.py`): Provides a unified interface for loading models across different backends. Handles memory-efficient loading and backend-specific optimizations.

3. **Quantization Wrapper** (`src/core/quantization_wrapper.py`): Abstracts different quantization methods (BitsAndBytes, HQQ, MLX) behind a common interface. Enables switching between quantization backends without changing training code.

4. **Import Abstraction** (`src/core/imports.py`): Manages conditional imports based on available backends, preventing import errors on systems without specific hardware support.

### Training Flow

The main training script (`train.py`) orchestrates the training process:
1. Backend detection and initialization via `BackendManager`
2. Model loading through `ModelLoader` with optional quantization
3. FSDP wrapper application for distributed training
4. Training loop with backend-specific optimizations

### Supported Training Types

- `full`: Standard full parameter fine-tuning
- `lora`/`custom_lora`: LoRA implementations (HF PEFT or custom)
- `qlora`/`custom_qlora`: 4-bit quantized LoRA variants
- `hqq_lora`/`hqq_dora`: HQQ quantization variants
- `bnb_dora`: DoRA with BitsAndBytes
- `bnb_llama_pro`/`hqq_llama_pro`: Llama-Pro block expansion methods

### Key Design Principles

1. **Backend Abstraction**: All backend-specific code is isolated behind interfaces, allowing seamless switching between CUDA, MPS, MLX, and CPU.

2. **Lazy Loading**: Models are loaded iteratively to minimize memory usage, especially important for large models.

3. **Quantization Flexibility**: Multiple quantization backends are supported with a common interface, enabling experimentation with different methods.

4. **FSDP Integration**: Deep integration with PyTorch's Fully Sharded Data Parallel for efficient multi-GPU training.

### Directory Structure

```
src/
├── core/               # Core abstractions and interfaces
│   ├── backend_manager.py    # Backend detection and management
│   ├── imports.py           # Import abstraction layer
│   ├── model_loader.py      # Model loading strategies
│   └── quantization_wrapper.py # Quantization method abstractions
│
├── backends/          # Backend-specific implementations
│   ├── mlx/          # Apple MLX framework support
│   │   ├── mlx_model_wrapper.py
│   │   ├── mlx_quantization.py
│   │   ├── mlx_trainer.py
│   │   ├── mlx_utils.py
│   │   └── pytorch_mlx_bridge.py
│   │
│   ├── mps/          # Apple Metal Performance Shaders support
│   │   ├── mps_fsdp_wrapper.py
│   │   ├── mps_quantization.py
│   │   └── mps_quantization_quanto.py
│   │
│   └── cuda/         # NVIDIA CUDA support (reserved for future)
│
└── utils/            # Utility modules
    ├── profiling_utils.py    # Performance profiling
    └── unified_quantization.py # Unified quantization interface
```

Additional directories:
- `scripts/`: Utility scripts for LoRA merging, DoRA conversion, and block expansion
- `tests/`: Comprehensive test suite covering unit and integration tests
- `docs/`: Detailed documentation for each component and backend
- `examples/`: Usage examples and configuration templates
- `archive/`: Archived files from development

### Import Pattern

All imports now follow the organized structure:
```python
# Core imports
from src.core.backend_manager import BackendManager
from src.core.imports import get_module
from src.core.model_loader import ModelLoader
from src.core.quantization_wrapper import QuantizationWrapper

# Backend-specific imports
from src.backends.mlx.mlx_model_wrapper import MLXModelWrapper
from src.backends.mps.mps_fsdp_wrapper import MPSFSDPWrapper

# Utility imports
from src.utils.profiling_utils import profiling_context
from src.utils.unified_quantization import UnifiedQuantization
```

When modifying the codebase, ensure compatibility across all supported backends by testing with the appropriate test files and maintaining the abstraction boundaries.

## Organizational Memories

- Implement a centralized organizational structure with clear separation of concerns between backend management, model loading, quantization, and training components
- Maintain modular design to allow easy extension and swapping of backend implementations
- Prioritize backend-agnostic interfaces to support future hardware and quantization method integrations