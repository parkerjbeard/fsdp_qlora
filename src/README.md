# Source Code Organization

This directory contains the organized source code for the FSDP+QLoRA project.

## Directory Structure

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
│   └── cuda/         # NVIDIA CUDA support (placeholder for future)
│
└── utils/            # Utility modules
    ├── profiling_utils.py    # Performance profiling
    └── unified_quantization.py # Unified quantization interface
```

## Module Descriptions

### Core Modules
- **backend_manager.py**: Manages backend detection, capabilities, and configuration
- **imports.py**: Handles conditional imports based on available backends
- **model_loader.py**: Provides different strategies for loading large models
- **quantization_wrapper.py**: Abstracts different quantization libraries (BitsAndBytes, HQQ, MLX, Quanto)

### Backend Modules
- **mlx/**: Native Apple Silicon support using MLX framework
- **mps/**: PyTorch Metal Performance Shaders backend for Apple GPUs
- **cuda/**: Reserved for CUDA-specific optimizations

### Utilities
- **profiling_utils.py**: Memory and performance profiling tools
- **unified_quantization.py**: Unified interface for quantization across backends

## Import Examples

```python
# Import core modules
from src.core.backend_manager import BackendManager, Backend
from src.core.imports import get_module, check_import_availability
from src.core.model_loader import ModelLoader
from src.core.quantization_wrapper import QuantizationWrapper

# Import backend-specific modules
from src.backends.mlx.mlx_model_wrapper import MLXModelWrapper
from src.backends.mps.mps_fsdp_wrapper import MPSFSDPWrapper

# Import utilities
from src.utils.profiling_utils import profiling_context
from src.utils.unified_quantization import UnifiedQuantization
```