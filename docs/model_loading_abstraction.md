# Model Loading Abstraction Documentation

The Model Loading Abstraction Layer provides a unified interface for loading models across different backends with support for various quantization methods and memory-efficient loading strategies.

## Overview

The model loading abstraction (`model_loader.py`) simplifies the complex model loading logic in FSDP+QLoRA by providing:

- **Unified Interface**: Common API for all loading scenarios
- **Backend-Specific Optimizations**: Tailored loading strategies for each backend
- **Memory-Efficient Loading**: Support for low-memory and unified memory architectures
- **Quantization Integration**: Seamless integration with the quantization abstraction layer
- **Parallel Loading**: Efficient parallel weight loading for large models

## Architecture

```
ModelLoadingConfig (configuration)
    ↓
ModelLoaderFactory.create_loader()
    ↓
ModelLoader (ABC)
    ├── StandardModelLoader
    └── QuantizedModelLoader
         ├── CUDAModelLoader
         ├── MPSModelLoader
         ├── MLXModelLoader
         └── CPUModelLoader
```

## Key Components

### LoadingStrategy Enum

```python
class LoadingStrategy(enum.Enum):
    FULL = "full"                      # Load full model to device
    LOW_MEMORY = "low_memory"          # Load to CPU/meta for memory efficiency
    UNIFIED_MEMORY = "unified_memory"  # For Apple Silicon unified memory
    STREAMING = "streaming"            # Load weights on demand (future)
```

### ModelLoadingConfig

Comprehensive configuration for model loading:

```python
@dataclass
class ModelLoadingConfig:
    model_name: str
    backend: Backend
    loading_strategy: LoadingStrategy = LoadingStrategy.FULL
    quantization_config: Optional[QuantizationConfig] = None
    dtype: torch.dtype = torch.float16
    device: Optional[Union[str, torch.device]] = None
    
    # Memory settings
    low_memory: bool = False
    offload_to_cpu: bool = False
    max_memory: Optional[Dict[str, str]] = None
    
    # Performance settings
    loading_workers: int = -1  # -1 for auto
    use_safetensors: bool = True
    
    # Model settings
    use_cache: bool = False
    attn_implementation: str = "sdpa"
```

## Usage Examples

### Basic Model Loading

```python
from model_loader import load_model_and_tokenizer
from backend_manager import Backend

# Simple loading
model, tokenizer = load_model_and_tokenizer(
    "meta-llama/Llama-2-7b-hf",
    backend="cuda"
)
```

### Quantized Model Loading

```python
from model_loader import ModelLoadingConfig, ModelLoaderFactory
from quantization_wrapper import QuantizationConfig, QuantizationMethod

# Create quantization config
quant_config = QuantizationConfig(
    method=QuantizationMethod.BNB_NF4,
    bits=4,
    compute_dtype=torch.bfloat16
)

# Create loading config
config = ModelLoadingConfig(
    model_name="meta-llama/Llama-2-70b-hf",
    backend=Backend.CUDA,
    quantization_config=quant_config,
    low_memory=True,
    loading_workers=4
)

# Load model
loader = ModelLoaderFactory.create_loader(config)
model = loader.load_model()
tokenizer = loader.load_tokenizer()
```

### Backend-Specific Examples

#### CUDA with Low Memory

```python
config = ModelLoadingConfig(
    model_name="meta-llama/Llama-2-70b-hf",
    backend=Backend.CUDA,
    loading_strategy=LoadingStrategy.LOW_MEMORY,
    quantization_config=QuantizationConfig(method=QuantizationMethod.BNB_NF4),
    dtype=torch.bfloat16,
    low_memory=True
)
```

#### Apple Silicon (MPS)

```python
config = ModelLoadingConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    backend=Backend.MPS,
    # Automatically uses UNIFIED_MEMORY strategy
    dtype=torch.float16,  # MPS doesn't support bfloat16
    quantization_config=QuantizationConfig(method=QuantizationMethod.MLX_INT4)
)
```

#### CPU with HQQ Quantization

```python
config = ModelLoadingConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    backend=Backend.CPU,
    quantization_config=QuantizationConfig(
        method=QuantizationMethod.HQQ,
        bits=8,
        group_size=128
    ),
    loading_workers=8  # More workers for CPU
)
```

### Automatic Configuration

```python
from model_loader import get_recommended_loader_config

# Get recommended configuration based on system
config = get_recommended_loader_config(
    "meta-llama/Llama-2-70b-hf",
    backend=Backend.CUDA,
    available_memory_gb=24.0  # 24GB GPU
)
# Automatically selects quantization and loading strategy
```

## Integration with train.py

The model loader can replace the complex loading logic in train.py:

### Before (train.py original logic)

```python
# Complex if/else logic for different train types
if args["train_type"] in ["full", "lora", "custom_lora"]:
    if (args["low_memory"] and rank == 0) or (not args["low_memory"]):
        model = AutoModelForCausalLM.from_pretrained(...)
    else:
        cfg = AutoConfig.from_pretrained(...)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg, ...)
elif args["train_type"] in ["qlora", "custom_qlora", ...]:
    # Complex quantization loading logic
    cfg = AutoConfig.from_pretrained(...)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    # Replace linear layers
    # Load and quantize weights in parallel
    # ...100+ lines of code
```

### After (with model loader)

```python
from model_loader import ModelLoadingConfig, ModelLoaderFactory
from quantization_wrapper import QuantizationConfig, QuantizationMethod

# Map train_type to quantization config
quant_config = None
if args.train_type in ["qlora", "custom_qlora"]:
    quant_config = QuantizationConfig(
        method=QuantizationMethod.BNB_NF4,
        bits=args.q_bits,
        compute_dtype=dtype
    )
elif args.train_type in ["hqq_lora", "hqq_dora"]:
    quant_config = QuantizationConfig(
        method=QuantizationMethod.HQQ,
        bits=args.n_bits
    )

# Create config and load
config = ModelLoadingConfig(
    model_name=args.model_name,
    backend=backend_manager.backend,
    quantization_config=quant_config,
    dtype=dtype,
    low_memory=args.low_memory,
    loading_workers=args.loading_workers,
    rank=args.rank
)

loader = ModelLoaderFactory.create_loader(config)
model = loader.load_model()
tokenizer = loader.load_tokenizer()
```

## Memory-Efficient Loading

### Low Memory Mode

For multi-GPU setups with limited memory:

```python
config = ModelLoadingConfig(
    model_name="meta-llama/Llama-2-70b-hf",
    backend=Backend.CUDA,
    loading_strategy=LoadingStrategy.LOW_MEMORY,
    low_memory=True,
    rank=rank,  # Rank 0 loads to CPU, others create empty model
    world_size=world_size
)
```

### Unified Memory (Apple Silicon)

Automatically enabled for MPS/MLX backends:

```python
# Unified memory is automatically used for Apple Silicon
config = ModelLoadingConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    backend=Backend.MPS
)
# config.loading_strategy will be UNIFIED_MEMORY
```

## Parallel Weight Loading

The loader automatically determines optimal parallel loading:

```python
# Auto-determine workers based on backend and model size
config = ModelLoadingConfig(
    model_name="meta-llama/Llama-2-70b-hf",
    backend=Backend.CUDA,
    loading_workers=-1  # Auto
)

# Or specify manually
config.loading_workers = 8
```

Worker selection heuristics:
- **CUDA**: 1-8 workers based on GPU memory and quantization method
- **MPS/MLX**: 4 workers (unified memory allows more parallelism)
- **CPU**: Half of available CPU cores

## Advanced Features

### Custom Loading Strategies

Extend the loader for custom scenarios:

```python
class StreamingModelLoader(QuantizedModelLoader):
    """Custom loader that streams weights on demand."""
    
    def load_model(self) -> nn.Module:
        # Create model structure
        model = self._create_empty_model()
        
        # Set up weight streaming
        self._setup_weight_streaming(model)
        
        return model
    
    def _setup_weight_streaming(self, model):
        # Custom logic for on-demand weight loading
        pass
```

### LLaMA Pro Support

Load models with expanded layers:

```python
config = ModelLoadingConfig(
    model_name="base-model",
    backend=Backend.CUDA,
    llama_pro_path="/path/to/llama_pro_blk_exp-32-40",
    quantization_config=QuantizationConfig(method=QuantizationMethod.HQQ)
)
```

### Integration with FSDP

The loader works seamlessly with FSDP:

```python
# Load model with appropriate strategy
model, tokenizer = load_model_and_tokenizer(
    args.model_name,
    backend=backend_manager.backend,
    low_memory=True,
    rank=rank,
    world_size=world_size
)

# Wrap with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=args.use_cpu_offload),
    ...
)
```

## Performance Optimization

### Backend-Specific Optimizations

1. **CUDA**
   - Uses Flash Attention 2 via SDPA
   - Optimized parallel loading for bitsandbytes
   - Automatic device placement

2. **MPS (Apple Silicon)**
   - Converts bfloat16 to float16 automatically
   - Uses unified memory architecture
   - Optimized for MLX quantization

3. **CPU**
   - More parallel workers for loading
   - Prefers HQQ quantization
   - Memory-mapped file support

### Loading Time Comparison

Approximate loading times for Llama-2-7B:

| Backend | Standard | Quantized (4-bit) | Low Memory |
|---------|----------|-------------------|------------|
| CUDA    | 15s      | 45s               | 60s        |
| MPS     | 20s      | 50s               | N/A        |
| CPU     | 30s      | 90s               | 120s       |

## Troubleshooting

### Common Issues

1. **"CUDA out of memory" during loading**
   ```python
   # Use low memory mode
   config.loading_strategy = LoadingStrategy.LOW_MEMORY
   config.low_memory = True
   ```

2. **"bitsandbytes not available" on Mac**
   ```python
   # Use HQQ or MLX quantization instead
   config.quantization_config = QuantizationConfig(
       method=QuantizationMethod.HQQ
   )
   ```

3. **Slow loading on CPU**
   ```python
   # Increase loading workers
   config.loading_workers = multiprocessing.cpu_count()
   ```

### Debug Information

Enable verbose logging:

```python
config = ModelLoadingConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    backend=Backend.CUDA,
    verbose=True,
    rank=0
)
```

## Future Enhancements

Planned features for the model loading abstraction:

1. **Streaming Loading**: Load weights on-demand during inference
2. **Checkpoint Resumption**: Resume from partially loaded checkpoints
3. **Multi-Model Loading**: Load multiple models efficiently
4. **Dynamic Quantization**: Quantize during loading based on available memory
5. **Remote Model Loading**: Stream models from cloud storage

## API Reference

### Core Classes

- `ModelLoadingConfig`: Configuration dataclass
- `ModelLoader`: Abstract base class
- `StandardModelLoader`: Non-quantized model loading
- `QuantizedModelLoader`: Quantized model loading base
- `ModelLoaderFactory`: Factory for creating loaders

### Convenience Functions

- `load_model_and_tokenizer()`: Load model and tokenizer in one call
- `get_recommended_loader_config()`: Get recommended configuration

### Backend-Specific Loaders

- `CUDAModelLoader`: CUDA-optimized loading
- `MPSModelLoader`: Apple Silicon MPS loading
- `MLXModelLoader`: MLX framework loading
- `CPUModelLoader`: CPU-optimized loading