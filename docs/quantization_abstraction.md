# Quantization Abstraction Documentation

The Quantization Abstraction Layer provides a unified interface for quantization across different backends (bitsandbytes, HQQ, MLX) in FSDP+QLoRA. It abstracts backend-specific differences and provides a consistent API for quantizing models.

## Overview

The quantization abstraction layer (`quantization_wrapper.py`) provides:

- **Unified Interface**: Common API for all quantization methods
- **Backend Adapters**: Specialized implementations for each backend
- **Automatic Selection**: Choose the best quantization method based on hardware
- **Configuration Validation**: Ensure compatibility between backend and quantization
- **Fallback Strategies**: Graceful degradation when quantization isn't supported

## Architecture

```
QuantizationConfig (dataclass)
    ↓
create_quantization_adapter(backend, config)
    ↓
QuantizationAdapter (ABC)
    ├── BitsAndBytesAdapter (CUDA only)
    ├── HQQAdapter (All backends)
    ├── MLXAdapter (Apple Silicon)
    ├── QuantoAdapter (Cross-platform)
    └── FallbackAdapter (No quantization)
```

## Key Components

### QuantizationMethod Enum

```python
class QuantizationMethod(enum.Enum):
    BNB_NF4 = "bnb_nf4"       # bitsandbytes Normal Float 4-bit
    BNB_INT8 = "bnb_int8"     # bitsandbytes INT8
    HQQ = "hqq"               # Half-Quadratic Quantization
    MLX_INT4 = "mlx_int4"     # MLX 4-bit integer
    MLX_INT8 = "mlx_int8"     # MLX 8-bit integer
    QUANTO_INT2 = "quanto_int2"  # Quanto 2-bit integer
    QUANTO_INT4 = "quanto_int4"  # Quanto 4-bit integer
    QUANTO_INT8 = "quanto_int8"  # Quanto 8-bit integer
    NONE = "none"             # No quantization
```

### QuantizationConfig

Comprehensive configuration for quantization:

```python
@dataclass
class QuantizationConfig:
    method: QuantizationMethod = QuantizationMethod.BNB_NF4
    bits: int = 4
    group_size: int = 64
    compute_dtype: Optional[torch.dtype] = None
    storage_dtype: Optional[torch.dtype] = None
    double_quant: bool = True
    quant_type: str = "nf4"
    skip_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    layer_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    backend_options: Dict[str, Any] = field(default_factory=dict)
```

### QuantizationAdapter

Abstract base class for backend-specific adapters:

```python
class QuantizationAdapter(ABC):
    @abstractmethod
    def create_quantized_linear(self, in_features: int, out_features: int, bias: bool = True) -> QuantizedLinear
    
    @abstractmethod
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module
    
    @abstractmethod
    def prepare_model_for_training(self, model: nn.Module) -> nn.Module
```

## Usage Examples

### Basic Quantization

```python
from quantization_wrapper import QuantizationConfig, create_quantization_adapter
from backend_manager import Backend

# Create configuration
config = QuantizationConfig(
    method=QuantizationMethod.BNB_NF4,
    bits=4,
    compute_dtype=torch.bfloat16
)

# Create adapter
adapter = create_quantization_adapter(Backend.CUDA, config)

# Quantize model
quantized_model = adapter.quantize_model(model)
```

### Automatic Configuration

```python
from quantization_wrapper import get_recommended_config

# Get recommended config based on hardware and memory
config = get_recommended_config(
    backend=Backend.MPS,
    model_size_b=7.0,  # 7B parameter model
    available_memory_gb=16.0
)
# Returns MLX_INT4 for Apple Silicon with limited memory
```

### Backend-Specific Examples

#### CUDA with bitsandbytes

```python
config = QuantizationConfig(
    method=QuantizationMethod.BNB_NF4,
    bits=4,
    double_quant=True,
    quant_type="nf4",
    compute_dtype=torch.bfloat16
)
adapter = create_quantization_adapter(Backend.CUDA, config)
```

#### Apple Silicon with MLX

```python
config = QuantizationConfig(
    method=QuantizationMethod.MLX_INT4,
    bits=4,
    compute_dtype=torch.float16  # MPS doesn't support bfloat16
)
adapter = create_quantization_adapter(Backend.MPS, config)
```

#### CPU with HQQ

```python
config = QuantizationConfig(
    method=QuantizationMethod.HQQ,
    bits=8,
    group_size=128,
    compute_dtype=torch.float32
)
adapter = create_quantization_adapter(Backend.CPU, config)
```

### Validation

```python
from quantization_wrapper import validate_quantization_config

# Check if configuration is valid for backend
issues = validate_quantization_config(config, Backend.MPS)
if issues:
    print("Configuration issues:", issues)
```

### Layer-Specific Quantization

```python
config = QuantizationConfig(
    method=QuantizationMethod.HQQ,
    bits=4,  # Default
    layer_configs={
        'transformer.layers.0': {'bits': 8},  # First layer uses 8-bit
        'transformer.layers.23': {'bits': 2}, # Last layer uses 2-bit
    }
)
```

## Integration with train.py

The quantization abstraction integrates seamlessly with the training script:

```python
# In train.py or your training script
from quantization_wrapper import (
    QuantizationConfig,
    QuantizationMethod,
    create_quantization_adapter,
    replace_linear_with_quantized
)

def setup_quantization(args, backend_manager):
    """Setup quantization based on CLI arguments."""
    
    # Map string arguments to enum
    method_map = {
        'bnb_nf4': QuantizationMethod.BNB_NF4,
        'bnb_int8': QuantizationMethod.BNB_INT8,
        'hqq': QuantizationMethod.HQQ,
        'mlx_int4': QuantizationMethod.MLX_INT4,
        'mlx_int8': QuantizationMethod.MLX_INT8,
        'none': QuantizationMethod.NONE,
    }
    
    # Create config from arguments
    config = QuantizationConfig(
        method=method_map.get(args.quantize, QuantizationMethod.NONE),
        bits=args.q_bits,
        group_size=args.q_group_size,
        compute_dtype=getattr(torch, args.q_compute_dtype),
        skip_modules=args.q_skip_modules.split(',') if args.q_skip_modules else ['lm_head'],
        double_quant=args.q_double_quant
    )
    
    # Validate configuration
    issues = validate_quantization_config(config, backend_manager.backend)
    if issues:
        for issue in issues:
            print(f"Warning: {issue}")
    
    # Create adapter
    return create_quantization_adapter(backend_manager.backend, config)
```

## Backend Compatibility Matrix

| Backend | BNB_NF4 | BNB_INT8 | HQQ | MLX_INT4 | MLX_INT8 | QUANTO_INT2 | QUANTO_INT4 | QUANTO_INT8 |
|---------|---------|----------|-----|----------|----------|-------------|-------------|-------------|
| CUDA    | ✓       | ✓        | ✓   | ✗        | ✗        | ✓           | ✓           | ✓           |
| MPS     | ✗       | ✗        | ✓   | ✓        | ✓        | ✓           | ✓           | ✓           |
| MLX     | ✗       | ✗        | ✓   | ✓        | ✓        | ✗           | ✗           | ✗           |
| CPU     | ✗       | ✗        | ✓   | ✗        | ✗        | ✓           | ✓           | ✓           |

## Memory Requirements

Approximate memory usage for different quantization methods:

- **No quantization (FP16)**: 2 bytes per parameter
- **8-bit quantization**: 1 byte per parameter
- **4-bit quantization**: 0.5 bytes per parameter
- **2-bit quantization**: 0.25 bytes per parameter

Example for a 7B parameter model:
- FP16: ~14 GB
- INT8: ~7 GB
- INT4: ~3.5 GB

## Advanced Features

### Custom Adapters

Create custom adapters for new quantization methods:

```python
class MyCustomAdapter(QuantizationAdapter):
    def _validate_backend_support(self):
        # Check if your method is supported
        pass
    
    def create_quantized_linear(self, in_features, out_features, bias=True):
        # Return your custom quantized layer
        return MyQuantizedLinear(in_features, out_features, bias, self.config)
    
    def quantize_model(self, model, **kwargs):
        # Implement model quantization
        return replace_linear_with_quantized(model, self)
```

### QLoRA Training

The quantization abstraction supports QLoRA (Quantized LoRA) training:

```python
# Quantize base model
adapter = create_quantization_adapter(backend, config)
quantized_model = adapter.quantize_model(model)

# Prepare for PEFT/LoRA training
training_model = adapter.prepare_model_for_training(quantized_model)

# Apply LoRA (using PEFT library)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
model = get_peft_model(training_model, lora_config)
```

### Model Loading and Saving

The quantization abstraction now supports loading pre-quantized models:

```python
# Save a quantized model
adapter = create_quantization_adapter(backend, config)
quantized_model = adapter.quantize_model(model)
adapter.save_quantized_model(quantized_model, "path/to/save")

# Load a quantized model
loaded_model = adapter.load_quantized_model("path/to/saved/model", model_config)
```

All adapters support loading from:
- Directory structures with separate config files
- Single checkpoint files with embedded configs
- Native format files (e.g., HQQ .pt files, MLX .npz files)

For detailed model loading documentation, see [Quantization Model Loading Guide](quantization_model_loading.md).

## Troubleshooting

### Common Issues

1. **"bitsandbytes not available" on non-CUDA**
   - Solution: Use HQQ or MLX quantization instead
   - The system will automatically fallback

2. **"MPS doesn't support bfloat16"**
   - Solution: Use float16 as compute_dtype
   - Config will auto-adjust if not specified

3. **"MLX quantization requires model conversion"**
   - Solution: This is expected, MLX uses a different tensor format
   - The wrapper handles basic compatibility

### Debug Information

Enable verbose logging:

```python
# Get detailed information about quantization
from quantization_wrapper import validate_quantization_config
from imports import report_import_status

# Check available quantization methods
print(report_import_status(backend='cuda'))

# Validate configuration
issues = validate_quantization_config(config, backend)
for issue in issues:
    print(f"Config issue: {issue}")
```

## Performance Tips

1. **Use appropriate bit width**: 4-bit is usually sufficient for inference, 8-bit for training
2. **Enable double quantization**: Reduces memory further with minimal quality loss
3. **Skip critical layers**: Keep lm_head and embeddings in full precision
4. **Group size tuning**: Smaller groups (32-64) for better quality, larger (128-256) for speed

## Future Extensions

The quantization abstraction is designed to be extensible:

- Add new quantization methods by creating new adapters
- Support for dynamic quantization
- Integration with other PEFT methods beyond LoRA
- Automatic mixed-precision quantization
- Quantization-aware training support