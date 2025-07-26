# Import Abstraction Layer Documentation

The Import Abstraction Layer provides a flexible, backend-aware system for managing conditional imports in FSDP+QLoRA. It ensures that optional dependencies are handled gracefully and provides appropriate fallbacks when libraries are not available.

## Overview

The import abstraction layer (`imports.py`) provides:

- **Conditional Imports**: Try-except wrapped imports for optional dependencies
- **Fallback Mechanisms**: Dummy modules when libraries aren't available
- **Backend-Specific Registry**: Different imports for different backends (CUDA, MPS, MLX, CPU)
- **Import Validation**: Check availability and validate imported modules
- **Error Reporting**: Clear messages about missing dependencies

## Key Components

### ImportResult

A dataclass that encapsulates the result of an import attempt:

```python
@dataclass
class ImportResult:
    success: bool
    module: Optional[Any] = None
    error: Optional[Exception] = None
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)
```

### ImportRegistry

The central registry that manages all conditional imports:

```python
registry = ImportRegistry()

# Register an import with fallback
registry.register(
    'bitsandbytes',
    import_func=_import_bitsandbytes,
    fallback=_bitsandbytes_fallback,
    validator=_validate_bitsandbytes,
    backends=['cuda']  # Only for CUDA
)
```

## Usage

### Basic Import

```python
from imports import get_module

# Get a module with automatic fallback handling
bnb = get_module('bitsandbytes', backend='cuda')
```

### Checking Availability

```python
from imports import check_import_availability

if check_import_availability('wandb'):
    # Use wandb features
    pass
else:
    # Use alternative logging
    pass
```

### Validating Multiple Imports

```python
from imports import validate_imports

results = validate_imports(['bitsandbytes', 'wandb', 'hqq'], backend='mps')
for name, result in results.items():
    if not result.success:
        print(f"Warning: {name} not available: {result.error}")
```

### Generating Import Status Report

```python
from imports import report_import_status

print(report_import_status(backend='cuda'))
```

Output:
```
Import Status Report
==================================================
Backend: cuda

bitsandbytes         ✓ Available
dora                 ✓ Available
flash_attn           ✗ Not Available
  ✗ No module named 'flash_attn'
hqq                  ✓ Available (fallback)
  ⚠ Using fallback for 'hqq': No module named 'hqq'
lora                 ✓ Available
wandb                ✓ Available
```

## Supported Libraries

### bitsandbytes
- **Backends**: CUDA only
- **Fallback**: Dummy module with basic structure
- **Use Case**: 4-bit and 8-bit quantization

### HQQ
- **Backends**: All
- **Fallback**: Dummy module with basic classes
- **Use Case**: Alternative quantization method

### wandb
- **Backends**: All
- **Fallback**: No-op logging functions
- **Use Case**: Experiment tracking and logging

### Flash Attention
- **Backends**: CUDA only
- **Fallback**: Raises error if used
- **Use Case**: Optimized attention computation

### DORA/LoRA
- **Backends**: All
- **Fallback**: None (local modules)
- **Use Case**: Parameter-efficient fine-tuning

## Integration with train.py

The import abstraction is integrated into `train.py` through the `load_conditional_imports` function:

```python
def load_conditional_imports(backend_str):
    """Load conditional imports based on the detected backend."""
    global bnb, Linear4bit, Params4bit, BNBDORA, HQQDORA, DORALayer, MagnitudeLayer, LORA
    
    # Import bitsandbytes
    bnb_module = get_module('bitsandbytes', backend_str)
    bnb = bnb_module
    Linear4bit = bnb_module.Linear4bit
    Params4bit = bnb_module.Params4bit
    
    # Import DORA modules
    try:
        dora_module = get_module('dora', backend_str)
        BNBDORA = dora_module.BNBDORA
        # ... other DORA imports
    except ImportError as e:
        print(f"Warning: DORA modules not available: {e}")
```

This function is called after backend detection in `fsdp_qlora`:

```python
# Initialize backend manager
backend_manager = BackendManager(backend=backend_arg, verbose=verbose)

# Load conditional imports based on detected backend
load_conditional_imports(str(backend_manager.backend))
```

## Adding New Conditional Imports

To add a new conditional import:

1. **Create Import Function**:
```python
def _import_mylib():
    """Import mylib with any necessary setup."""
    import mylib
    # Any setup code
    return mylib
```

2. **Create Fallback (Optional)**:
```python
def _mylib_fallback():
    """Fallback for when mylib is not available."""
    class DummyMyLib:
        def __getattr__(self, name):
            raise ImportError(f"mylib.{name} is not available")
    return DummyMyLib()
```

3. **Register the Import**:
```python
_import_registry.register(
    'mylib',
    _import_mylib,
    fallback=_mylib_fallback,
    backends=['cuda', 'mps']  # Specify supported backends
)
```

4. **Use in Code**:
```python
mylib = get_module('mylib', backend='cuda')
```

## Best Practices

1. **Always Provide Fallbacks**: For optional dependencies, provide meaningful fallbacks
2. **Backend Awareness**: Specify which backends support each import
3. **Clear Error Messages**: Ensure fallbacks provide clear error messages when features are used
4. **Cache Results**: The registry automatically caches import results for performance
5. **Validate Early**: Use `validate_imports` at startup to check all required dependencies

## Error Handling

The import system provides several levels of error handling:

1. **Import Errors**: Caught and stored in ImportResult
2. **Fallback Errors**: If fallback also fails, both errors are reported
3. **Backend Mismatch**: Clear warnings when imports aren't supported on a backend
4. **Validation Failures**: Custom validators can enforce additional requirements

## Testing

The import abstraction layer includes comprehensive tests:

- Unit tests for the registry and import mechanisms (`test_imports.py`)
- Integration tests with train.py (`test_train_imports_integration.py`)
- Backend-specific behavior tests
- Performance tests for caching

Run tests with:
```bash
pytest tests/test_imports.py tests/test_train_imports_integration.py -v
```