# Integration Test Suite Documentation

This document describes the comprehensive integration test suite created for the FSDP QLoRA project.

## Overview

The integration test suite consists of several test modules designed to ensure end-to-end functionality across different backends, quantization methods, and training types.

## Test Structure

### 1. Test Utilities (`tests/test_utils.py`)

Core utilities for integration testing:

- **`MemoryStats`**: Dataclass for tracking memory usage during tests
- **`DummyDataset`**: Simple dataset for testing without real data
- **`TinyLlamaConfig`**: Small LLaMA configuration for fast testing
- **`memory_tracker`**: Context manager for tracking memory usage
- **`create_tiny_model`**: Creates small models for testing
- **`check_convergence`**: Verifies training is converging
- **`create_test_tokenizer`**: Mock tokenizer for tests
- **`skip_if_backend_unavailable`**: Decorator to skip tests based on backend availability

### 2. Backend Integration Tests (`tests/test_backend_integration_comprehensive.py`)

Tests backend-specific functionality:

- **CUDA backend tests**: Model loading and quantization with BitsAndBytes
- **MPS backend tests**: Apple Silicon support with MLX/Quanto quantization
- **CPU backend tests**: HQQ quantization support
- **Memory efficiency tests**: Compares memory usage across backends
- **Quantization compatibility**: Matrix of which methods work with which backends
- **Cross-backend transfer**: Tests moving models between devices

### 3. Training Integration Tests (`tests/test_training_integration_comprehensive.py`)

Comprehensive training tests for all training types:

- **LoRA convergence**: Standard LoRA training
- **QLoRA with quantization**: 4-bit quantized training
- **Custom implementations**: Tests custom LoRA/QLoRA
- **Memory efficiency comparison**: Compares full vs LoRA training
- **Gradient accumulation**: Tests with different batch sizes
- **Mixed precision**: Tests fp32, fp16, bf16
- **Dataset handling**: Different dataset configurations
- **Optimizer tests**: AdamW, Adam, SGD
- **Learning rate schedulers**: Constant, linear, cosine

## Running the Tests

### Run all integration tests:
```bash
python -m pytest tests/test_backend_integration_comprehensive.py tests/test_training_integration_comprehensive.py -v
```

### Run specific backend tests:
```bash
# Test MPS backend
python -m pytest tests/test_backend_integration_comprehensive.py::TestBackendIntegrationComprehensive::test_mps_model_loading_and_quantization -v

# Test memory efficiency
python -m pytest tests/test_backend_integration_comprehensive.py::TestBackendIntegrationComprehensive::test_backend_memory_efficiency -v
```

### Run training convergence tests:
```bash
# Test LoRA training
python -m pytest tests/test_training_integration_comprehensive.py::TestTrainingIntegrationComprehensive::test_lora_training_convergence -v

# Test memory comparison
python -m pytest tests/test_training_integration_comprehensive.py::TestTrainingIntegrationComprehensive::test_training_memory_efficiency_comparison -v
```

## Test Requirements

The tests use mocking to avoid downloading real models, but some tests require specific libraries:

- **CUDA tests**: Require NVIDIA GPU and CUDA toolkit
- **MPS tests**: Require Apple Silicon Mac
- **MLX tests**: Require `mlx` library (Apple Silicon)
- **Quantization tests**: May require `bitsandbytes`, `hqq`, or `quanto`

Tests will automatically skip if required backends/libraries are not available.

## Key Test Patterns

### 1. Backend-Aware Testing
Tests automatically detect available backends and skip incompatible tests:
```python
@skip_if_backend_unavailable(Backend.CUDA)
def test_cuda_specific_feature(self):
    # This test only runs on CUDA systems
```

### 2. Memory Tracking
Tests track memory usage to ensure efficiency:
```python
with memory_tracker(backend_manager) as mem_stats:
    # Perform operations
    model(input_data)
# Check mem_stats.total_used_mb
```

### 3. Convergence Testing
Tests verify that training actually reduces loss:
```python
losses = run_mini_training(model, dataloader, optimizer)
self.assertTrue(check_convergence(losses))
```

### 4. Mock-Based Testing
Tests use mocks to avoid external dependencies:
```python
with patch('train.AutoModelForCausalLM') as mock_model:
    # Test without downloading real models
```

## Common Issues and Solutions

### Issue 1: Import Errors
If you see import errors, ensure you're in the virtual environment:
```bash
source venv/bin/activate
```

### Issue 2: Backend Not Available
Tests will skip if backends aren't available. To test all backends:
- CUDA: Requires NVIDIA GPU
- MPS: Requires macOS with Apple Silicon
- MLX: Install with `pip install mlx`

### Issue 3: Quantization Library Missing
Install required quantization libraries:
```bash
pip install bitsandbytes  # For CUDA
pip install hqq-python    # For HQQ
pip install optimum-quanto # For Quanto
```

## Test Coverage

The test suite covers:
- ✅ All backend types (CUDA, MPS, MLX, CPU)
- ✅ All quantization methods (BnB, HQQ, MLX, Quanto)
- ✅ All training types (full, lora, qlora, custom variants)
- ✅ Memory efficiency tracking
- ✅ Training convergence verification
- ✅ Cross-backend compatibility
- ✅ Configuration validation

## Future Improvements

1. Add distributed training tests (multi-GPU)
2. Add checkpoint save/load tests
3. Add more realistic convergence tests with real datasets
4. Add performance benchmarking tests
5. Add integration with CI/CD pipeline

## Contributing

When adding new tests:
1. Use the provided test utilities for consistency
2. Add appropriate skip decorators for backend-specific tests
3. Mock external dependencies to keep tests fast
4. Track memory usage for performance-critical code
5. Verify training convergence for new training methods