# MLX Utilities Guide

Comprehensive utilities for working with MLX on Apple Silicon, providing efficient dataset conversion, memory profiling, and performance monitoring for FSDP QLoRA training.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Conversion](#dataset-conversion)
3. [Tokenizer Integration](#tokenizer-integration)
4. [Memory Profiling](#memory-profiling)
5. [Performance Monitoring](#performance-monitoring)
6. [Helper Utilities](#helper-utilities)
7. [Best Practices](#best-practices)
8. [API Reference](#api-reference)

## Overview

The MLX utilities module provides essential tools for:

- **Dataset Conversion**: Convert PyTorch/HuggingFace datasets to MLX arrays
- **Tokenizer Integration**: Seamless tokenization with MLX output
- **Memory Profiling**: Track unified memory usage on Apple Silicon
- **Performance Monitoring**: Benchmark and optimize MLX operations
- **Helper Functions**: Model size estimation, batch size optimization

### Key Features

- Optimized for Apple Silicon unified memory architecture
- Based on MLX LoRA fine-tuning examples
- Comprehensive profiling and monitoring tools
- Support for multiple dataset formats
- Efficient batch processing

## Dataset Conversion

### Basic Conversions

Convert various data formats to MLX arrays:

```python
from mlx_utils import DatasetConverter

# PyTorch tensor to MLX
tensor = torch.randn(batch_size, seq_length, hidden_size)
mlx_array = DatasetConverter.torch_to_mlx(tensor)

# NumPy array to MLX
np_array = np.random.randn(batch_size, seq_length)
mlx_array = DatasetConverter.numpy_to_mlx(np_array)

# Dictionary of tensors to MLX
batch = {
    "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)),
    "attention_mask": torch.ones(batch_size, seq_length),
    "labels": torch.randint(0, vocab_size, (batch_size, seq_length)),
}
mlx_batch = DatasetConverter.dict_to_mlx(batch)
```

### HuggingFace Dataset Conversion

Convert HuggingFace datasets for MLX training:

```python
from datasets import load_dataset
from mlx_utils import HuggingFaceDatasetConverter

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

# Create converter
converter = HuggingFaceDatasetConverter(tokenizer)

# Convert to MLX format
mlx_data = converter.convert_dataset(
    dataset,
    text_field="text",  # or automatically handles instruction/input/output
    max_length=2048,
    padding="max_length",
    truncation=True,
)
```

### MLX DataLoader

Efficient data loading with MLX arrays:

```python
from mlx_utils import MLXDataLoader

# Create dataloader
dataloader = MLXDataLoader(
    mlx_data,
    batch_size=4,
    shuffle=True,
    drop_last=True,
)

# Iterate through batches
for batch in dataloader:
    # batch contains stacked MLX arrays
    input_ids = batch["input_ids"]  # Shape: (batch_size, seq_length)
    labels = batch["labels"]
```

### Complete Pipeline

End-to-end dataset processing:

```python
from mlx_utils import create_mlx_dataloader

# Automatically handles conversion and creates dataloader
dataloader = create_mlx_dataloader(
    dataset,  # HuggingFace, PyTorch, or list of dicts
    tokenizer,
    batch_size=4,
    max_length=2048,
    shuffle=True,
)
```

## Tokenizer Integration

### MLX Tokenizer Wrapper

Tokenize directly to MLX arrays:

```python
from transformers import AutoTokenizer
from mlx_utils import MLXTokenizer

# Wrap HuggingFace tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
mlx_tokenizer = MLXTokenizer(tokenizer)

# Single text tokenization
tokens = mlx_tokenizer(
    "Fine-tune models efficiently on Apple Silicon!",
    max_length=128,
    padding="max_length",
    truncation=True,
)
# tokens["input_ids"] is mx.array

# Batch tokenization
texts = ["First sentence.", "Second sentence.", "Third sentence."]
batch_tokens = mlx_tokenizer(texts, max_length=128)

# Decode back to text
decoded = mlx_tokenizer.decode(tokens["input_ids"][0])
```

## Memory Profiling

### Basic Memory Stats

Get current memory statistics:

```python
from mlx_utils import MemoryProfiler

profiler = MemoryProfiler()
stats = profiler.get_memory_stats()

print(f"Process Memory: {stats.process_rss_gb:.2f} GB")
print(f"Available Memory: {stats.available_memory_gb:.2f} GB")
print(f"Total Memory: {stats.total_memory_gb:.2f} GB")
```

### Profile Operations

Track memory usage during operations:

```python
# Profile specific operations
with profiler.profile("Model Loading"):
    model = load_model()

with profiler.profile("Dataset Preparation"):
    dataset = prepare_dataset()

# Get profiling summary
print(profiler.get_summary())
```

Output:
```
Memory Profiling Summary:
--------------------------------------------------
Model Loading:
  Duration: 2.34s
  Memory: 2.10 GB → 6.50 GB
  Delta: +4.40 GB
--------------------------------------------------
Dataset Preparation:
  Duration: 0.89s
  Memory: 6.50 GB → 7.20 GB
  Delta: +0.70 GB
--------------------------------------------------
```

### Continuous Monitoring

Monitor memory usage over time:

```python
# Start continuous monitoring
profiler.start_monitoring(interval=1.0)  # Sample every second

# Run your training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        train_step(batch)

# Stop monitoring
profiler.stop_monitoring()

# Plot memory usage (requires matplotlib)
profiler.plot_memory_usage(save_path="memory_usage.png")
```

## Performance Monitoring

### Benchmark Operations

Measure performance of MLX operations:

```python
from mlx_utils import PerformanceMonitor

monitor = PerformanceMonitor()

# Benchmark training steps
with monitor.benchmark(
    num_samples=1000,
    num_tokens=1000 * 512,  # samples * seq_length
    label="Training Epoch"
):
    for batch in dataloader:
        loss = train_step(batch)

# Get metrics
metrics = monitor.get_latest_metrics()
print(f"Throughput: {metrics.tokens_per_second:.0f} tokens/sec")
print(f"Samples/sec: {metrics.samples_per_second:.1f}")
print(f"Peak memory: {metrics.peak_memory_gb:.2f} GB")
```

### Track MLX Compilation

Separate compilation from computation time:

```python
with monitor.benchmark(num_samples=100, label="With Compilation"):
    # Mark compilation phase
    monitor.mark_compile_start()
    model.compile()  # MLX graph compilation
    monitor.mark_compile_end()
    
    # Run computation
    for i in range(100):
        output = model(input_batch)

metrics = monitor.get_latest_metrics()
print(f"Compilation time: {metrics.compile_time:.2f}s ({metrics.compile_time/metrics.total_time*100:.1f}%)")
print(f"Compute time: {metrics.compute_time:.2f}s")
```

### Compare Benchmarks

Compare different approaches:

```python
# Benchmark different batch sizes
for batch_size in [1, 2, 4, 8]:
    with monitor.benchmark(
        num_samples=100 * batch_size,
        num_tokens=100 * batch_size * 512,
        label=f"Batch Size {batch_size}"
    ):
        run_training(batch_size)

# Compare results
print(monitor.compare_benchmarks())
```

Output:
```
Benchmark Comparison:
================================================================================
Label                Time (s)   Samples/s   Tokens/s    Memory (GB)
--------------------------------------------------------------------------------
Batch Size 1         10.23      9.78        5007.68     4.50
Batch Size 2         11.45      17.47       17855.04    5.20
Batch Size 4         13.67      29.26       59911.04    6.80
Batch Size 8         18.92      42.28       173137.92   10.30
```

### Profile Individual Operations

Fine-grained profiling:

```python
from mlx_utils import profile_mlx_operation

def matrix_multiply(a, b):
    return mx.matmul(a, b)

# Create test matrices
a = mx.random.normal((1000, 1000))
b = mx.random.normal((1000, 1000))

# Profile with warmup
stats = profile_mlx_operation(
    matrix_multiply,
    a, b,
    num_warmup=5,
    num_runs=20,
)

print(f"Average time: {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s")
print(f"Min/Max: {stats['min_time']:.4f}s / {stats['max_time']:.4f}s")
```

## Helper Utilities

### Model Size Estimation

Estimate memory requirements:

```python
from mlx_utils import estimate_model_size

# 7B model with 4-bit quantization
model_size = estimate_model_size(
    num_parameters=7e9,
    bits=4,
    include_gradients=False,  # LoRA only trains adapters
    include_optimizer_states=True,
)
print(f"Estimated memory: {model_size:.1f} GB")

# Compare different configurations
configs = [
    (16, True, True),   # FP16 full training
    (8, True, True),    # INT8 full training
    (4, False, True),   # INT4 LoRA training
]

for bits, grad, opt in configs:
    size = estimate_model_size(7e9, bits, grad, opt)
    print(f"{bits}-bit: {size:.1f} GB")
```

### Optimal Batch Size

Calculate optimal batch size for available memory:

```python
from mlx_utils import get_optimal_batch_size

# For 7B model on M2 Max (32GB)
batch_size = get_optimal_batch_size(
    model_size_gb=3.5,  # 7B model at 4-bit
    sequence_length=2048,
    available_memory_gb=32.0,
    safety_factor=0.8,  # Use 80% of available memory
)
print(f"Recommended batch size: {batch_size}")
```

### Device Information

Check MLX and Apple Silicon details:

```python
from mlx_utils import check_mlx_device

device_info = check_mlx_device()
print(f"MLX Available: {device_info['available']}")
print(f"Chip Series: {device_info.get('chip_series', 'Unknown')}")
print(f"Total Memory: {device_info.get('total_memory_gb', 0):.1f} GB")
```

### Memory Formatting

Human-readable memory sizes:

```python
from mlx_utils import format_memory_size

print(format_memory_size(7e9 * 4 / 8))  # "3.50 GB"
print(format_memory_size(1024))         # "1.00 KB"
```

## Best Practices

### 1. Dataset Conversion

- Convert datasets once and cache if possible
- Use appropriate max_length for your model
- Consider memory when choosing batch sizes

```python
# Good: Convert once
mlx_data = converter.convert_dataset(dataset)
dataloader = MLXDataLoader(mlx_data, batch_size=4)

# Avoid: Converting in loop
for epoch in range(epochs):
    # Don't convert every epoch!
    mlx_data = converter.convert_dataset(dataset)
```

### 2. Memory Management

- Profile before and after major operations
- Monitor continuously during training
- Set conservative batch sizes

```python
# Profile major operations
with profiler.profile("Critical Operation"):
    result = expensive_operation()
    
# Check if memory is growing
if profiler.history[-1]["memory_delta_gb"] > 1.0:
    print("Warning: Large memory increase detected!")
```

### 3. Performance Optimization

- Warm up MLX operations before benchmarking
- Track compilation separately from computation
- Compare different approaches systematically

```python
# Always warm up MLX operations
for _ in range(3):
    _ = model(dummy_input)
    mx.eval()  # Force evaluation

# Then benchmark
with monitor.benchmark(num_samples=1000):
    for batch in dataloader:
        output = model(batch)
```

### 4. Error Handling

- Check MLX availability before use
- Handle missing dependencies gracefully
- Provide fallbacks when possible

```python
from mlx_utils import MLX_AVAILABLE

if not MLX_AVAILABLE:
    print("MLX not available. Install with: pip install mlx mlx-lm")
    # Use alternative approach or exit
else:
    # Proceed with MLX operations
    mlx_array = DatasetConverter.torch_to_mlx(tensor)
```

## API Reference

### DatasetConverter

Static methods for data conversion:

- `torch_to_mlx(tensor: torch.Tensor) -> mx.array`
- `numpy_to_mlx(array: np.ndarray) -> mx.array`
- `mlx_to_torch(array: mx.array, device: Optional[str]) -> torch.Tensor`
- `dict_to_mlx(data: Dict[str, Any]) -> Dict[str, Any]`

### HuggingFaceDatasetConverter

Convert HuggingFace datasets:

```python
converter = HuggingFaceDatasetConverter(tokenizer)
mlx_data = converter.convert_dataset(
    dataset,
    text_field="text",
    max_length=512,
    padding="max_length",
    truncation=True,
)
```

### MLXDataLoader

DataLoader for MLX arrays:

```python
dataloader = MLXDataLoader(
    data: List[Dict[str, mx.array]],
    batch_size: int = 1,
    shuffle: bool = False,
    drop_last: bool = False,
)
```

### MemoryProfiler

Memory profiling tools:

```python
profiler = MemoryProfiler()

# Get current stats
stats = profiler.get_memory_stats()

# Profile operations
with profiler.profile("Operation Name"):
    # Your code here

# Continuous monitoring
profiler.start_monitoring(interval=1.0)
profiler.stop_monitoring()

# Get summary
summary = profiler.get_summary()
```

### PerformanceMonitor

Performance benchmarking:

```python
monitor = PerformanceMonitor()

# Benchmark operations
with monitor.benchmark(
    num_samples: int,
    num_tokens: Optional[int] = None,
    label: str = "",
):
    # Your code here

# Track compilation
monitor.mark_compile_start()
monitor.mark_compile_end()

# Get results
metrics = monitor.get_latest_metrics()
comparison = monitor.compare_benchmarks()
```

## Examples

See the [examples directory](../examples/) for complete demonstrations:

- `mlx_utilities_demo.py`: Comprehensive demo of all features
- `mlx_complete_training.py`: Full training example using utilities
- `mlx_training_example.py`: Simple training setup

## Troubleshooting

### MLX Not Available

If MLX is not installed:

```bash
pip install mlx mlx-lm
```

### Memory Errors

If encountering memory errors:

1. Reduce batch size
2. Use more aggressive quantization (4-bit)
3. Enable gradient checkpointing
4. Profile to find memory leaks

### Performance Issues

If performance is lower than expected:

1. Check for compilation overhead
2. Ensure proper batch sizes
3. Profile individual operations
4. Verify MLX is using Metal acceleration

## Contributing

Contributions to improve MLX utilities are welcome! Please ensure:

1. Add tests for new features
2. Update documentation
3. Follow existing code style
4. Benchmark performance impact