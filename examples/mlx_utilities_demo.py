"""
MLX Utilities Demonstration

This script demonstrates the comprehensive features of MLX utilities for:
- Dataset conversion from various formats
- Tokenizer integration
- Memory profiling during model operations
- Performance benchmarking
- Real-world usage scenarios
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MLX utilities
from mlx_utils import (
    DatasetConverter,
    HuggingFaceDatasetConverter,
    MLXDataLoader,
    MLXTokenizer,
    MemoryProfiler,
    PerformanceMonitor,
    estimate_model_size,
    get_optimal_batch_size,
    format_memory_size,
    check_mlx_device,
    create_mlx_dataloader,
    profile_mlx_operation,
)

# Check if MLX is available
try:
    import mlx
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. Install with: pip install mlx mlx-lm")


def demo_dataset_conversion():
    """Demonstrate dataset conversion capabilities."""
    print("\n" + "=" * 60)
    print("DATASET CONVERSION DEMO")
    print("=" * 60)
    
    # 1. PyTorch Tensor Conversion
    print("\n1. PyTorch Tensor to MLX Array:")
    tensor = torch.randn(2, 3, 4)
    print(f"   Original tensor shape: {tensor.shape}")
    
    if MLX_AVAILABLE:
        mlx_array = DatasetConverter.torch_to_mlx(tensor)
        print(f"   MLX array shape: {mlx_array.shape}")
    else:
        print("   [MLX not available - skipping conversion]")
    
    # 2. Dictionary Conversion
    print("\n2. Dictionary of Tensors to MLX:")
    batch = {
        "input_ids": torch.randint(0, 1000, (4, 128)),
        "attention_mask": torch.ones(4, 128),
        "labels": torch.randint(0, 1000, (4, 128)),
        "metadata": {"task": "translation"},
    }
    
    if MLX_AVAILABLE:
        mlx_batch = DatasetConverter.dict_to_mlx(batch)
        print("   Converted keys:", list(mlx_batch.keys()))
        print(f"   input_ids shape: {mlx_batch['input_ids'].shape}")
    else:
        print("   [MLX not available - skipping conversion]")
    
    # 3. Custom Dataset Class
    print("\n3. Custom PyTorch Dataset:")
    
    class DemoDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 1000, (128,)),
                "labels": torch.randint(0, 1000, (128,)),
            }
    
    dataset = DemoDataset(size=50)
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Sample item keys: {list(dataset[0].keys())}")


def demo_tokenizer_integration():
    """Demonstrate MLX tokenizer integration."""
    print("\n" + "=" * 60)
    print("TOKENIZER INTEGRATION DEMO")
    print("=" * 60)
    
    # Mock tokenizer for demonstration
    from transformers import AutoTokenizer
    
    try:
        # Try to load a real tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("\nUsing GPT-2 tokenizer")
    except:
        # Fallback to mock
        print("\nUsing mock tokenizer")
        tokenizer = None
    
    if tokenizer and MLX_AVAILABLE:
        # Create MLX tokenizer
        mlx_tokenizer = MLXTokenizer(tokenizer)
        
        # Single text
        text = "MLX makes training on Apple Silicon efficient!"
        tokens = mlx_tokenizer(text, max_length=20)
        print(f"\nSingle text tokenization:")
        print(f"  Input: '{text}'")
        print(f"  Token shape: {tokens['input_ids'].shape}")
        
        # Batch tokenization
        texts = [
            "First sentence for batch processing.",
            "Second sentence is a bit longer than the first.",
            "Third one is short.",
        ]
        batch_tokens = mlx_tokenizer(texts, max_length=20, padding="max_length")
        print(f"\nBatch tokenization:")
        print(f"  Batch size: {len(texts)}")
        print(f"  Batch shape: {batch_tokens['input_ids'].shape}")
        
        # Decode example
        decoded = mlx_tokenizer.decode(tokens['input_ids'][0])
        print(f"\nDecoded text: '{decoded}'")
    else:
        print("\n[Tokenizer demo requires MLX and transformers]")


def demo_huggingface_dataset():
    """Demonstrate HuggingFace dataset conversion."""
    print("\n" + "=" * 60)
    print("HUGGINGFACE DATASET CONVERSION DEMO")
    print("=" * 60)
    
    # Create mock Alpaca-style dataset
    class MockAlpacaDataset:
        def __init__(self, size=10):
            self.data = []
            for i in range(size):
                self.data.append({
                    "instruction": f"Question {i}: What is the capital of France?",
                    "input": "",
                    "output": "The capital of France is Paris.",
                })
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def __iter__(self):
            return iter(self.data)
    
    # Mock tokenizer
    mock_tokenizer = lambda text, **kwargs: {
        "input_ids": np.random.randint(0, 1000, (1, kwargs.get('max_length', 128))),
        "attention_mask": np.ones((1, kwargs.get('max_length', 128))),
    }
    
    dataset = MockAlpacaDataset(size=20)
    converter = HuggingFaceDatasetConverter(mock_tokenizer)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Sample instruction: '{dataset[0]['instruction']}'")
    
    if MLX_AVAILABLE:
        # Convert dataset
        mlx_data = converter.convert_dataset(dataset, max_length=256)
        print(f"\nConverted {len(mlx_data)} samples to MLX format")
        
        # Create MLX DataLoader
        dataloader = MLXDataLoader(mlx_data, batch_size=4, shuffle=True)
        print(f"Created MLX DataLoader with {len(dataloader)} batches")
        
        # Show first batch info
        first_batch = next(iter(dataloader))
        print(f"\nFirst batch shapes:")
        for key, value in first_batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
    else:
        print("\n[MLX not available - skipping conversion]")


def demo_memory_profiling():
    """Demonstrate memory profiling capabilities."""
    print("\n" + "=" * 60)
    print("MEMORY PROFILING DEMO")
    print("=" * 60)
    
    profiler = MemoryProfiler()
    
    # Get current memory stats
    current_stats = profiler.get_memory_stats()
    print(f"\nCurrent Memory Status:")
    print(current_stats)
    
    # Profile different operations
    print("\nProfiling memory usage for different operations:")
    
    # 1. Array allocation
    with profiler.profile("NumPy Array Allocation (100MB)"):
        # Allocate ~100MB
        array = np.random.randn(25_000_000)  # 25M * 8 bytes = 200MB
    
    # 2. Dictionary creation
    with profiler.profile("Large Dictionary Creation"):
        large_dict = {f"key_{i}": np.random.randn(1000) for i in range(100)}
    
    # 3. Cleanup
    with profiler.profile("Memory Cleanup"):
        del array
        del large_dict
        import gc
        gc.collect()
    
    # Show profiling summary
    print(profiler.get_summary())
    
    # Continuous monitoring example
    print("\nStarting continuous memory monitoring for 2 seconds...")
    profiler.start_monitoring(interval=0.5)
    
    # Simulate some work
    import time
    for i in range(4):
        # Allocate and free memory
        temp = np.random.randn(10_000_000)  # 80MB
        time.sleep(0.5)
        del temp
    
    profiler.stop_monitoring()
    
    # Get monitoring data
    monitoring_entries = [h for h in profiler.history if "timestamp" in h]
    if monitoring_entries:
        print(f"Collected {len(monitoring_entries)} monitoring samples")
        
        # Show memory fluctuation
        memory_values = [h["memory_gb"] for h in monitoring_entries]
        print(f"Memory range: {min(memory_values):.2f} - {max(memory_values):.2f} GB")


def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "=" * 60)
    print("PERFORMANCE MONITORING DEMO")
    print("=" * 60)
    
    monitor = PerformanceMonitor()
    
    # Benchmark different operations
    print("\nBenchmarking various operations:")
    
    # 1. Matrix multiplication
    def matrix_multiply(size=1000):
        a = np.random.randn(size, size)
        b = np.random.randn(size, size)
        return np.matmul(a, b)
    
    with monitor.benchmark(num_samples=10, label="Matrix Multiply (1000x1000)"):
        for _ in range(10):
            result = matrix_multiply(1000)
    
    # 2. Array operations
    def array_operations(size=10_000_000):
        a = np.random.randn(size)
        b = np.random.randn(size)
        return np.sin(a) + np.cos(b)
    
    with monitor.benchmark(num_samples=5, label="Array Operations (10M elements)"):
        for _ in range(5):
            result = array_operations()
    
    # 3. Simulated training step
    def simulated_training_step(batch_size=32, seq_length=512):
        # Simulate forward pass
        inputs = np.random.randn(batch_size, seq_length, 768)  # 768 = hidden size
        weights = np.random.randn(768, 768)
        
        # Simulate operations
        hidden = np.matmul(inputs, weights)
        output = np.tanh(hidden)
        loss = np.mean(output ** 2)
        
        return loss
    
    with monitor.benchmark(
        num_samples=100,
        num_tokens=100 * 32 * 512,  # steps * batch * seq_length
        label="Simulated Training"
    ):
        # Simulate compilation time
        monitor.mark_compile_start()
        import time
        time.sleep(0.05)  # 50ms "compilation"
        monitor.mark_compile_end()
        
        # Run training steps
        for _ in range(100):
            loss = simulated_training_step()
    
    # Show performance summary
    print(monitor.get_summary())
    
    # Compare benchmarks
    print("\n" + monitor.compare_benchmarks())


def demo_model_optimization():
    """Demonstrate model size estimation and optimization."""
    print("\n" + "=" * 60)
    print("MODEL OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Model configurations
    models = [
        ("LLaMA-7B", 7e9),
        ("LLaMA-13B", 13e9),
        ("LLaMA-70B", 70e9),
    ]
    
    quantization_configs = [
        ("FP16", 16, False, False),
        ("INT8", 8, False, False),
        ("INT4", 4, False, False),
        ("INT4 + LoRA", 4, True, False),  # LoRA trains adapters only
        ("INT4 + Training", 4, True, True),
    ]
    
    print("\nModel Memory Requirements:")
    print("-" * 80)
    print(f"{'Model':<15} {'Config':<20} {'Size':<15} {'Notes':<30}")
    print("-" * 80)
    
    for model_name, num_params in models:
        for config_name, bits, include_grad, include_opt in quantization_configs:
            size_gb = estimate_model_size(
                num_params,
                bits=bits,
                include_gradients=include_grad,
                include_optimizer_states=include_opt,
            )
            
            # Determine notes
            notes = []
            if size_gb > 128:
                notes.append("Requires M2/M3 Ultra (192GB)")
            elif size_gb > 64:
                notes.append("Requires M1/M2 Ultra (128GB)")
            elif size_gb > 32:
                notes.append("Requires M1/M2 Max (64GB)")
            elif size_gb > 16:
                notes.append("Requires M1/M2 Pro (32GB)")
            else:
                notes.append("Works on M1/M2 (16GB)")
            
            print(f"{model_name:<15} {config_name:<20} {format_memory_size(size_gb * 1e9):<15} {', '.join(notes):<30}")
    
    # Batch size recommendations
    print("\n\nOptimal Batch Size Recommendations:")
    print("-" * 60)
    
    memory_configs = [16, 32, 64, 128]  # GB
    
    for memory_gb in memory_configs:
        print(f"\nWith {memory_gb}GB unified memory:")
        
        for model_name, num_params in models[:2]:  # Just 7B and 13B
            model_size_gb = estimate_model_size(num_params, bits=4, include_gradients=False)
            batch_size = get_optimal_batch_size(
                model_size_gb,
                sequence_length=2048,
                available_memory_gb=memory_gb,
                safety_factor=0.8,
            )
            
            print(f"  {model_name} (INT4): batch_size = {batch_size}")


def demo_mlx_device_info():
    """Demonstrate MLX device information."""
    print("\n" + "=" * 60)
    print("MLX DEVICE INFORMATION")
    print("=" * 60)
    
    device_info = check_mlx_device()
    
    print("\nDevice Information:")
    for key, value in device_info.items():
        if key != "error":
            print(f"  {key}: {value}")
    
    if device_info.get("chip_series"):
        print(f"\nDetected Apple Silicon: {device_info['chip_series']} series")
        
        # Provide optimization tips based on chip
        print("\nOptimization Tips:")
        if "M3" in device_info.get("chip_series", ""):
            print("  - M3 series has improved ML accelerators")
            print("  - Supports larger batch sizes for same memory")
            print("  - Enhanced unified memory bandwidth")
        elif "M2" in device_info.get("chip_series", ""):
            print("  - M2 series offers good ML performance")
            print("  - Optimal for models up to 13B with quantization")
        elif "M1" in device_info.get("chip_series", ""):
            print("  - M1 series pioneered unified memory for ML")
            print("  - Works well with 7B models")


def demo_real_world_scenario():
    """Demonstrate a real-world training scenario."""
    print("\n" + "=" * 60)
    print("REAL-WORLD TRAINING SCENARIO")
    print("=" * 60)
    
    print("\nScenario: Fine-tuning a 7B model with QLoRA on M2 Max (32GB)")
    
    # 1. Check device
    device_info = check_mlx_device()
    total_memory = device_info.get("total_memory_gb", 32)
    print(f"\nAvailable memory: {total_memory:.1f} GB")
    
    # 2. Model configuration
    model_params = 7e9
    quantization_bits = 4
    sequence_length = 2048
    
    # 3. Memory estimation
    model_memory = estimate_model_size(
        model_params,
        bits=quantization_bits,
        include_gradients=False,  # QLoRA only updates adapters
        include_optimizer_states=True,
    )
    print(f"Model memory requirement: {model_memory:.1f} GB")
    
    # 4. Optimal batch size
    batch_size = get_optimal_batch_size(
        model_memory,
        sequence_length=sequence_length,
        available_memory_gb=total_memory,
        safety_factor=0.7,  # Conservative for stability
    )
    print(f"Recommended batch size: {batch_size}")
    
    # 5. Create monitoring setup
    memory_profiler = MemoryProfiler()
    perf_monitor = PerformanceMonitor()
    
    print("\nSimulating training pipeline...")
    
    # 6. Dataset preparation
    with memory_profiler.profile("Dataset Loading"):
        # Simulate dataset loading
        dataset_size = 1000
        print(f"  Loading {dataset_size} samples...")
        import time
        time.sleep(0.1)
    
    # 7. Model initialization
    with memory_profiler.profile("Model Initialization"):
        print("  Initializing 7B model with 4-bit quantization...")
        time.sleep(0.2)
    
    # 8. Training simulation
    num_steps = 10
    with perf_monitor.benchmark(
        num_samples=num_steps * batch_size,
        num_tokens=num_steps * batch_size * sequence_length,
        label="Training Simulation"
    ):
        print(f"  Running {num_steps} training steps...")
        
        for step in range(num_steps):
            # Simulate training step
            time.sleep(0.05)
            
            if step % 5 == 0:
                mem_stats = memory_profiler.get_memory_stats()
                print(f"    Step {step}: Memory used = {mem_stats.process_rss_gb:.1f} GB")
    
    # 9. Show results
    print("\nTraining Performance:")
    metrics = perf_monitor.get_latest_metrics()
    if metrics:
        print(f"  Throughput: {metrics.tokens_per_second:.0f} tokens/sec")
        print(f"  Time per step: {metrics.total_time / num_steps:.3f} seconds")
    
    print("\nMemory Profile:")
    print(memory_profiler.get_summary())


def main():
    """Run all demonstrations."""
    parser = argparse.ArgumentParser(description="MLX Utilities Demonstration")
    parser.add_argument(
        "--demo",
        choices=[
            "all",
            "dataset",
            "tokenizer",
            "huggingface",
            "memory",
            "performance",
            "optimization",
            "device",
            "scenario",
        ],
        default="all",
        help="Which demo to run",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MLX UTILITIES DEMONSTRATION")
    print("=" * 60)
    print(f"MLX Available: {MLX_AVAILABLE}")
    
    demos = {
        "dataset": demo_dataset_conversion,
        "tokenizer": demo_tokenizer_integration,
        "huggingface": demo_huggingface_dataset,
        "memory": demo_memory_profiling,
        "performance": demo_performance_monitoring,
        "optimization": demo_model_optimization,
        "device": demo_mlx_device_info,
        "scenario": demo_real_world_scenario,
    }
    
    if args.demo == "all":
        for demo_func in demos.values():
            try:
                demo_func()
            except Exception as e:
                print(f"\nError in demo: {e}")
                import traceback
                traceback.print_exc()
    else:
        demos[args.demo]()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()