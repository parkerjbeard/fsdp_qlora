"""
MLX Utilities for FSDP QLoRA

This module provides utilities for working with MLX on Apple Silicon, including:
- Dataset conversion from PyTorch/HuggingFace to MLX arrays
- Tokenizer integration for MLX models
- Memory profiling and monitoring
- Performance benchmarking tools
- Helper functions for MLX operations

Based on MLX LoRA examples and optimized for unified memory architecture.
"""

import time
import psutil
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
from collections import defaultdict
import threading
from contextlib import contextmanager

# Import MLX conditionally
try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn_mlx
    from mlx.utils import tree_flatten, tree_unflatten, tree_map
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    warnings.warn("MLX is not available. MLX utilities will have limited functionality.")

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from datasets import Dataset as HFDataset


# Dataset Conversion Utilities
# Based on MLX LoRA fine-tuning examples

class DatasetConverter:
    """Convert various dataset formats to MLX arrays."""
    
    @staticmethod
    def torch_to_mlx(tensor: torch.Tensor) -> "mx.array":
        """
        Convert PyTorch tensor to MLX array.
        
        Example from MLX LoRA:
        ```python
        # Convert input_ids for language modeling
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        mlx_input_ids = DatasetConverter.torch_to_mlx(input_ids)
        ```
        """
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for tensor conversion")
        
        # Move to CPU and convert to numpy first (as per research note)
        np_array = tensor.detach().cpu().numpy()
        
        # Handle different dtypes
        if tensor.dtype == torch.float16:
            np_array = np_array.astype(np.float16)
        elif tensor.dtype == torch.bfloat16:
            # MLX doesn't support bfloat16 directly, use float32
            np_array = np_array.astype(np.float32)
        
        # Convert to MLX array
        return "mx.array"(np_array)
    
    @staticmethod
    def numpy_to_mlx(array: np.ndarray) -> "mx.array":
        """Convert numpy array to MLX array."""
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for array conversion")
        return "mx.array"(array)
    
    @staticmethod
    def mlx_to_torch(array: "mx.array", device: Optional[str] = None) -> torch.Tensor:
        """Convert MLX array back to PyTorch tensor."""
        # Convert to numpy first
        np_array = np.array(array)
        
        # Create torch tensor
        tensor = torch.from_numpy(np_array)
        
        # Move to device if specified
        if device:
            tensor = tensor.to(device)
        
        return tensor
    
    @staticmethod
    def dict_to_mlx(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert dictionary of tensors/arrays to MLX format.
        
        Example:
        ```python
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[2, 3, 4]])
        }
        mlx_batch = DatasetConverter.dict_to_mlx(batch)
        ```
        """
        mlx_data = {}
        
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                mlx_data[key] = DatasetConverter.torch_to_mlx(value)
            elif isinstance(value, np.ndarray):
                mlx_data[key] = DatasetConverter.numpy_to_mlx(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # Convert lists/tuples of numbers to MLX arrays
                if isinstance(value[0], (int, float)):
                    mlx_data[key] = "mx.array"(value)
                else:
                    mlx_data[key] = value
            else:
                mlx_data[key] = value
        
        return mlx_data


class HuggingFaceDatasetConverter:
    """
    Convert HuggingFace datasets to MLX-compatible format.
    
    Based on MLX LoRA training examples for Alpaca dataset.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def convert_dataset(
        self,
        dataset: HFDataset,
        text_field: str = "text",
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> List[Dict[str, "mx.array"]]:
        """
        Convert HuggingFace dataset to MLX format.
        
        Example from MLX LoRA fine-tuning:
        ```python
        # Load Alpaca dataset
        from datasets import load_dataset
        dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
        
        # Convert to MLX
        converter = HuggingFaceDatasetConverter(tokenizer)
        mlx_dataset = converter.convert_dataset(
            dataset,
            text_field="text",
            max_length=2048
        )
        ```
        """
        mlx_data = []
        
        for item in dataset:
            # Get text
            if isinstance(item[text_field], str):
                text = item[text_field]
            else:
                # Handle instruction-based datasets like Alpaca
                text = self._format_alpaca_prompt(item)
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors="np",  # Return numpy for MLX conversion
            )
            
            # Convert to MLX
            mlx_item = {
                "input_ids": "mx.array"(encoding["input_ids"].squeeze()),
                "attention_mask": "mx.array"(encoding["attention_mask"].squeeze()),
            }
            
            # Add labels (shifted input_ids for language modeling)
            labels = encoding["input_ids"].squeeze().copy()
            labels[:-1] = labels[1:]
            labels[-1] = -100  # Ignore last token
            mlx_item["labels"] = "mx.array"(labels)
            
            mlx_data.append(mlx_item)
        
        return mlx_data
    
    def _format_alpaca_prompt(self, item: Dict[str, str]) -> str:
        """
        Format Alpaca-style instruction dataset.
        
        Based on MLX LoRA examples for instruction tuning.
        """
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt


class MLXDataLoader:
    """
    DataLoader-like interface for MLX arrays.
    
    Optimized for unified memory on Apple Silicon.
    """
    
    def __init__(
        self,
        data: List[Dict[str, "mx.array"]],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._indices = None
        self._reset()
    
    def _reset(self):
        """Reset indices for iteration."""
        n = len(self.data)
        self._indices = np.arange(n)
        
        if self.shuffle:
            np.random.shuffle(self._indices)
        
        # Calculate number of batches
        self._num_batches = n // self.batch_size
        if not self.drop_last and n % self.batch_size != 0:
            self._num_batches += 1
        
        self._batch_idx = 0
    
    def __iter__(self):
        """Iterate over batches."""
        self._reset()
        return self
    
    def __next__(self) -> Dict[str, "mx.array"]:
        """Get next batch."""
        if self._batch_idx >= self._num_batches:
            raise StopIteration
        
        # Get batch indices
        start_idx = self._batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data))
        batch_indices = self._indices[start_idx:end_idx]
        
        # Collect batch data
        batch = defaultdict(list)
        for idx in batch_indices:
            for key, value in self.data[idx].items():
                batch[key].append(value)
        
        # Stack arrays
        stacked_batch = {}
        for key, values in batch.items():
            if len(values) > 0 and isinstance(values[0], "mx.array"):
                # Stack along first dimension
                stacked_batch[key] = mx.stack(values)
            else:
                stacked_batch[key] = values
        
        self._batch_idx += 1
        return stacked_batch
    
    def __len__(self):
        """Number of batches."""
        return self._num_batches


# Tokenizer Integration

class MLXTokenizer:
    """
    Wrapper for HuggingFace tokenizers with MLX output.
    
    Provides efficient tokenization directly to MLX arrays.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_attention_mask: bool = True,
    ) -> Dict[str, "mx.array"]:
        """
        Tokenize text(s) and return MLX arrays.
        
        Example:
        ```python
        mlx_tokenizer = MLXTokenizer(tokenizer)
        outputs = mlx_tokenizer(
            "Fine-tune models with MLX on Apple Silicon!",
            max_length=128
        )
        # outputs["input_ids"] is "mx.array"
        ```
        """
        # Tokenize to numpy
        encoding = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="np",
        )
        
        # Convert to MLX
        mlx_encoding = {}
        for key, value in encoding.items():
            if isinstance(value, np.ndarray):
                mlx_encoding[key] = "mx.array"(value)
            else:
                mlx_encoding[key] = value
        
        return mlx_encoding
    
    def decode(self, token_ids: Union["mx.array", List[int]], **kwargs) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, "mx.array"):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def batch_decode(self, token_ids: "mx.array", **kwargs) -> List[str]:
        """Batch decode token IDs."""
        if isinstance(token_ids, "mx.array"):
            token_ids = np.array(token_ids).tolist()
        
        return self.tokenizer.batch_decode(token_ids, **kwargs)


# Memory Profiling Utilities

@dataclass
class MemoryStats:
    """Memory statistics for Apple Silicon."""
    
    # Process memory
    process_rss_gb: float  # Resident Set Size
    process_vms_gb: float  # Virtual Memory Size
    process_percent: float  # Percentage of total memory
    
    # System memory
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    free_memory_gb: float
    
    # Unified memory specific
    wired_memory_gb: float = 0.0  # Memory that can't be swapped
    compressed_memory_gb: float = 0.0  # Compressed memory
    
    # GPU memory (unified on Apple Silicon)
    gpu_memory_gb: float = 0.0  # Same as process memory on M-series
    
    def __str__(self) -> str:
        """Pretty print memory stats."""
        return (
            f"Memory Stats:\n"
            f"  Process: {self.process_rss_gb:.2f} GB ({self.process_percent:.1f}%)\n"
            f"  Available: {self.available_memory_gb:.2f} GB / {self.total_memory_gb:.2f} GB\n"
            f"  Used: {self.used_memory_gb:.2f} GB\n"
        )


class MemoryProfiler:
    """
    Memory profiler for MLX on Apple Silicon.
    
    Tracks unified memory usage during training.
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.history = []
        self._monitoring = False
        self._monitor_thread = None
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # Process memory
        mem_info = self.process.memory_info()
        process_rss_gb = mem_info.rss / 1e9
        process_vms_gb = mem_info.vms / 1e9
        process_percent = self.process.memory_percent()
        
        # System memory
        vm = psutil.virtual_memory()
        total_memory_gb = vm.total / 1e9
        available_memory_gb = vm.available / 1e9
        used_memory_gb = vm.used / 1e9
        free_memory_gb = vm.free / 1e9
        
        # macOS specific memory stats
        wired_memory_gb = getattr(vm, 'wired', 0) / 1e9
        compressed_memory_gb = getattr(vm, 'compressed', 0) / 1e9
        
        # On Apple Silicon, GPU memory is unified
        gpu_memory_gb = process_rss_gb
        
        return MemoryStats(
            process_rss_gb=process_rss_gb,
            process_vms_gb=process_vms_gb,
            process_percent=process_percent,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            used_memory_gb=used_memory_gb,
            free_memory_gb=free_memory_gb,
            wired_memory_gb=wired_memory_gb,
            compressed_memory_gb=compressed_memory_gb,
            gpu_memory_gb=gpu_memory_gb,
        )
    
    @contextmanager
    def profile(self, label: str = ""):
        """
        Context manager for memory profiling.
        
        Example:
        ```python
        profiler = MemoryProfiler()
        
        with profiler.profile("Model Loading"):
            model = load_model()
        
        print(profiler.get_summary())
        ```
        """
        # Get memory before
        start_stats = self.get_memory_stats()
        start_time = time.time()
        
        yield
        
        # Get memory after
        end_stats = self.get_memory_stats()
        duration = time.time() - start_time
        
        # Calculate deltas
        memory_delta = end_stats.process_rss_gb - start_stats.process_rss_gb
        
        # Store in history
        self.history.append({
            "label": label,
            "duration": duration,
            "start_memory_gb": start_stats.process_rss_gb,
            "end_memory_gb": end_stats.process_rss_gb,
            "memory_delta_gb": memory_delta,
            "peak_memory_gb": max(start_stats.process_rss_gb, end_stats.process_rss_gb),
        })
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        def monitor():
            while self._monitoring:
                stats = self.get_memory_stats()
                self.history.append({
                    "timestamp": time.time(),
                    "memory_gb": stats.process_rss_gb,
                    "available_gb": stats.available_memory_gb,
                })
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def get_summary(self) -> str:
        """Get summary of memory profiling."""
        if not self.history:
            return "No profiling data available"
        
        # Filter out monitoring entries
        profile_entries = [h for h in self.history if "label" in h]
        
        if not profile_entries:
            return "No labeled profiling sections"
        
        summary = "Memory Profiling Summary:\n"
        summary += "-" * 50 + "\n"
        
        for entry in profile_entries:
            summary += f"{entry['label']}:\n"
            summary += f"  Duration: {entry['duration']:.2f}s\n"
            summary += f"  Memory: {entry['start_memory_gb']:.2f} GB → {entry['end_memory_gb']:.2f} GB\n"
            summary += f"  Delta: {entry['memory_delta_gb']:+.2f} GB\n"
            summary += "-" * 50 + "\n"
        
        return summary
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage over time (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib required for plotting. Install with: pip install matplotlib")
            return
        
        # Extract monitoring data
        monitoring_data = [h for h in self.history if "timestamp" in h]
        
        if not monitoring_data:
            print("No monitoring data to plot")
            return
        
        # Extract data
        timestamps = [h["timestamp"] for h in monitoring_data]
        memory_usage = [h["memory_gb"] for h in monitoring_data]
        available = [h["available_gb"] for h in monitoring_data]
        
        # Normalize timestamps
        start_time = timestamps[0]
        timestamps = [(t - start_time) for t in timestamps]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, memory_usage, label="Process Memory", linewidth=2)
        plt.plot(timestamps, available, label="Available Memory", linewidth=2, linestyle="--")
        
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory (GB)")
        plt.title("Memory Usage Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


# Performance Monitoring Tools

@dataclass
class PerformanceMetrics:
    """Performance metrics for MLX operations."""
    
    # Timing
    total_time: float
    samples_per_second: float
    
    # Token metrics (for language models)
    tokens_per_second: float = 0.0
    
    # Memory
    peak_memory_gb: float = 0.0
    avg_memory_gb: float = 0.0
    
    # MLX specific
    compile_time: float = 0.0  # Time for MLX graph compilation
    compute_time: float = 0.0  # Actual computation time
    
    def __str__(self) -> str:
        """Pretty print metrics."""
        output = "Performance Metrics:\n"
        output += f"  Total time: {self.total_time:.2f}s\n"
        output += f"  Throughput: {self.samples_per_second:.1f} samples/s"
        
        if self.tokens_per_second > 0:
            output += f" ({self.tokens_per_second:.1f} tokens/s)"
        
        output += f"\n  Peak memory: {self.peak_memory_gb:.2f} GB"
        
        if self.compile_time > 0:
            output += f"\n  MLX compile: {self.compile_time:.2f}s"
            output += f" ({self.compile_time/self.total_time*100:.1f}%)"
        
        return output


class PerformanceMonitor:
    """
    Performance monitoring for MLX operations.
    
    Based on MLX benchmarking practices.
    """
    
    def __init__(self):
        self.memory_profiler = MemoryProfiler()
        self.metrics_history = []
    
    @contextmanager
    def benchmark(
        self,
        num_samples: int,
        num_tokens: Optional[int] = None,
        label: str = "",
    ):
        """
        Benchmark MLX operations.
        
        Example from MLX LoRA benchmarks:
        ```python
        monitor = PerformanceMonitor()
        
        with monitor.benchmark(num_samples=100, num_tokens=100*512, label="Training"):
            for batch in dataloader:
                loss = train_step(batch)
        
        print(monitor.get_latest_metrics())
        ```
        """
        # Start memory monitoring
        self.memory_profiler.start_monitoring(interval=0.5)
        
        # Record start
        start_time = time.time()
        start_mem = self.memory_profiler.get_memory_stats()
        compile_start = None
        
        # Context for tracking MLX compilation
        self._compile_time = 0.0
        
        try:
            yield self
            
            # Record end
            end_time = time.time()
            total_time = end_time - start_time
            
            # Stop memory monitoring
            self.memory_profiler.stop_monitoring()
            
            # Calculate metrics
            samples_per_second = num_samples / total_time if total_time > 0 else 0
            tokens_per_second = num_tokens / total_time if num_tokens and total_time > 0 else 0
            
            # Get memory stats
            memory_history = [h for h in self.memory_profiler.history if "memory_gb" in h]
            if memory_history:
                peak_memory = max(h["memory_gb"] for h in memory_history)
                avg_memory = sum(h["memory_gb"] for h in memory_history) / len(memory_history)
            else:
                peak_memory = start_mem.process_rss_gb
                avg_memory = start_mem.process_rss_gb
            
            # Create metrics
            metrics = PerformanceMetrics(
                total_time=total_time,
                samples_per_second=samples_per_second,
                tokens_per_second=tokens_per_second,
                peak_memory_gb=peak_memory,
                avg_memory_gb=avg_memory,
                compile_time=self._compile_time,
                compute_time=total_time - self._compile_time,
            )
            
            # Store in history
            self.metrics_history.append({
                "label": label,
                "metrics": metrics,
                "timestamp": time.time(),
            })
            
        except Exception as e:
            self.memory_profiler.stop_monitoring()
            raise e
    
    def mark_compile_start(self):
        """Mark the start of MLX compilation."""
        self._compile_start_time = time.time()
    
    def mark_compile_end(self):
        """Mark the end of MLX compilation."""
        if hasattr(self, '_compile_start_time'):
            self._compile_time += time.time() - self._compile_start_time
            delattr(self, '_compile_start_time')
    
    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]["metrics"]
        return None
    
    def get_summary(self) -> str:
        """Get summary of all benchmarks."""
        if not self.metrics_history:
            return "No benchmark data available"
        
        summary = "Performance Benchmark Summary:\n"
        summary += "=" * 60 + "\n"
        
        for entry in self.metrics_history:
            summary += f"\n{entry['label']}:\n"
            summary += str(entry['metrics'])
            summary += "\n" + "-" * 60
        
        return summary
    
    def compare_benchmarks(self, labels: Optional[List[str]] = None) -> str:
        """Compare different benchmarks."""
        if not self.metrics_history:
            return "No benchmark data available"
        
        # Filter by labels if provided
        entries = self.metrics_history
        if labels:
            entries = [e for e in entries if e["label"] in labels]
        
        if not entries:
            return "No matching benchmarks found"
        
        # Create comparison table
        summary = "Benchmark Comparison:\n"
        summary += "=" * 80 + "\n"
        summary += f"{'Label':<20} {'Time (s)':<10} {'Samples/s':<12} {'Tokens/s':<12} {'Memory (GB)':<12}\n"
        summary += "-" * 80 + "\n"
        
        for entry in entries:
            m = entry["metrics"]
            summary += f"{entry['label']:<20} "
            summary += f"{m.total_time:<10.2f} "
            summary += f"{m.samples_per_second:<12.1f} "
            summary += f"{m.tokens_per_second:<12.1f} "
            summary += f"{m.peak_memory_gb:<12.2f}\n"
        
        return summary


# Helper Utilities

def estimate_model_size(
    num_parameters: int,
    bits: int = 16,
    include_gradients: bool = True,
    include_optimizer_states: bool = True,
) -> float:
    """
    Estimate memory requirements for a model.
    
    Based on MLX memory calculations for QLoRA:
    - 4-bit: ~0.5 bytes per parameter
    - 8-bit: 1 byte per parameter
    - 16-bit: 2 bytes per parameter
    - 32-bit: 4 bytes per parameter
    
    Example:
    ```python
    # 7B parameter model with 4-bit quantization
    memory_gb = estimate_model_size(7e9, bits=4, include_gradients=False)
    print(f"Estimated memory: {memory_gb:.1f} GB")
    ```
    """
    # Base model size
    bytes_per_param = bits / 8
    model_size = num_parameters * bytes_per_param
    
    total_size = model_size
    
    if include_gradients:
        # Gradients are typically fp32
        gradient_size = num_parameters * 4
        total_size += gradient_size
    
    if include_optimizer_states:
        # AdamW has 2 momentum states (fp32 each)
        optimizer_size = num_parameters * 4 * 2
        total_size += optimizer_size
    
    return total_size / 1e9  # Convert to GB


def get_optimal_batch_size(
    model_size_gb: float,
    sequence_length: int = 2048,
    available_memory_gb: Optional[float] = None,
    safety_factor: float = 0.8,
) -> int:
    """
    Estimate optimal batch size for Apple Silicon.
    
    Based on MLX benchmarks and unified memory constraints.
    """
    if available_memory_gb is None:
        # Get available memory
        vm = psutil.virtual_memory()
        available_memory_gb = vm.available / 1e9
    
    # Apply safety factor
    usable_memory_gb = available_memory_gb * safety_factor
    
    # Estimate memory per sample (rough approximation)
    # Includes activations, attention matrices, etc.
    memory_per_sample_gb = (sequence_length * 4096 * 4) / 1e9  # Assuming ~4K hidden size
    
    # Calculate batch size
    batch_size = int((usable_memory_gb - model_size_gb) / memory_per_sample_gb)
    
    # Clamp to reasonable values
    batch_size = max(1, min(batch_size, 32))
    
    return batch_size


def format_memory_size(size_bytes: Union[int, float]) -> str:
    """Format memory size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def check_mlx_device() -> Dict[str, Any]:
    """
    Check MLX device information.
    
    Returns information about the Apple Silicon chip.
    """
    if not MLX_AVAILABLE:
        return {"available": False, "error": "MLX not installed"}
    
    info = {
        "available": True,
        "device": "Apple Silicon",
        "unified_memory": True,
    }
    
    # Get system info
    try:
        import platform
        info["platform"] = platform.platform()
        info["processor"] = platform.processor()
        
        # Get memory info
        vm = psutil.virtual_memory()
        info["total_memory_gb"] = vm.total / 1e9
        
        # Try to detect chip type from system info
        if "apple" in platform.processor().lower():
            processor = platform.processor()
            if "m3" in processor.lower():
                info["chip_series"] = "M3"
            elif "m2" in processor.lower():
                info["chip_series"] = "M2"
            elif "m1" in processor.lower():
                info["chip_series"] = "M1"
            else:
                info["chip_series"] = "Unknown"
    except Exception as e:
        info["error"] = str(e)
    
    return info


# Convenience functions

def create_mlx_dataloader(
    dataset: Union[HFDataset, List[Dict], Dataset],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1,
    max_length: int = 512,
    shuffle: bool = True,
    **kwargs,
) -> MLXDataLoader:
    """
    Create an MLX DataLoader from various dataset formats.
    
    Example:
    ```python
    from datasets import load_dataset
    
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    dataloader = create_mlx_dataloader(
        dataset,
        tokenizer,
        batch_size=4,
        max_length=2048,
    )
    
    for batch in dataloader:
        # batch contains MLX arrays
        print(batch["input_ids"].shape)
    ```
    """
    # Convert dataset to MLX format
    if isinstance(dataset, HFDataset):
        converter = HuggingFaceDatasetConverter(tokenizer)
        mlx_data = converter.convert_dataset(dataset, max_length=max_length, **kwargs)
    elif isinstance(dataset, list):
        # Assume list of dicts with tensor/array values
        mlx_data = [DatasetConverter.dict_to_mlx(item) for item in dataset]
    elif isinstance(dataset, Dataset):
        # PyTorch dataset - convert each item
        mlx_data = []
        for i in range(len(dataset)):
            item = dataset[i]
            mlx_item = DatasetConverter.dict_to_mlx(item)
            mlx_data.append(mlx_item)
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")
    
    # Create MLX DataLoader
    return MLXDataLoader(
        mlx_data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=kwargs.get("drop_last", False),
    )


def profile_mlx_operation(
    func: Callable,
    *args,
    num_warmup: int = 3,
    num_runs: int = 10,
    **kwargs,
) -> Dict[str, float]:
    """
    Profile an MLX operation with warmup.
    
    Based on MLX benchmarking best practices.
    
    Example:
    ```python
    def matrix_multiply(a, b):
        return mx.matmul(a, b)
    
    a = mx.random.normal((1000, 1000))
    b = mx.random.normal((1000, 1000))
    
    stats = profile_mlx_operation(matrix_multiply, a, b)
    print(f"Average time: {stats['mean_time']:.4f}s")
    ```
    """
    if not MLX_AVAILABLE:
        raise ImportError("MLX required for profiling")
    
    times = []
    
    # Warmup runs
    for _ in range(num_warmup):
        _ = func(*args, **kwargs)
        mx.eval()  # Ensure computation completes
    
    # Timed runs
    for _ in range(num_runs):
        start = time.time()
        result = func(*args, **kwargs)
        mx.eval(result)  # Force evaluation
        end = time.time()
        times.append(end - start)
    
    # Calculate statistics
    times_array = np.array(times)
    stats = {
        "mean_time": np.mean(times_array),
        "std_time": np.std(times_array),
        "min_time": np.min(times_array),
        "max_time": np.max(times_array),
        "median_time": np.median(times_array),
    }
    
    return stats


# Example usage based on MLX LoRA fine-tuning

def example_alpaca_finetuning():
    """
    Example of using MLX utilities for Alpaca fine-tuning.
    
    Based on MLX LoRA examples.
    """
    print("MLX Utilities Example: Alpaca Fine-tuning")
    print("-" * 50)
    
    # Check device
    device_info = check_mlx_device()
    print(f"Device: {device_info}")
    
    if not MLX_AVAILABLE:
        print("MLX not available. Install with: pip install mlx mlx-lm")
        return
    
    # Example configuration (would come from actual model)
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    mlx_tokenizer = MLXTokenizer(tokenizer)
    
    # Example text
    text = "Fine-tune large language models efficiently on Apple Silicon with MLX!"
    
    # Tokenize
    tokens = mlx_tokenizer(text, max_length=128)
    print(f"\nTokenized shape: {tokens['input_ids'].shape}")
    
    # Memory profiling
    profiler = MemoryProfiler()
    
    with profiler.profile("Tokenization"):
        # Simulate some work
        large_batch = mlx_tokenizer([text] * 100, max_length=512)
    
    print(f"\n{profiler.get_summary()}")
    
    # Performance monitoring
    monitor = PerformanceMonitor()
    
    # Simulate training step
    def dummy_forward(batch):
        # Simulate model forward pass
        return mx.mean(batch["input_ids"].astype(mx.float32))
    
    with monitor.benchmark(num_samples=100, num_tokens=100*128, label="Forward Pass"):
        for i in range(100):
            loss = dummy_forward(tokens)
            mx.eval(loss)
    
    print(f"\n{monitor.get_latest_metrics()}")
    
    # Model size estimation
    model_size_7b = estimate_model_size(7e9, bits=4, include_gradients=False)
    print(f"\nEstimated 7B model size (4-bit): {model_size_7b:.1f} GB")
    
    # Optimal batch size
    optimal_batch = get_optimal_batch_size(model_size_7b, sequence_length=2048)
    print(f"Recommended batch size: {optimal_batch}")


if __name__ == "__main__":
    # Run example
    example_alpaca_finetuning()


__all__ = [
    # Dataset conversion
    "DatasetConverter",
    "HuggingFaceDatasetConverter",
    "MLXDataLoader",
    
    # Tokenizer
    "MLXTokenizer",
    
    # Memory profiling
    "MemoryStats",
    "MemoryProfiler",
    
    # Performance monitoring
    "PerformanceMetrics",
    "PerformanceMonitor",
    
    # Helper functions
    "estimate_model_size",
    "get_optimal_batch_size",
    "format_memory_size",
    "check_mlx_device",
    "create_mlx_dataloader",
    "profile_mlx_operation",
]