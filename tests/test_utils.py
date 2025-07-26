"""
Test utilities for integration tests.
"""

import os
import sys
import torch
import time
import psutil
import gc
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset, DataLoader
import tempfile
from contextlib import contextmanager

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend, BackendManager


@dataclass
class MemoryStats:
    """Memory statistics for a test run."""
    initial_mb: float
    peak_mb: float
    final_mb: float
    allocated_mb: float  # GPU memory if available
    
    @property
    def total_used_mb(self) -> float:
        """Total memory used during test."""
        return self.peak_mb - self.initial_mb


class DummyDataset(Dataset):
    """Simple dummy dataset for testing."""
    
    def __init__(self, size: int = 100, seq_length: int = 128, vocab_size: int = 32000):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random token IDs
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        # Use input_ids as labels (next token prediction)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones(self.seq_length, dtype=torch.long)
        }


from transformers import LlamaConfig


def TinyLlamaConfig(**kwargs):
    """Configuration for a tiny LLaMA model for testing."""
    # Set small defaults for testing
    defaults = {
        'hidden_size': 256,
        'intermediate_size': 512,
        'num_hidden_layers': 4,
        'num_attention_heads': 4,
        'num_key_value_heads': 4,
        'vocab_size': 32000,
        'max_position_embeddings': 512,
        'use_cache': False,
        '_attn_implementation': 'eager'
    }
    defaults.update(kwargs)
    return LlamaConfig(**defaults)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    stats = {
        'cpu_rss_mb': memory_info.rss / 1024 / 1024,
        'cpu_vms_mb': memory_info.vms / 1024 / 1024,
    }
    
    # Check for GPU memory
    if torch.cuda.is_available():
        stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        # MPS doesn't have detailed memory tracking
        stats['mps_available'] = True
    
    return stats


@contextmanager
def memory_tracker(backend_manager: Optional[BackendManager] = None):
    """Context manager to track memory usage during a test."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    initial_memory = get_memory_usage()
    peak_memory = initial_memory.copy()
    
    # Create stats object to yield
    stats = MemoryStats(
        initial_mb=initial_memory.get('cpu_rss_mb', 0),
        peak_mb=0,  # Will be updated after
        final_mb=0,  # Will be updated after
        allocated_mb=0  # Will be updated after
    )
    
    yield stats
    
    final_memory = get_memory_usage()
    
    # Calculate peaks
    for key in final_memory:
        if key in peak_memory:
            peak_memory[key] = max(peak_memory[key], final_memory[key])
    
    # Update stats with final values
    stats.peak_mb = peak_memory.get('cpu_rss_mb', 0)
    stats.final_mb = final_memory.get('cpu_rss_mb', 0)
    stats.allocated_mb = final_memory.get('gpu_allocated_mb', 0)


def create_tiny_model(model_type: str = "llama", backend: Backend = Backend.CPU):
    """Create a tiny model for testing."""
    if model_type == "llama":
        # Import the actual LLaMA implementation
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        
        config = TinyLlamaConfig()
        model = LlamaForCausalLM(config)
        
        # Move to appropriate device
        if backend == Backend.CUDA and torch.cuda.is_available():
            model = model.cuda()
        elif backend == Backend.MPS and torch.mps.is_available():
            model = model.to("mps")
            
        return model, config
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def check_convergence(losses: List[float], window: int = 10, threshold: float = 0.1) -> bool:
    """Check if training is converging based on loss history."""
    if len(losses) < window * 2:
        return False
    
    # Compare average of last window with previous window
    recent_avg = np.mean(losses[-window:])
    previous_avg = np.mean(losses[-2*window:-window])
    
    # Check if loss is decreasing
    return recent_avg < previous_avg * (1 - threshold)


def run_mini_training(
    model, 
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_steps: int = 50,
    device: str = "cpu"
) -> List[float]:
    """Run a mini training loop and return losses."""
    model.train()
    losses = []
    
    for step, batch in enumerate(train_dataloader):
        if step >= num_steps:
            break
            
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        
    return losses


def create_test_tokenizer():
    """Create a mock tokenizer for testing."""
    from unittest.mock import MagicMock
    
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.bos_token_id = 0
    tokenizer.model_max_length = 512
    tokenizer.encode = lambda x: [1] * 10  # Return dummy tokens
    tokenizer.decode = lambda x: "dummy text"
    
    return tokenizer


def get_available_device(backend: Backend) -> str:
    """Get the appropriate device string for a backend."""
    if backend == Backend.CUDA and torch.cuda.is_available():
        return "cuda"
    elif backend == Backend.MPS and hasattr(torch, 'mps') and torch.mps.is_available():
        return "mps"
    elif backend == Backend.MLX:
        # MLX uses CPU device in PyTorch
        return "cpu"
    else:
        return "cpu"


def skip_if_backend_unavailable(backend: Backend):
    """Decorator to skip test if backend is not available."""
    def decorator(test_func):
        def wrapper(self, *args, **kwargs):
            backend_manager = BackendManager(backend=backend.value, verbose=False)
            if backend_manager.backend != backend:
                self.skipTest(f"{backend.value} backend not available")
            return test_func(self, *args, **kwargs)
        return wrapper
    return decorator


def get_test_model_name() -> str:
    """Get a small model name for testing."""
    # Use a very small model that's likely to be cached
    return "hf-internal-testing/tiny-random-llama"


@contextmanager
def temp_env_var(key: str, value: str):
    """Temporarily set an environment variable."""
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value