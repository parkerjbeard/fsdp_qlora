# MLX Training Guide

This guide provides comprehensive documentation for using the MLX training functionality in FSDP QLoRA on Apple Silicon devices.

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [MLX Training Configuration](#mlx-training-configuration)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [API Reference](#api-reference)

## Overview

The MLX training implementation provides native Apple Silicon optimization for training large language models with QLoRA (Quantized Low-Rank Adaptation). It leverages the unified memory architecture of Apple Silicon chips to efficiently train models that would otherwise require multiple GPUs.

### Key Features

- **Native MLX Training Loop**: Optimized for Apple Silicon's unified memory architecture
- **4-bit and 8-bit Quantization**: Reduce memory usage while maintaining model quality
- **LoRA Adapters**: Parameter-efficient fine-tuning with customizable rank and alpha
- **Gradient Accumulation**: Train with larger effective batch sizes
- **Mixed Precision**: Automatic optimization for Apple Silicon
- **Memory-Efficient Checkpointing**: Save and resume training efficiently

## Hardware Requirements

### Minimum Requirements

- Apple Silicon Mac (M1, M2, or M3 series)
- macOS 13.0 or later
- 16GB unified memory (for 7B models)

### Recommended Configurations

| Model Size | Chip Type | Memory | Max Batch Size |
|------------|-----------|---------|----------------|
| 7B | M1/M2 | 16GB | 2 |
| 7B | M1/M2 Max | 32GB | 4 |
| 7B | M1/M2/M3 Ultra | 64GB+ | 8 |
| 13B | M1/M2 Max | 32GB | 2 |
| 13B | M1/M2/M3 Ultra | 64GB+ | 4 |
| 70B | M1/M2/M3 Ultra | 128GB+ | 2 |

## MLX Training Configuration

### Basic Configuration

```python
from mlx_trainer import MLXTrainingConfig
from mlx_model_wrapper import MLXConfig

# Model configuration
model_config = MLXConfig(
    model_name="llama-7b",
    vocab_size=32000,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    use_quantization=True,
    quantization_bits=4,
    use_lora=True,
    lora_rank=64,
    lora_alpha=128.0,
    lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Training configuration
training_config = MLXTrainingConfig(
    model_config=model_config,
    learning_rate=2e-4,
    batch_size=4,
    gradient_accumulation_steps=4,
    num_epochs=3,
    warmup_steps=100,
    output_dir="./mlx_checkpoints",
)
```

### Configuration Parameters

#### MLXConfig

- `model_name`: Model identifier (e.g., "llama-7b")
- `use_quantization`: Enable quantization (default: True)
- `quantization_bits`: Quantization precision (4 or 8)
- `use_lora`: Enable LoRA adapters (default: False)
- `lora_rank`: LoRA rank (default: 16)
- `lora_alpha`: LoRA scaling factor (default: 32.0)
- `lora_target_modules`: Modules to apply LoRA (default: ["q_proj", "v_proj"])
- `use_unified_memory`: Optimize for unified memory (default: True)

#### MLXTrainingConfig

- `learning_rate`: Learning rate (default: 5e-4)
- `batch_size`: Training batch size (default: 4)
- `gradient_accumulation_steps`: Steps to accumulate gradients (default: 1)
- `num_epochs`: Number of training epochs (default: 3)
- `warmup_steps`: Linear warmup steps (default: 100)
- `max_grad_norm`: Gradient clipping threshold (default: 1.0)
- `logging_steps`: Steps between logging (default: 10)
- `save_steps`: Steps between checkpoints (default: 500)
- `eval_steps`: Steps between evaluations (default: 500)

## Basic Usage

### 1. Create MLX Model

```python
from mlx_model_wrapper import MLXModelWrapper, create_mlx_model
from mlx_trainer import MLXTrainer, create_mlx_trainer

# Option 1: Create from HuggingFace model (not yet fully implemented)
# model = create_mlx_model("meta-llama/Llama-2-7b-hf", quantization_config, lora_config)

# Option 2: Create custom MLX model
from examples.mlx_llama_model import MLXLlamaModel  # Your model implementation

mlx_model = MLXLlamaModel(model_config)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Wrap for PyTorch compatibility
model_wrapper = MLXModelWrapper(mlx_model, tokenizer)
```

### 2. Prepare Dataset

```python
from torch.utils.data import DataLoader
from datasets import load_dataset

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Create dataloaders
train_dataloader = DataLoader(
    dataset,
    batch_size=training_config.batch_size,
    shuffle=True,
    collate_fn=data_collator,
)
```

### 3. Create and Run Trainer

```python
# Create trainer
trainer = MLXTrainer(
    model=model_wrapper,
    config=training_config,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,  # Optional
)

# Run training
history = trainer.train()

# Training history contains:
# - train_losses: List of (step, loss) tuples
# - eval_losses: List of (step, loss) tuples
# - learning_rates: List of (step, lr) tuples
# - best_eval_loss: Best evaluation loss achieved
```

## Advanced Features

### Gradient Accumulation

Gradient accumulation allows training with larger effective batch sizes:

```python
training_config = MLXTrainingConfig(
    model_config=model_config,
    batch_size=2,  # Small batch per step
    gradient_accumulation_steps=8,  # Effective batch size = 16
)
```

### Mixed Precision Training

MLX automatically handles mixed precision optimization:

```python
training_config = MLXTrainingConfig(
    model_config=model_config,
    mixed_precision=True,  # Enabled by default
)
```

### Custom Learning Rate Schedule

The trainer implements linear warmup followed by cosine decay:

```python
training_config = MLXTrainingConfig(
    model_config=model_config,
    warmup_steps=500,  # Linear warmup for 500 steps
    # After warmup, cosine decay to 0
)
```

### Checkpointing

#### Save Checkpoints

Checkpoints are automatically saved during training:

```python
# Automatic checkpointing every save_steps
training_config = MLXTrainingConfig(
    save_steps=1000,  # Save every 1000 steps
    output_dir="./checkpoints",
)

# Manual checkpoint
trainer.save_checkpoint("my_checkpoint")
```

#### Load Checkpoints

```python
# Resume training from checkpoint
trainer.load_checkpoint("my_checkpoint")

# Or create new trainer from checkpoint
model_wrapper = MLXModelWrapper.from_pretrained("./checkpoints/my_checkpoint")
```

### LoRA Fine-tuning

Apply LoRA adapters for parameter-efficient training:

```python
# Configure LoRA in model config
model_config = MLXConfig(
    model_name="llama-7b",
    use_lora=True,
    lora_rank=128,  # Higher rank for more capacity
    lora_alpha=256.0,  # Alpha = 2 * rank is common
    lora_dropout=0.05,
    lora_target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",  # MLP
    ],
)

# Apply LoRA to existing model
mlx_model.apply_lora(target_modules=["q_proj", "v_proj"])
```

## Performance Optimization

### Memory Optimization

```python
from mlx_model_wrapper import UnifiedMemoryOptimizer

# Create memory optimizer
memory_optimizer = UnifiedMemoryOptimizer(model_wrapper)

# Optimize memory layout
memory_optimizer.optimize_memory_layout()

# Profile memory usage
memory_stats = memory_optimizer.profile_memory_usage()
print(f"Memory usage: {memory_stats['rss_gb']:.2f} GB")
```

### Batch Size Selection

Use the built-in recommendations:

```python
# Get recommended batch size
model_size_gb = 7.0  # For 7B model
chip_type = "m1_ultra"  # Your chip type

max_batch = training_config.get_max_batch_size(model_size_gb, chip_type)
print(f"Recommended max batch size: {max_batch}")
```

### Benchmarking

Benchmark training performance:

```python
from mlx_trainer import benchmark_mlx_training

# Test different batch sizes
results = benchmark_mlx_training(
    model=mlx_model,
    batch_sizes=[1, 2, 4, 8],
    seq_length=2048,
    num_steps=10,
)

for batch_size, metrics in results.items():
    if metrics["status"] == "success":
        print(f"Batch {batch_size}: {metrics['tokens_per_sec']:.1f} tokens/sec")
    else:
        print(f"Batch {batch_size}: {metrics['error']}")
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)

**Symptoms**: Training crashes with memory errors

**Solutions**:
1. Reduce batch size
2. Increase gradient accumulation steps
3. Enable quantization (4-bit recommended)
4. Reduce sequence length
5. Use fewer LoRA target modules

```python
# Memory-efficient configuration
training_config = MLXTrainingConfig(
    batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,  # Future feature
)
```

#### Slow Training

**Symptoms**: Training is slower than expected

**Solutions**:
1. Ensure you're using native MLX operations
2. Check batch size (too small can be inefficient)
3. Verify unified memory optimization is enabled
4. Profile with Activity Monitor

#### Loss Not Decreasing

**Symptoms**: Training loss plateaus or increases

**Solutions**:
1. Reduce learning rate
2. Increase warmup steps
3. Check data preprocessing
4. Verify model initialization

```python
# Conservative training settings
training_config = MLXTrainingConfig(
    learning_rate=1e-4,  # Lower LR
    warmup_steps=1000,  # More warmup
    max_grad_norm=0.5,  # Aggressive clipping
)
```

### Debug Mode

Enable verbose logging for debugging:

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# Training will now show detailed progress
trainer.train()
```

## API Reference

### MLXTrainer

Main trainer class for MLX models.

```python
class MLXTrainer:
    def __init__(
        self,
        model: Union[MLXModel, MLXModelWrapper],
        config: MLXTrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        backend_manager: Optional[BackendManager] = None,
    ):
        """Initialize MLX trainer."""
    
    def train(self) -> Dict[str, Any]:
        """Run the training loop."""
    
    def evaluate(self) -> float:
        """Evaluate the model."""
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint."""
```

### MLXOptimizer

Optimizer wrapper with gradient accumulation.

```python
class MLXOptimizer:
    def __init__(
        self,
        optimizer: Any,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ):
        """Initialize optimizer wrapper."""
    
    def accumulate_gradients(self, grads: Dict[str, mx.array]):
        """Accumulate gradients."""
    
    def step(self) -> bool:
        """Perform optimizer step if ready."""
```

### MLXLossComputer

Loss computation utilities.

```python
class MLXLossComputer:
    @staticmethod
    def cross_entropy_loss(
        logits: mx.array,
        labels: mx.array,
        ignore_index: int = -100,
        reduction: str = "mean"
    ) -> mx.array:
        """Compute cross-entropy loss."""
    
    @staticmethod
    def compute_perplexity(loss: mx.array) -> mx.array:
        """Compute perplexity from loss."""
```

### Convenience Functions

```python
def create_mlx_trainer(
    model: Union[MLXModel, MLXModelWrapper],
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    **kwargs
) -> MLXTrainer:
    """Create trainer with sensible defaults."""

def benchmark_mlx_training(
    model: Union[MLXModel, MLXModelWrapper],
    batch_sizes: List[int] = [1, 2, 4, 8],
    seq_length: int = 512,
    num_steps: int = 10,
) -> Dict[int, Dict[str, float]]:
    """Benchmark training performance."""
```

## Example: Complete Training Script

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

from mlx_model_wrapper import MLXConfig, MLXModelWrapper
from mlx_trainer import MLXTrainingConfig, create_mlx_trainer
from examples.mlx_llama_model import MLXLlamaModel

def main():
    # 1. Configuration
    model_config = MLXConfig(
        model_name="llama-7b",
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        use_quantization=True,
        quantization_bits=4,
        use_lora=True,
        lora_rank=64,
        lora_alpha=128.0,
    )
    
    # 2. Load model and tokenizer
    mlx_model = MLXLlamaModel(model_config)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Create model wrapper
    model_wrapper = MLXModelWrapper(mlx_model, tokenizer)
    
    # 4. Load and prepare dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 5. Create data collator and dataloader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=data_collator,
    )
    
    # 6. Create and run trainer
    trainer = create_mlx_trainer(
        model=model_wrapper,
        train_dataloader=train_dataloader,
        learning_rate=2e-4,
        num_epochs=3,
        warmup_steps=100,
        output_dir="./mlx_outputs",
    )
    
    # 7. Train
    history = trainer.train()
    
    # 8. Save final model
    model_wrapper.save_pretrained("./final_model")
    
    print(f"Training complete! Final loss: {history['train_losses'][-1][1]:.4f}")

if __name__ == "__main__":
    main()
```

## Next Steps

1. Review the [MLX Model Wrapper Documentation](../mlx_model_wrapper.py) for model creation details
2. Check the [examples directory](../examples/) for more usage examples
3. See the [main README](../README.md) for installation instructions
4. Join the community discussions for tips and best practices

## Contributing

We welcome contributions to improve MLX training support! Please see our contributing guidelines for more information.