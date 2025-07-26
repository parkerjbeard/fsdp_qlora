"""
MPS FSDP Example

This example demonstrates how to use FSDP (Fully Sharded Data Parallel) 
on Apple Silicon with the MPS backend. It includes:
- Model wrapping with FSDP
- Memory-efficient training
- Checkpoint management
- Performance optimization
"""

import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import ShardingStrategy

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mps_fsdp_wrapper import (
    MPSFSDPConfig,
    MPSFSDPWrapper,
    create_mps_fsdp_wrapper,
    check_mps_fsdp_compatibility,
)
from backend_manager import BackendManager


# Simple dataset for demonstration
class TextDataset(Dataset):
    """Simple text dataset for demonstration."""
    
    def __init__(self, num_samples=1000, seq_length=512, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random data
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        labels = torch.randint(0, self.vocab_size, (self.seq_length,))
        
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(self.seq_length),
            "labels": labels,
        }


# Transformer block for LLaMA-style model
class TransformerBlock(nn.Module):
    """Simplified transformer block."""
    
    def __init__(self, hidden_size=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            batch_first=True,
            dropout=0.1,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(0.1),
        )
    
    def forward(self, x, attention_mask=None):
        # Pre-norm architecture
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, key_padding_mask=attention_mask)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# Simple LLaMA-style model
class SimpleLLaMA(nn.Module):
    """Simplified LLaMA model for demonstration."""
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=2048,
    ):
        super().__init__()
        
        # Token embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Store config
        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
        }
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits


def check_system_compatibility():
    """Check system compatibility for MPS FSDP."""
    print("Checking MPS FSDP Compatibility...")
    print("-" * 50)
    
    info = check_mps_fsdp_compatibility()
    
    print(f"MPS Available: {info['mps_available']}")
    print(f"MPS Built: {info['mps_built']}")
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"FSDP Available: {info['fsdp_available']}")
    
    if info.get('float16_supported') is not None:
        print(f"Float16 Supported: {info['float16_supported']}")
    if info.get('bfloat16_supported') is not None:
        print(f"BFloat16 Supported: {info['bfloat16_supported']} (should be False)")
    
    if info.get('warnings'):
        print("\nWarnings:")
        for warning in info['warnings']:
            print(f"  - {warning}")
    
    print("-" * 50)
    
    return info['mps_available']


def create_model_and_wrap(args):
    """Create model and wrap with FSDP."""
    print(f"\nCreating {args.model_size} model...")
    
    # Model configurations
    model_configs = {
        "tiny": {"hidden_size": 256, "num_layers": 4, "num_heads": 4},
        "small": {"hidden_size": 768, "num_layers": 12, "num_heads": 12},
        "medium": {"hidden_size": 1024, "num_layers": 24, "num_heads": 16},
        "large": {"hidden_size": 2048, "num_layers": 32, "num_heads": 32},
    }
    
    config = model_configs[args.model_size]
    
    # Create model
    model = SimpleLLaMA(
        vocab_size=args.vocab_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        max_seq_length=args.max_seq_length,
    )
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_gb = total_params * 2 / 1e9  # FP16
    print(f"Model parameters: {total_params:,}")
    print(f"Model size (FP16): {model_size_gb:.2f} GB")
    
    # Determine sharding strategy based on model size
    if args.sharding_strategy == "auto":
        if model_size_gb < 1:
            strategy = "NO_SHARD"
        elif model_size_gb < 5:
            strategy = "SHARD_GRAD_OP"
        else:
            strategy = "FULL_SHARD"
        print(f"Auto-selected sharding strategy: {strategy}")
    else:
        strategy = args.sharding_strategy
    
    # Create FSDP wrapper
    print("\nWrapping model with FSDP...")
    wrapper = create_mps_fsdp_wrapper(
        sharding_strategy=strategy,
        mixed_precision=args.mixed_precision,
        cpu_offload=args.cpu_offload,
        min_num_params=args.min_wrap_params,
        profile_memory=args.profile_memory,
    )
    
    # Wrap model
    if args.auto_wrap:
        # Use transformer auto-wrap policy
        fsdp_model = wrapper.wrap_transformer(
            model,
            transformer_layer_cls=TransformerBlock,
        )
    else:
        # Manual wrapping
        fsdp_model = wrapper.wrap_model(model)
    
    # Move to MPS
    fsdp_model = fsdp_model.to("mps")
    
    return fsdp_model, wrapper, model_size_gb


def train_step(
    model,
    batch,
    criterion,
    optimizer,
    wrapper,
    accumulation_steps=1,
    current_step=0,
):
    """Perform a single training step."""
    # Move batch to MPS
    input_ids = batch["input_ids"].to("mps")
    labels = batch["labels"].to("mps")
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to("mps")
    
    # Forward pass
    with wrapper.profile_memory() if wrapper.config.profile_memory else nullcontext():
        logits = model(input_ids, attention_mask)
        
        # Reshape for loss calculation
        logits = logits.reshape(-1, logits.size(-1))
        labels = labels.reshape(-1)
        
        # Calculate loss
        loss = criterion(logits, labels)
        loss = loss / accumulation_steps
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    if (current_step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return loss.item() * accumulation_steps


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    wrapper,
    epoch,
    args,
):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        loss = train_step(
            model,
            batch,
            criterion,
            optimizer,
            wrapper,
            args.gradient_accumulation_steps,
            batch_idx,
        )
        
        total_loss += loss
        
        # Logging
        if batch_idx % args.log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss:.4f} "
                  f"LR: {current_lr:.2e} "
                  f"Time: {elapsed:.1f}s")
            
            # Memory stats
            if args.profile_memory:
                stats = wrapper.get_memory_stats()
                print(f"  Memory - Allocated: {stats['allocated_gb']:.2f} GB, "
                      f"Reserved: {stats['reserved_gb']:.2f} GB")
        
        # Clear cache periodically
        if batch_idx % 100 == 0:
            torch.mps.empty_cache()
    
    avg_loss = total_loss / len(train_loader)
    epoch_time = time.time() - start_time
    
    return avg_loss, epoch_time


def main(args):
    """Main training function."""
    # Check compatibility
    if not check_system_compatibility():
        print("MPS is not available on this system!")
        return
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create model and wrap with FSDP
    model, wrapper, model_size_gb = create_model_and_wrap(args)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset and dataloader
    print("\nCreating dataset...")
    train_dataset = TextDataset(
        num_samples=args.train_samples,
        seq_length=args.max_seq_length,
        vocab_size=args.vocab_size,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # MPS works best with single process
        pin_memory=False,
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(train_loader)}")
    
    # Training loop
    print("\nStarting training...")
    print("=" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Train epoch
        avg_loss, epoch_time = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            wrapper,
            epoch + 1,
            args,
        )
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Throughput: {len(train_loader) / epoch_time:.2f} batches/sec")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            if args.save_checkpoint:
                checkpoint_path = os.path.join(
                    args.output_dir,
                    f"best_model_epoch_{epoch + 1}.pt"
                )
                
                print(f"  Saving checkpoint to {checkpoint_path}")
                wrapper.save_checkpoint(
                    model,
                    checkpoint_path,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    best_loss=best_loss,
                    model_config=model.config,
                )
        
        print("=" * 50)
    
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    
    # Final memory stats
    if args.profile_memory:
        final_stats = wrapper.get_memory_stats()
        print(f"\nFinal Memory Stats:")
        print(f"  Allocated: {final_stats['allocated_gb']:.2f} GB")
        print(f"  Reserved: {final_stats['reserved_gb']:.2f} GB")


# Context manager for optional memory profiling
from contextlib import nullcontext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPS FSDP Training Example")
    
    # Model arguments
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Model size configuration",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    
    # FSDP arguments
    parser.add_argument(
        "--sharding-strategy",
        type=str,
        choices=["auto", "NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD"],
        default="auto",
        help="FSDP sharding strategy",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload parameters to CPU",
    )
    parser.add_argument(
        "--auto-wrap",
        action="store_true",
        help="Use automatic layer wrapping",
    )
    parser.add_argument(
        "--min-wrap-params",
        type=float,
        default=1e6,
        help="Minimum parameters for FSDP wrapping",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000,
        help="Number of training samples",
    )
    
    # Other arguments
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Profile memory usage",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Logging interval",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mps_fsdp_output",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training
    main(args)