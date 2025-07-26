"""
Complete MLX Training Example

This script demonstrates a full end-to-end training pipeline using MLX
on Apple Silicon, including dataset loading, model configuration,
training with evaluation, and checkpoint management.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check MLX availability
try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn_mlx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. This example requires MLX installation.")
    print("Install with: pip install mlx mlx-lm")

from backend_manager import BackendManager
from mlx_model_wrapper import MLXConfig, MLXModel, MLXModelWrapper, MLXLinear, LoRALinear
from mlx_trainer import MLXTrainingConfig, MLXTrainer, create_mlx_trainer


class SimpleLlamaModel(MLXModel):
    """Simplified LLaMA-style model for demonstration."""
    
    def __init__(self, config: MLXConfig):
        super().__init__(config)
        
        # Token embeddings
        self.embed_tokens = nn_mlx.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers (simplified)
        self.layers = []
        for _ in range(config.num_hidden_layers):
            layer = nn_mlx.Module()
            
            # Self-attention (simplified)
            if config.use_quantization:
                layer.q_proj = MLXLinear(
                    config.hidden_size,
                    config.hidden_size,
                    bias=False,
                    quantized=True,
                    bits=config.quantization_bits,
                )
                layer.v_proj = MLXLinear(
                    config.hidden_size,
                    config.hidden_size,
                    bias=False,
                    quantized=True,
                    bits=config.quantization_bits,
                )
            else:
                layer.q_proj = nn_mlx.Linear(config.hidden_size, config.hidden_size, bias=False)
                layer.v_proj = nn_mlx.Linear(config.hidden_size, config.hidden_size, bias=False)
            
            self.layers.append(layer)
        
        # Output layer
        self.norm = nn_mlx.LayerNorm(config.hidden_size)
        self.lm_head = nn_mlx.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def __call__(self, input_ids: mx.array, **kwargs) -> mx.array:
        """Forward pass through the model."""
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Pass through layers (simplified - no attention mechanism)
        for layer in self.layers:
            # Simplified attention (just projections for demo)
            q = layer.q_proj(hidden_states)
            v = layer.v_proj(hidden_states)
            hidden_states = hidden_states + (q + v) * 0.5
        
        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits


class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(
        self,
        texts: list,
        tokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Prepare labels (shifted input_ids)
        input_ids = encoding["input_ids"].squeeze()
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def load_alpaca_dataset(tokenizer, num_samples: int = 1000):
    """Load and prepare Alpaca dataset."""
    print(f"Loading Alpaca dataset (first {num_samples} samples)...")
    
    # Load dataset
    dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{num_samples}]")
    
    # Format texts
    texts = []
    for item in dataset:
        text = f"### Instruction:\n{item['instruction']}\n\n"
        if item.get('input'):
            text += f"### Input:\n{item['input']}\n\n"
        text += f"### Response:\n{item['output']}"
        texts.append(text)
    
    # Create dataset
    train_size = int(0.9 * len(texts))
    train_texts = texts[:train_size]
    eval_texts = texts[train_size:]
    
    train_dataset = TextDataset(train_texts, tokenizer)
    eval_dataset = TextDataset(eval_texts, tokenizer)
    
    return train_dataset, eval_dataset


def create_model_and_tokenizer(args):
    """Create model and tokenizer."""
    print("Creating model and tokenizer...")
    
    # Model configuration
    model_config = MLXConfig(
        model_name=args.model_name,
        vocab_size=32000,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        use_quantization=args.use_quantization,
        quantization_bits=args.quantization_bits,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=["q_proj", "v_proj"],
    )
    
    # Create MLX model
    if MLX_AVAILABLE:
        mlx_model = SimpleLlamaModel(model_config)
        
        # Apply LoRA if requested
        if args.use_lora:
            print(f"Applying LoRA with rank={args.lora_rank}, alpha={args.lora_alpha}")
            mlx_model.apply_lora()
    else:
        # Mock model for testing without MLX
        from unittest.mock import MagicMock
        mlx_model = MagicMock()
        mlx_model.config = model_config
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model wrapper
    model_wrapper = MLXModelWrapper(
        mlx_model,
        tokenizer=tokenizer,
        backend_manager=BackendManager(backend="mlx"),
    )
    
    return model_wrapper, tokenizer


def main(args):
    """Main training function."""
    print("=" * 50)
    print("MLX Complete Training Example")
    print("=" * 50)
    
    # Check MLX availability
    if not MLX_AVAILABLE and not args.dry_run:
        print("Error: MLX is required for training. Use --dry-run for testing without MLX.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model and tokenizer
    model_wrapper, tokenizer = create_model_and_tokenizer(args)
    
    # Load dataset
    train_dataset, eval_dataset = load_alpaca_dataset(tokenizer, args.num_samples)
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # MLX works best with single process
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Training configuration
    training_config = MLXTrainingConfig(
        model_config=model_wrapper.mlx_model.config,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
    )
    
    # Print configuration
    print("\nTraining Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Quantization: {args.use_quantization} ({args.quantization_bits}-bit)")
    print(f"  LoRA: {args.use_lora} (rank={args.lora_rank})")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Output directory: {args.output_dir}")
    
    # Get batch size recommendation
    model_size_gb = args.hidden_size * args.num_layers / 1e9 * 4  # Rough estimate
    max_batch = training_config.get_max_batch_size(model_size_gb, "m1_ultra")
    if args.batch_size > max_batch:
        print(f"\nWarning: Batch size {args.batch_size} may be too large.")
        print(f"Recommended max for this model: {max_batch}")
    
    if args.dry_run:
        print("\nDry run mode - skipping actual training")
        return
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = MLXTrainer(
        model=model_wrapper,
        config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
    
    # Run training
    print("\nStarting training...")
    print("-" * 50)
    
    try:
        history = trainer.train()
        
        # Print results
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        
        if history["train_losses"]:
            final_loss = history["train_losses"][-1][1]
            print(f"Final training loss: {final_loss:.4f}")
        
        if history["eval_losses"]:
            best_eval = history["best_eval_loss"]
            print(f"Best evaluation loss: {best_eval:.4f}")
        
        # Save final model
        final_path = os.path.join(args.output_dir, "final_model")
        print(f"\nSaving final model to {final_path}")
        model_wrapper.save_pretrained(final_path)
        
        # Save training history
        history_path = os.path.join(args.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete MLX training example for FSDP QLoRA"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="simple-llama",
        help="Model name for configuration",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="gpt2",
        help="Tokenizer to use (HuggingFace name)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=768,
        help="Hidden size of the model",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    
    # Quantization arguments
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Enable quantization",
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA adapters",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=32.0,
        help="LoRA alpha (scaling factor)",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    
    # Data arguments
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to use from dataset",
    )
    
    # Logging arguments
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Steps between logging",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Steps between checkpoints",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Steps between evaluations",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mlx_training_output",
        help="Output directory for checkpoints",
    )
    
    # Other arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actual training (for testing)",
    )
    
    args = parser.parse_args()
    main(args)