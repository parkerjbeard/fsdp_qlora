"""
MLX Training Loop for FSDP QLoRA

This module provides a native MLX training loop implementation with support for
gradient accumulation, MLX-specific optimizers, and memory-efficient training
on Apple Silicon.
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import json
import os

# Import MLX conditionally
try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn_mlx
    import mlx.optimizers as optim_mlx
    from mlx.utils import tree_flatten, tree_unflatten, tree_map
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    warnings.warn("MLX is not available. MLX trainer will have limited functionality.")

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.core.backend_manager import BackendManager
from src.backends.mlx.mlx_model_wrapper import MLXModel, MLXModelWrapper, MLXConfig


@dataclass
class MLXTrainingConfig:
    """Configuration for MLX training."""
    
    # Model configuration
    model_config: MLXConfig
    
    # Training hyperparameters
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Training settings
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Memory optimization
    gradient_checkpointing: bool = False
    mixed_precision: bool = True  # MLX handles this automatically
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./mlx_checkpoints"
    
    # Device settings
    seed: int = 42
    
    # Batch size limits for different Apple Silicon chips
    max_batch_sizes: Dict[str, Dict[int, int]] = field(default_factory=lambda: {
        "m1": {7: 2, 13: 1, 70: 0},      # M1/M2 base
        "m1_max": {7: 4, 13: 2, 70: 1},  # M1/M2 Max
        "m1_ultra": {7: 8, 13: 4, 70: 2}, # M1/M2/M3 Ultra
    })
    
    def get_max_batch_size(self, model_size_b: float, chip_type: str = "m1_ultra") -> int:
        """Get maximum recommended batch size for model and chip."""
        size_key = 7 if model_size_b <= 7 else (13 if model_size_b <= 13 else 70)
        return self.max_batch_sizes.get(chip_type, {}).get(size_key, 1)


class MLXOptimizer:
    """Wrapper for MLX optimizers with gradient accumulation support."""
    
    def __init__(
        self,
        optimizer: Any,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.accumulated_grads = {}
        self.accumulation_counter = 0
    
    def zero_grad(self):
        """Reset accumulated gradients."""
        self.accumulated_grads = {}
    
    def accumulate_gradients(self, grads: Dict[str, "mx.array"]):
        """Accumulate gradients for gradient accumulation."""
        if not self.accumulated_grads:
            # First accumulation - just copy
            self.accumulated_grads = tree_map(lambda x: x.astype(mx.float32), grads)
        else:
            # Add to existing gradients
            def add_grads(acc, new):
                return acc + new.astype(mx.float32)
            
            self.accumulated_grads = tree_map(add_grads, self.accumulated_grads, grads)
        
        self.accumulation_counter += 1
    
    def step(self) -> bool:
        """Perform optimizer step if accumulation is complete."""
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            # Average accumulated gradients
            def average_grad(g):
                return g / self.gradient_accumulation_steps
            
            averaged_grads = tree_map(average_grad, self.accumulated_grads)
            
            # Apply gradient clipping if specified
            if self.max_grad_norm is not None:
                averaged_grads = self._clip_gradients(averaged_grads)
            
            # Update parameters
            self.optimizer.update(self.optimizer.state, averaged_grads)
            
            # Reset accumulation
            self.zero_grad()
            self.accumulation_counter = 0
            return True
        
        return False
    
    def _clip_gradients(self, grads: Dict[str, "mx.array"]) -> Dict[str, "mx.array"]:
        """Clip gradients by global norm."""
        # Compute global norm
        grad_norm = mx.sqrt(
            mx.sum("mx.array"([mx.sum(g ** 2) for g in tree_flatten(grads)]))
        )
        
        # Clip if needed
        if grad_norm > self.max_grad_norm:
            scale = self.max_grad_norm / grad_norm
            grads = tree_map(lambda g: g * scale, grads)
        
        return grads
    
    @property
    def learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.learning_rate


class MLXLossComputer:
    """Compute various losses for language modeling in MLX."""
    
    @staticmethod
    def cross_entropy_loss(
        logits: "mx.array",
        labels: "mx.array",
        ignore_index: int = -100,
        reduction: str = "mean"
    ) -> "mx.array":
        """Compute cross-entropy loss for language modeling."""
        # Reshape logits and labels
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)
        
        # Create mask for valid positions
        mask = labels != ignore_index
        
        # Compute log probabilities
        log_probs = mx.log_softmax(logits, axis=-1)
        
        # Select log probs for correct labels
        label_log_probs = log_probs[mx.arange(labels.size), labels]
        
        # Apply mask and compute loss
        masked_loss = -label_log_probs * mask
        
        if reduction == "mean":
            # Mean over valid positions
            loss = mx.sum(masked_loss) / mx.maximum(mx.sum(mask), 1)
        elif reduction == "sum":
            loss = mx.sum(masked_loss)
        else:
            loss = masked_loss
        
        return loss
    
    @staticmethod
    def compute_perplexity(loss: "mx.array") -> "mx.array":
        """Compute perplexity from loss."""
        return mx.exp(loss)


class MLXTrainer:
    """MLX-native trainer for language models."""
    
    def __init__(
        self,
        model: Union[MLXModel, MLXModelWrapper],
        config: MLXTrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        backend_manager: Optional[BackendManager] = None,
    ):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for MLXTrainer. Install with: pip install mlx mlx-lm")
        
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.backend_manager = backend_manager or BackendManager(backend="mlx")
        
        # Initialize components
        self._setup_optimizer()
        self._setup_loss_computer()
        self._setup_directories()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float("inf")
        
        # Metrics tracking
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
    
    def _setup_optimizer(self):
        """Set up MLX optimizer with gradient accumulation."""
        # Get trainable parameters
        trainable_params = self._get_trainable_parameters()
        
        # Create MLX AdamW optimizer
        mlx_optimizer = optim_mlx.AdamW(
            learning_rate=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        
        # Initialize optimizer state
        mlx_optimizer.state = tree_map(lambda p: p, trainable_params)
        
        # Wrap with gradient accumulation support
        self.optimizer = MLXOptimizer(
            mlx_optimizer,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
        )
    
    def _setup_loss_computer(self):
        """Set up loss computation."""
        self.loss_computer = MLXLossComputer()
    
    def _setup_directories(self):
        """Create output directories."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "checkpoints"), exist_ok=True)
    
    def _get_trainable_parameters(self) -> Dict[str, "mx.array"]:
        """Get trainable parameters from model."""
        if isinstance(self.model, MLXModelWrapper):
            # Get MLX model from wrapper
            mlx_model = self.model.mlx_model
        else:
            mlx_model = self.model
        
        # Collect trainable parameters (LoRA parameters if using LoRA)
        trainable_params = {}
        
        def collect_params(module, prefix=""):
            """Recursively collect parameters."""
            for name, value in module.__dict__.items():
                if isinstance(value, "mx.array"):
                    # Check if this is a LoRA parameter
                    if "lora_a" in name or "lora_b" in name:
                        param_name = f"{prefix}.{name}" if prefix else name
                        trainable_params[param_name] = value
                elif isinstance(value, nn_mlx.Module):
                    # Recursively collect from submodules
                    new_prefix = f"{prefix}.{name}" if prefix else name
                    collect_params(value, new_prefix)
        
        collect_params(mlx_model)
        
        # If no LoRA parameters found, return all parameters
        if not trainable_params:
            trainable_params = tree_flatten(mlx_model.parameters())
            trainable_params = {f"param_{i}": p for i, p in enumerate(trainable_params)}
        
        return trainable_params
    
    def train(self) -> Dict[str, Any]:
        """Run the training loop."""
        print(f"Starting MLX training for {self.config.num_epochs} epochs")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch()
            
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} - Loss: {epoch_loss:.4f}")
            
            # Evaluation
            if self.eval_dataloader is not None and (epoch + 1) % self.config.eval_steps == 0:
                eval_loss = self.evaluate()
                print(f"Evaluation loss: {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best")
        
        # Save final checkpoint
        self.save_checkpoint("final")
        
        # Return training history
        return {
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "learning_rates": self.learning_rates,
            "best_eval_loss": self.best_eval_loss,
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        epoch_losses = []
        
        # Progress bar
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            disable=not (self.backend_manager.rank == 0),
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Perform training step
            loss = self._training_step(batch)
            epoch_losses.append(float(loss))
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{self.optimizer.learning_rate:.2e}"})
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                avg_loss = np.mean(epoch_losses[-self.config.logging_steps:])
                self.train_losses.append((self.global_step, avg_loss))
                self.learning_rates.append((self.global_step, self.optimizer.learning_rate))
            
            # Checkpointing
            if self.global_step % self.config.save_steps == 0 and self.global_step > 0:
                self.save_checkpoint(f"step_{self.global_step}")
            
            self.global_step += 1
        
        return np.mean(epoch_losses)
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> "mx.array":
        """Perform a single training step."""
        # Convert batch to MLX arrays
        mlx_batch = self._prepare_batch(batch)
        
        # Define loss function for automatic differentiation
        def loss_fn(params):
            # Update model parameters
            self._update_model_params(params)
            
            # Forward pass
            outputs = self._forward_pass(mlx_batch)
            
            # Compute loss
            loss = self.loss_computer.cross_entropy_loss(
                outputs["logits"],
                mlx_batch["labels"],
                ignore_index=-100
            )
            
            return loss
        
        # Get current parameters
        params = self._get_trainable_parameters()
        
        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(params)
        
        # Accumulate gradients
        self.optimizer.accumulate_gradients(grads)
        
        # Perform optimizer step if ready
        if self.optimizer.step():
            # Update learning rate if using scheduler
            self._update_learning_rate()
        
        return loss
    
    def _forward_pass(self, batch: Dict[str, "mx.array"]) -> Dict[str, "mx.array"]:
        """Perform forward pass through model."""
        if isinstance(self.model, MLXModelWrapper):
            # Use wrapper's forward method (handles conversion)
            outputs = self.model.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            # Convert outputs to MLX format
            return {
                "logits": "mx.array"(outputs["logits"].detach().cpu().numpy()),
                "loss": "mx.array"(outputs.get("loss", 0.0))
            }
        else:
            # Direct MLX model forward
            logits = self.model(batch["input_ids"])
            return {"logits": logits}
    
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, "mx.array"]:
        """Convert PyTorch batch to MLX arrays."""
        mlx_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # Convert to numpy then MLX
                np_array = value.detach().cpu().numpy()
                mlx_batch[key] = "mx.array"(np_array)
            else:
                mlx_batch[key] = value
        return mlx_batch
    
    def _update_model_params(self, params: Dict[str, "mx.array"]):
        """Update model parameters with new values."""
        if isinstance(self.model, MLXModelWrapper):
            mlx_model = self.model.mlx_model
        else:
            mlx_model = self.model
        
        # Update parameters
        for name, value in params.items():
            # Parse parameter path and update
            parts = name.split(".")
            module = mlx_model
            
            # Navigate to the parameter
            for part in parts[:-1]:
                if hasattr(module, part):
                    module = getattr(module, part)
            
            # Set the parameter
            if hasattr(module, parts[-1]):
                setattr(module, parts[-1], value)
    
    def _update_learning_rate(self):
        """Update learning rate with linear warmup and decay."""
        if self.global_step < self.config.warmup_steps:
            # Linear warmup
            lr = self.config.learning_rate * (self.global_step / self.config.warmup_steps)
        else:
            # Cosine decay after warmup
            progress = (self.global_step - self.config.warmup_steps) / (
                len(self.train_dataloader) * self.config.num_epochs - self.config.warmup_steps
            )
            lr = self.config.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
        
        self.optimizer.optimizer.learning_rate = lr
    
    def evaluate(self) -> float:
        """Evaluate the model."""
        if self.eval_dataloader is None:
            return float("inf")
        
        eval_losses = []
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            mlx_batch = self._prepare_batch(batch)
            
            # Forward pass without gradients
            with mx.no_grad():
                outputs = self._forward_pass(mlx_batch)
                
                # Compute loss
                loss = self.loss_computer.cross_entropy_loss(
                    outputs["logits"],
                    mlx_batch["labels"],
                    ignore_index=-100
                )
                
                eval_losses.append(float(loss))
        
        avg_loss = np.mean(eval_losses)
        self.eval_losses.append((self.global_step, avg_loss))
        
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints", name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        if isinstance(self.model, MLXModelWrapper):
            self.model.save_pretrained(checkpoint_dir)
        else:
            # Save MLX model directly
            model_params = tree_flatten(self.model.parameters())
            mx.save(os.path.join(checkpoint_dir, "model.npz"), model_params)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config.__dict__,
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "learning_rates": self.learning_rates,
        }
        
        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)
        
        print(f"Saved checkpoint: {checkpoint_dir}")
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints", name)
        
        # Load model
        if isinstance(self.model, MLXModelWrapper):
            self.model = MLXModelWrapper.from_pretrained(checkpoint_dir)
        else:
            # Load MLX model directly
            model_params = mx.load(os.path.join(checkpoint_dir, "model.npz"))
            # Update model parameters
            # This would need model-specific implementation
        
        # Load training state
        with open(os.path.join(checkpoint_dir, "training_state.json")) as f:
            training_state = json.load(f)
        
        self.global_step = training_state["global_step"]
        self.current_epoch = training_state["current_epoch"]
        self.best_eval_loss = training_state["best_eval_loss"]
        self.train_losses = training_state["train_losses"]
        self.eval_losses = training_state["eval_losses"]
        self.learning_rates = training_state["learning_rates"]
        
        print(f"Loaded checkpoint: {checkpoint_dir}")


# Convenience functions

def create_mlx_trainer(
    model: Union[MLXModel, MLXModelWrapper],
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    **kwargs
) -> MLXTrainer:
    """Create an MLX trainer with sensible defaults."""
    # Create config
    model_config = model.config if hasattr(model, 'config') else MLXConfig("default")
    
    config = MLXTrainingConfig(
        model_config=model_config,
        **kwargs
    )
    
    # Validate batch size for chip
    if hasattr(model_config, 'model_name'):
        # Estimate model size from name
        model_size = 7.0  # Default
        if "13b" in model_config.model_name.lower():
            model_size = 13.0
        elif "70b" in model_config.model_name.lower():
            model_size = 70.0
        
        max_batch_size = config.get_max_batch_size(model_size)
        if config.batch_size > max_batch_size:
            warnings.warn(
                f"Batch size {config.batch_size} may be too large for {model_size}B model. "
                f"Recommended max: {max_batch_size}"
            )
    
    return MLXTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )


def benchmark_mlx_training(
    model: Union[MLXModel, MLXModelWrapper],
    batch_sizes: List[int] = [1, 2, 4, 8],
    seq_length: int = 512,
    num_steps: int = 10,
) -> Dict[int, Dict[str, float]]:
    """Benchmark MLX training with different batch sizes."""
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        try:
            # Create dummy data
            dummy_batch = {
                "input_ids": mx.random.randint(0, 1000, (batch_size, seq_length)),
                "labels": mx.random.randint(0, 1000, (batch_size, seq_length)),
            }
            
            # Time forward and backward passes
            start_time = time.time()
            
            for _ in range(num_steps):
                # Forward pass
                outputs = model(dummy_batch["input_ids"])
                
                # Compute loss
                loss = mx.mean(outputs)
                
                # Backward pass (simplified)
                mx.grad(loss)
            
            # Compute metrics
            total_time = time.time() - start_time
            time_per_step = total_time / num_steps
            tokens_per_sec = (batch_size * seq_length * num_steps) / total_time
            
            # Memory usage (simplified - would need proper profiling)
            memory_gb = 0.0  # Placeholder
            
            results[batch_size] = {
                "time_per_step": time_per_step,
                "tokens_per_sec": tokens_per_sec,
                "memory_gb": memory_gb,
                "status": "success"
            }
            
        except Exception as e:
            results[batch_size] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"Failed at batch size {batch_size}: {e}")
    
    return results


__all__ = [
    "MLXTrainingConfig",
    "MLXOptimizer",
    "MLXLossComputer",
    "MLXTrainer",
    "create_mlx_trainer",
    "benchmark_mlx_training",
]