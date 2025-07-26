"""
Integration tests for learning rate schedulers in training.

Tests the integration of schedulers with the training loop.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.lr_scheduler import (
    SchedulerType,
    get_scheduler_with_config,
)


# Mock the train module functions we need
def mock_get_lr_scheduler(optimizer, dataloader, gradient_accumulation_steps, args):
    """Mock implementation of get_lr_scheduler from train.py."""
    from src.utils.lr_scheduler import get_scheduler_with_config
    
    num_training_steps = (
        args["num_epochs"] * len(dataloader) // gradient_accumulation_steps
    )
    
    # Determine warmup configuration
    if args.get("warmup_steps") is None and args.get("warmup_ratio") is None:
        if args["lr_scheduler"] != "constant":
            warmup_ratio = 0.1  # Default 10% warmup
        else:
            warmup_ratio = None
    else:
        warmup_ratio = args.get("warmup_ratio")
    
    # Prepare scheduler-specific kwargs
    scheduler_kwargs = {}
    if args["lr_scheduler"] == "cosine" or args["lr_scheduler"] == "cosine_with_restarts":
        scheduler_kwargs["min_lr_ratio"] = args.get("cosine_min_lr_ratio", 0.1)
        if args["lr_scheduler"] == "cosine_with_restarts":
            scheduler_kwargs["num_cycles"] = args.get("cosine_cycles", 3)
    
    # Get the scheduler
    lr_scheduler = get_scheduler_with_config(
        optimizer=optimizer,
        scheduler_type=args["lr_scheduler"],
        num_training_steps=num_training_steps,
        warmup_steps=args.get("warmup_steps"),
        warmup_ratio=warmup_ratio,
        steps_per_epoch=len(dataloader) // gradient_accumulation_steps,
        **scheduler_kwargs
    )
    
    return lr_scheduler, num_training_steps


@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    # Create dummy data
    num_samples = 100
    input_size = 10
    output_size = 2
    
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, output_size, (num_samples,))
    
    return TensorDataset(X, y)


@pytest.fixture
def simple_dataloader(simple_dataset):
    """Create a dataloader for testing."""
    return DataLoader(simple_dataset, batch_size=10, shuffle=True)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )


class TestSchedulerIntegration:
    """Test scheduler integration with training."""
    
    def test_constant_scheduler_training(self, simple_model, simple_dataloader):
        """Test training with constant scheduler."""
        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        
        args = {
            "lr_scheduler": "constant",
            "num_epochs": 2,
            "warmup_steps": None,
            "warmup_ratio": None,
        }
        
        scheduler, num_steps = mock_get_lr_scheduler(
            optimizer, simple_dataloader, 1, args
        )
        
        # Scheduler should be None for constant without warmup
        assert scheduler is None
        
        # Simulate training
        initial_lr = optimizer.param_groups[0]['lr']
        for epoch in range(args["num_epochs"]):
            for batch in simple_dataloader:
                # Simulate training step
                optimizer.zero_grad()
                # No scheduler step needed
                
        # LR should remain constant
        assert optimizer.param_groups[0]['lr'] == initial_lr
    
    def test_linear_scheduler_training(self, simple_model, simple_dataloader):
        """Test training with linear scheduler."""
        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        
        args = {
            "lr_scheduler": "linear",
            "num_epochs": 2,
            "warmup_steps": 20,
            "warmup_ratio": None,
        }
        
        scheduler, num_steps = mock_get_lr_scheduler(
            optimizer, simple_dataloader, 1, args
        )
        
        assert scheduler is not None
        assert num_steps == args["num_epochs"] * len(simple_dataloader)
        
        # Simulate training
        initial_lr = optimizer.param_groups[0]['lr']
        step = 0
        
        for epoch in range(args["num_epochs"]):
            for batch in simple_dataloader:
                # Simulate training step
                optimizer.zero_grad()
                scheduler.step()
                step += 1
                
                # During warmup
                if step <= args["warmup_steps"]:
                    expected_lr = initial_lr * step / args["warmup_steps"]
                    assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6
        
        # After training, LR should be lower
        assert optimizer.param_groups[0]['lr'] < initial_lr
    
    def test_cosine_scheduler_training(self, simple_model, simple_dataloader):
        """Test training with cosine scheduler."""
        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        
        args = {
            "lr_scheduler": "cosine",
            "num_epochs": 3,
            "warmup_ratio": 0.1,
            "cosine_min_lr_ratio": 0.05,
        }
        
        scheduler, num_steps = mock_get_lr_scheduler(
            optimizer, simple_dataloader, 1, args
        )
        
        assert scheduler is not None
        
        # Track LR values
        lr_values = []
        
        for epoch in range(args["num_epochs"]):
            for batch in simple_dataloader:
                optimizer.zero_grad()
                scheduler.step()
                lr_values.append(optimizer.param_groups[0]['lr'])
        
        # Check warmup phase
        warmup_steps = int(num_steps * args["warmup_ratio"])
        for i in range(1, warmup_steps):
            assert lr_values[i] > lr_values[i-1]  # Increasing during warmup
        
        # Check cosine decay phase
        # LR should generally decrease (with some tolerance for cosine curve)
        decay_start = warmup_steps + 5
        decay_end = len(lr_values) - 5
        assert lr_values[decay_end] < lr_values[decay_start]
        
        # Check minimum LR
        min_lr = min(lr_values[warmup_steps:])
        expected_min = optimizer.defaults['lr'] * args["cosine_min_lr_ratio"]
        assert abs(min_lr - expected_min) < 1e-6
    
    def test_scheduler_with_gradient_accumulation(self, simple_model, simple_dataloader):
        """Test scheduler with gradient accumulation."""
        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        
        gradient_accumulation_steps = 4
        args = {
            "lr_scheduler": "linear",
            "num_epochs": 2,
            "warmup_steps": 5,
        }
        
        scheduler, num_steps = mock_get_lr_scheduler(
            optimizer, simple_dataloader, gradient_accumulation_steps, args
        )
        
        # num_steps should be reduced by gradient accumulation
        expected_steps = args["num_epochs"] * len(simple_dataloader) // gradient_accumulation_steps
        assert num_steps == expected_steps
        
        # Simulate training with gradient accumulation
        step = 0
        for epoch in range(args["num_epochs"]):
            for i, batch in enumerate(simple_dataloader):
                optimizer.zero_grad()
                
                # Only step scheduler after accumulation
                if (i + 1) % gradient_accumulation_steps == 0:
                    scheduler.step()
                    step += 1
        
        # Should have stepped the correct number of times
        assert step == expected_steps


class TestSchedulerSaveLoad:
    """Test saving and loading scheduler state."""
    
    def test_save_load_scheduler_state(self, simple_model, simple_dataloader):
        """Test saving and loading scheduler state."""
        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        
        args = {
            "lr_scheduler": "cosine",
            "num_epochs": 5,
            "warmup_ratio": 0.2,
        }
        
        scheduler, num_steps = mock_get_lr_scheduler(
            optimizer, simple_dataloader, 1, args
        )
        
        # Train for some steps
        checkpoint_step = 50
        for _ in range(checkpoint_step):
            scheduler.step()
        
        # Save state
        checkpoint = {
            'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': checkpoint_step,
        }
        
        # Create new scheduler and load state
        new_optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        new_scheduler, _ = mock_get_lr_scheduler(
            new_optimizer, simple_dataloader, 1, args
        )
        
        new_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Continue training
        scheduler.step()
        new_scheduler.step()
        
        # Both should have the same LR
        assert optimizer.param_groups[0]['lr'] == new_optimizer.param_groups[0]['lr']


class TestSchedulerCLIIntegration:
    """Test CLI argument parsing and scheduler configuration."""
    
    def test_default_warmup_behavior(self):
        """Test default warmup behavior for different schedulers."""
        test_cases = [
            # (scheduler_type, should_have_warmup)
            ("constant", False),
            ("linear", True),
            ("cosine", True),
            ("polynomial", True),
        ]
        
        for scheduler_type, should_have_warmup in test_cases:
            args = {
                "lr_scheduler": scheduler_type,
                "num_epochs": 1,
                # No warmup specified
            }
            
            # Mock dataloader
            mock_dataloader = MagicMock()
            mock_dataloader.__len__.return_value = 100
            
            optimizer = optim.Adam(nn.Linear(10, 2).parameters(), lr=0.001)
            scheduler, _ = mock_get_lr_scheduler(
                optimizer, mock_dataloader, 1, args
            )
            
            if scheduler_type == "constant" and not should_have_warmup:
                assert scheduler is None
            else:
                assert scheduler is not None
    
    def test_explicit_warmup_steps(self):
        """Test explicit warmup steps override ratio."""
        args = {
            "lr_scheduler": "linear",
            "num_epochs": 1,
            "warmup_steps": 50,
            "warmup_ratio": 0.5,  # Should be ignored
        }
        
        mock_dataloader = MagicMock()
        mock_dataloader.__len__.return_value = 200
        
        optimizer = optim.Adam(nn.Linear(10, 2).parameters(), lr=0.001)
        
        # Patch compute_warmup_steps to verify it's called correctly
        with patch('src.utils.lr_scheduler.compute_warmup_steps') as mock_compute:
            mock_compute.return_value = 50
            
            scheduler, num_steps = mock_get_lr_scheduler(
                optimizer, mock_dataloader, 1, args
            )
            
            # Should use warmup_steps, not warmup_ratio
            mock_compute.assert_called_once()
            call_args = mock_compute.call_args[1]
            assert call_args['warmup_steps'] == 50
            assert call_args['warmup_ratio'] is None  # Should not pass ratio when steps is set


class TestSchedulerTypes:
    """Test all supported scheduler types."""
    
    @pytest.mark.parametrize("scheduler_type", [
        "constant", "linear", "cosine", "cosine_with_restarts", 
        "polynomial", "exponential"
    ])
    def test_all_scheduler_types(self, scheduler_type, simple_model, simple_dataloader):
        """Test that all scheduler types work correctly."""
        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        
        args = {
            "lr_scheduler": scheduler_type,
            "num_epochs": 1,
            "warmup_steps": 10,
            "cosine_min_lr_ratio": 0.1,
            "cosine_cycles": 2,
        }
        
        try:
            scheduler, num_steps = mock_get_lr_scheduler(
                optimizer, simple_dataloader, 1, args
            )
            
            # Run a few steps to ensure no errors
            if scheduler is not None:
                for _ in range(20):
                    scheduler.step()
                    
        except Exception as e:
            pytest.fail(f"Scheduler type '{scheduler_type}' failed: {str(e)}")


class TestRealTrainingSimulation:
    """Test with more realistic training simulation."""
    
    def test_full_training_simulation(self, simple_model):
        """Simulate a full training run with scheduler."""
        # Create a larger dataset
        num_samples = 1000
        dataset = TensorDataset(
            torch.randn(num_samples, 10),
            torch.randint(0, 2, (num_samples,))
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training configuration
        args = {
            "lr_scheduler": "cosine",
            "num_epochs": 5,
            "warmup_ratio": 0.1,
            "cosine_min_lr_ratio": 0.01,
            "lr": 0.001,
        }
        
        # Setup training
        optimizer = optim.Adam(simple_model.parameters(), lr=args["lr"])
        criterion = nn.CrossEntropyLoss()
        scheduler, num_steps = mock_get_lr_scheduler(
            optimizer, dataloader, 1, args
        )
        
        # Track metrics
        losses = []
        lrs = []
        
        # Training loop
        for epoch in range(args["num_epochs"]):
            epoch_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                # Forward pass
                outputs = simple_model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Step scheduler
                if scheduler is not None:
                    scheduler.step()
                
                # Track metrics
                epoch_loss += loss.item()
                lrs.append(optimizer.param_groups[0]['lr'])
            
            losses.append(epoch_loss / len(dataloader))
        
        # Verify training behavior
        # 1. Learning rate should follow expected pattern
        warmup_steps = int(num_steps * args["warmup_ratio"])
        
        # Check warmup
        for i in range(1, min(warmup_steps, len(lrs))):
            assert lrs[i] >= lrs[i-1], "LR should increase during warmup"
        
        # Check decay
        if len(lrs) > warmup_steps + 10:
            assert lrs[-1] < lrs[warmup_steps], "LR should decay after warmup"
        
        # 2. Loss should generally decrease
        if len(losses) > 2:
            assert losses[-1] < losses[0] * 1.5, "Loss should not increase dramatically"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_scheduler_with_zero_length_dataloader(self):
        """Test scheduler with empty dataloader."""
        empty_dataloader = DataLoader(TensorDataset(torch.tensor([]), torch.tensor([])))
        optimizer = optim.Adam(nn.Linear(10, 2).parameters())
        
        args = {
            "lr_scheduler": "linear",
            "num_epochs": 1,
            "warmup_steps": 10,
        }
        
        # Should handle gracefully
        scheduler, num_steps = mock_get_lr_scheduler(
            optimizer, empty_dataloader, 1, args
        )
        
        assert num_steps == 0
    
    def test_warmup_larger_than_training_steps(self, simple_dataloader):
        """Test when warmup is larger than total training steps."""
        optimizer = optim.Adam(nn.Linear(10, 2).parameters())
        
        args = {
            "lr_scheduler": "linear",
            "num_epochs": 1,
            "warmup_steps": 1000,  # Much larger than actual steps
        }
        
        scheduler, num_steps = mock_get_lr_scheduler(
            optimizer, simple_dataloader, 1, args
        )
        
        # Should still work, with entire training being warmup
        for _ in range(num_steps):
            scheduler.step()
        
        # LR should have increased but not reached peak
        assert optimizer.param_groups[0]['lr'] < optimizer.defaults['lr']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])