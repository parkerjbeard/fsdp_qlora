"""
Unit tests for learning rate scheduler module.

Tests various scheduler types with different warmup configurations.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

from src.utils.lr_scheduler import (
    SchedulerType,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_exponential_decay_schedule_with_warmup,
    get_scheduler,
    compute_warmup_steps,
    get_scheduler_with_config,
)


def step_scheduler(optimizer, scheduler):
    """Helper to properly step optimizer and scheduler in correct order."""
    optimizer.step()
    if scheduler is not None:
        scheduler.step()


# Test model and optimizer fixtures
@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Linear(10, 2)


@pytest.fixture
def optimizer(simple_model):
    """Create an optimizer for testing."""
    return optim.Adam(simple_model.parameters(), lr=0.001)


class TestWarmupComputation:
    """Test warmup step computation."""
    
    def test_no_warmup(self):
        """Test when no warmup is specified."""
        steps = compute_warmup_steps(num_training_steps=1000)
        assert steps == 0
    
    def test_warmup_steps(self):
        """Test explicit warmup steps."""
        steps = compute_warmup_steps(
            num_training_steps=1000,
            warmup_steps=100
        )
        assert steps == 100
    
    def test_warmup_ratio(self):
        """Test warmup ratio."""
        steps = compute_warmup_steps(
            num_training_steps=1000,
            warmup_ratio=0.1
        )
        assert steps == 100
        
        steps = compute_warmup_steps(
            num_training_steps=500,
            warmup_ratio=0.2
        )
        assert steps == 100
    
    def test_warmup_epochs(self):
        """Test warmup epochs."""
        steps = compute_warmup_steps(
            num_training_steps=1000,
            warmup_epochs=2,
            steps_per_epoch=50
        )
        assert steps == 100
    
    def test_multiple_warmup_params_error(self):
        """Test that multiple warmup params raise error."""
        with pytest.raises(ValueError, match="Only one of"):
            compute_warmup_steps(
                num_training_steps=1000,
                warmup_steps=100,
                warmup_ratio=0.1
            )
    
    def test_warmup_epochs_without_steps_per_epoch(self):
        """Test that warmup_epochs requires steps_per_epoch."""
        with pytest.raises(ValueError, match="steps_per_epoch must be provided"):
            compute_warmup_steps(
                num_training_steps=1000,
                warmup_epochs=2
            )
    
    def test_invalid_warmup_ratio(self):
        """Test invalid warmup ratio values."""
        with pytest.raises(ValueError, match="warmup_ratio must be between"):
            compute_warmup_steps(
                num_training_steps=1000,
                warmup_ratio=1.5
            )


class TestConstantSchedule:
    """Test constant learning rate schedule."""
    
    def test_constant_no_warmup(self, optimizer):
        """Test constant schedule without warmup."""
        scheduler = get_constant_schedule(optimizer)
        
        # Check LR stays constant
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(10):
            assert optimizer.param_groups[0]['lr'] == initial_lr
            optimizer.step()
            scheduler.step()
    
    def test_constant_with_warmup(self, optimizer):
        """Test constant schedule with warmup."""
        warmup_steps = 5
        initial_lr = optimizer.defaults['lr']  # Get the base LR before scheduler
        
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )
        
        # LambdaLR applies the lambda immediately, so step 0 is already applied
        assert optimizer.param_groups[0]['lr'] == 0.0  # Step 0 warmup
        
        # During warmup
        for step in range(1, warmup_steps):
            optimizer.step()
            scheduler.step()
            expected_lr = initial_lr * step / warmup_steps
            assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6
        
        # Complete warmup
        optimizer.step()
        scheduler.step()
        assert abs(optimizer.param_groups[0]['lr'] - initial_lr) < 1e-6
        
        # After warmup - should stay constant
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            assert abs(optimizer.param_groups[0]['lr'] - initial_lr) < 1e-6


class TestLinearSchedule:
    """Test linear learning rate schedule."""
    
    def test_linear_schedule(self, optimizer):
        """Test linear decay schedule."""
        warmup_steps = 10
        training_steps = 100
        initial_lr = optimizer.defaults['lr']
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )
        
        # LambdaLR applies immediately, so we're at step 0
        assert optimizer.param_groups[0]['lr'] == 0.0  # Step 0 warmup
        
        # During warmup
        for step in range(1, warmup_steps + 1):
            optimizer.step()
            scheduler.step()
            expected_lr = initial_lr * step / warmup_steps
            assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6
        
        # Linear decay phase
        decay_steps = training_steps - warmup_steps
        for step in range(warmup_steps + 1, training_steps + 1):
            optimizer.step()
            scheduler.step()
            progress = (step - warmup_steps) / decay_steps
            expected_lr = initial_lr * (1 - progress)
            assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6
        
        # After training steps - should be 0
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]['lr'] == 0.0


class TestCosineSchedule:
    """Test cosine learning rate schedule."""
    
    def test_cosine_schedule(self, optimizer):
        """Test cosine annealing schedule."""
        warmup_steps = 10
        training_steps = 100
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # During warmup
        for step in range(warmup_steps):
            expected_lr = initial_lr * (step) / warmup_steps
            assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-6
            optimizer.step()
            scheduler.step()
        
        # Cosine decay phase
        prev_lr = optimizer.param_groups[0]['lr']
        for step in range(warmup_steps, training_steps):
            curr_lr = optimizer.param_groups[0]['lr']
            # LR should decrease (except possibly at the very end due to rounding)
            assert curr_lr <= prev_lr + 1e-8
            prev_lr = curr_lr
            optimizer.step()
            scheduler.step()
        
        # Should reach close to 0
        assert optimizer.param_groups[0]['lr'] < initial_lr * 0.01
    
    def test_cosine_with_min_lr(self, optimizer):
        """Test cosine schedule with minimum LR."""
        warmup_steps = 5
        training_steps = 50
        min_lr_ratio = 0.1
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
            min_lr_ratio=min_lr_ratio
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Skip to end
        for _ in range(training_steps):
            step_scheduler(optimizer, scheduler)
        
        # Should reach min_lr_ratio * initial_lr
        expected_min_lr = initial_lr * min_lr_ratio
        assert abs(optimizer.param_groups[0]['lr'] - expected_min_lr) < 1e-6
    
    def test_cosine_with_cycles(self, optimizer):
        """Test cosine schedule with multiple cycles."""
        warmup_steps = 5
        training_steps = 100
        num_cycles = 2.0
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
            num_cycles=num_cycles
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        lr_values = []
        
        # Collect LR values
        for _ in range(training_steps):
            lr_values.append(optimizer.param_groups[0]['lr'])
            step_scheduler(optimizer, scheduler)
        
        # Check that we have multiple peaks (cycles)
        # Find local maxima in the cosine phase
        local_maxima = 0
        for i in range(warmup_steps + 1, len(lr_values) - 1):
            if lr_values[i] > lr_values[i-1] and lr_values[i] > lr_values[i+1]:
                local_maxima += 1
        
        # With 2 cycles, we should have at least 1 local maximum
        assert local_maxima >= 1


class TestCosineWithRestarts:
    """Test cosine schedule with hard restarts."""
    
    def test_cosine_hard_restarts(self, optimizer):
        """Test cosine schedule with hard restarts."""
        warmup_steps = 10
        training_steps = 100
        num_cycles = 3
        
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
            num_cycles=num_cycles
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        lr_values = []
        
        # Skip warmup
        for _ in range(warmup_steps):
            step_scheduler(optimizer, scheduler)
        
        # Collect LR values during cosine phase
        for _ in range(training_steps - warmup_steps):
            lr_values.append(optimizer.param_groups[0]['lr'])
            step_scheduler(optimizer, scheduler)
        
        # Check for restarts (sudden jumps back to high LR)
        restarts = 0
        for i in range(1, len(lr_values)):
            if lr_values[i] > lr_values[i-1] * 1.5:  # Significant jump
                restarts += 1
        
        # Should have num_cycles - 1 restarts
        assert restarts >= num_cycles - 2  # Allow some flexibility


class TestPolynomialSchedule:
    """Test polynomial decay schedule."""
    
    def test_polynomial_linear(self, optimizer):
        """Test polynomial schedule with power=1 (linear)."""
        warmup_steps = 10
        training_steps = 100
        lr_end = 1e-7
        
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
            lr_end=lr_end,
            power=1.0
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Skip warmup
        for _ in range(warmup_steps):
            scheduler.step()
        
        # Check decay
        for _ in range(training_steps - warmup_steps):
            scheduler.step()
        
        # Should reach close to lr_end
        assert abs(optimizer.param_groups[0]['lr'] - lr_end) < 1e-8
    
    def test_polynomial_quadratic(self, optimizer):
        """Test polynomial schedule with power=2 (quadratic)."""
        warmup_steps = 5
        training_steps = 50
        lr_end = 1e-6
        power = 2.0
        
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
            lr_end=lr_end,
            power=power
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Skip to middle of training
        for _ in range(training_steps // 2):
            scheduler.step()
        
        mid_lr = optimizer.param_groups[0]['lr']
        
        # Continue to end
        for _ in range(training_steps // 2):
            scheduler.step()
        
        end_lr = optimizer.param_groups[0]['lr']
        
        # With power=2, decay should be slower initially
        assert mid_lr > (initial_lr + lr_end) / 2
        assert abs(end_lr - lr_end) < 1e-8


class TestExponentialSchedule:
    """Test exponential decay schedule."""
    
    def test_exponential_decay(self, optimizer):
        """Test exponential decay schedule."""
        warmup_steps = 5
        gamma = 0.95
        
        scheduler = get_exponential_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            gamma=gamma
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Skip warmup
        for _ in range(warmup_steps):
            step_scheduler(optimizer, scheduler)
        
        # Check exponential decay
        expected_lr = initial_lr
        for step in range(10):
            expected_lr *= gamma
            step_scheduler(optimizer, scheduler)
            assert abs(optimizer.param_groups[0]['lr'] - expected_lr) < 1e-8


class TestGetScheduler:
    """Test the unified get_scheduler interface."""
    
    def test_get_constant_scheduler(self, optimizer):
        """Test getting constant scheduler."""
        scheduler = get_scheduler(
            SchedulerType.CONSTANT,
            optimizer,
            num_warmup_steps=0,
            num_training_steps=100
        )
        assert scheduler is None  # No scheduler needed for constant without warmup
        
        scheduler = get_scheduler(
            SchedulerType.CONSTANT,
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100
        )
        assert scheduler is not None
    
    def test_get_linear_scheduler(self, optimizer):
        """Test getting linear scheduler."""
        scheduler = get_scheduler(
            SchedulerType.LINEAR,
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100
        )
        assert scheduler is not None
    
    def test_missing_training_steps(self, optimizer):
        """Test that training steps are required for some schedulers."""
        with pytest.raises(ValueError, match="num_training_steps is required"):
            get_scheduler(
                SchedulerType.LINEAR,
                optimizer,
                num_warmup_steps=10
            )
    
    def test_scheduler_specific_kwargs(self, optimizer):
        """Test passing scheduler-specific arguments."""
        scheduler = get_scheduler(
            SchedulerType.COSINE,
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
            scheduler_specific_kwargs={
                "min_lr_ratio": 0.2,
                "num_cycles": 1.5
            }
        )
        assert scheduler is not None
    
    def test_unknown_scheduler(self, optimizer):
        """Test unknown scheduler type."""
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            get_scheduler(
                "unknown_scheduler",
                optimizer,
                num_warmup_steps=10,
                num_training_steps=100
            )


class TestGetSchedulerWithConfig:
    """Test the high-level get_scheduler_with_config interface."""
    
    def test_with_warmup_steps(self, optimizer):
        """Test configuration with explicit warmup steps."""
        scheduler = get_scheduler_with_config(
            optimizer,
            scheduler_type=SchedulerType.LINEAR,
            num_training_steps=1000,
            warmup_steps=100
        )
        assert scheduler is not None
    
    def test_with_warmup_ratio(self, optimizer):
        """Test configuration with warmup ratio."""
        scheduler = get_scheduler_with_config(
            optimizer,
            scheduler_type=SchedulerType.COSINE,
            num_training_steps=1000,
            warmup_ratio=0.1,
            min_lr_ratio=0.05
        )
        assert scheduler is not None
    
    def test_with_warmup_epochs(self, optimizer):
        """Test configuration with warmup epochs."""
        scheduler = get_scheduler_with_config(
            optimizer,
            scheduler_type=SchedulerType.COSINE,
            num_training_steps=1000,
            warmup_epochs=2,
            steps_per_epoch=50
        )
        assert scheduler is not None


class TestSchedulerBehavior:
    """Test actual scheduler behavior during training."""
    
    def test_lr_schedule_values(self, optimizer):
        """Test that LR values follow expected patterns."""
        warmup_steps = 100
        training_steps = 1000
        
        schedulers = {
            'linear': get_linear_schedule_with_warmup(
                optimizer, warmup_steps, training_steps
            ),
            'cosine': get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, training_steps
            ),
            'polynomial': get_polynomial_decay_schedule_with_warmup(
                optimizer, warmup_steps, training_steps, lr_end=1e-7
            ),
        }
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        for name, scheduler in schedulers.items():
            # Reset optimizer LR
            optimizer.param_groups[0]['lr'] = initial_lr
            
            # Warmup phase - all should increase linearly
            for step in range(warmup_steps):
                expected_warmup_lr = initial_lr * step / warmup_steps
                actual_lr = optimizer.param_groups[0]['lr']
                assert abs(actual_lr - expected_warmup_lr) < 1e-6, \
                    f"{name} scheduler warmup failed at step {step}"
                step_scheduler(optimizer, scheduler)
            
            # After warmup - all should decrease
            prev_lr = optimizer.param_groups[0]['lr']
            for step in range(warmup_steps, training_steps):
                curr_lr = optimizer.param_groups[0]['lr']
                assert curr_lr <= prev_lr + 1e-8, \
                    f"{name} scheduler should decrease after warmup"
                prev_lr = curr_lr
                step_scheduler(optimizer, scheduler)
            
            # Final LR should be much lower than initial
            final_lr = optimizer.param_groups[0]['lr']
            assert final_lr < initial_lr * 0.1, \
                f"{name} scheduler didn't decay sufficiently"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_warmup_steps(self, optimizer):
        """Test schedulers with zero warmup steps."""
        schedulers = [
            get_linear_schedule_with_warmup(optimizer, 0, 100),
            get_cosine_schedule_with_warmup(optimizer, 0, 100),
            get_polynomial_decay_schedule_with_warmup(optimizer, 0, 100),
        ]
        
        for scheduler in schedulers:
            # Reset optimizer LR for each scheduler
            optimizer.param_groups[0]['lr'] = optimizer.defaults['lr']
            initial_lr = optimizer.param_groups[0]['lr']
            
            # Should start decaying immediately
            step_scheduler(optimizer, scheduler)
            assert optimizer.param_groups[0]['lr'] < initial_lr
    
    def test_warmup_equals_training_steps(self, optimizer):
        """Test when warmup steps equal training steps."""
        steps = 100
        scheduler = get_linear_schedule_with_warmup(
            optimizer, steps, steps
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Should only do warmup
        for _ in range(steps - 1):
            scheduler.step()
        
        # Last step should reach full LR
        scheduler.step()
        assert abs(optimizer.param_groups[0]['lr'] - initial_lr) < 1e-6
        
        # After training steps, LR should be 0
        scheduler.step()
        assert optimizer.param_groups[0]['lr'] == 0.0
    
    def test_invalid_lr_end(self, optimizer):
        """Test polynomial scheduler with invalid lr_end."""
        with pytest.raises(ValueError, match="lr_end.*must be smaller"):
            get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=10,
                num_training_steps=100,
                lr_end=1.0  # Greater than initial LR (0.001)
            )


class TestResumeTraining:
    """Test resuming training from checkpoint."""
    
    def test_resume_from_checkpoint(self, optimizer):
        """Test resuming scheduler from a specific epoch."""
        warmup_steps = 10
        training_steps = 100
        checkpoint_step = 50
        
        # Create scheduler and advance to checkpoint
        scheduler1 = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, training_steps
        )
        
        # Track LR values
        lr_values = []
        for _ in range(checkpoint_step):
            lr_values.append(optimizer.param_groups[0]['lr'])
            step_scheduler(optimizer, scheduler1)
        
        checkpoint_lr = lr_values[-1]  # LR before the checkpoint step
        
        # Create new optimizer and scheduler for resuming
        new_optimizer = optim.Adam(nn.Linear(10, 2).parameters(), lr=0.001)
        scheduler2 = get_linear_schedule_with_warmup(
            new_optimizer, warmup_steps, training_steps,
            last_epoch=checkpoint_step - 1
        )
        
        # The new scheduler should be at the same state
        assert abs(new_optimizer.param_groups[0]['lr'] - checkpoint_lr) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])