"""
Learning Rate Scheduler Utilities

This module provides various learning rate schedulers with configurable warmup support.
Supports cosine, linear, constant, and polynomial schedules.
"""

import math
import warnings
from functools import partial
from typing import Callable, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class SchedulerType:
    """Available scheduler types."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a constant learning rate (no warmup).
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        last_epoch: The index of the last epoch when resuming training.
        
    Returns:
        LambdaLR scheduler that maintains constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a constant learning rate preceded by a warmup period.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        last_epoch: The index of the last epoch when resuming training.
        
    Returns:
        LambdaLR scheduler with warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr
    set in the optimizer to 0, after a warmup period.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        last_epoch: The index of the last epoch when resuming training.
        
    Returns:
        LambdaLR scheduler with linear decay after warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of
    the cosine function between the initial lr set in the optimizer to min_lr_ratio * initial_lr,
    after a warmup period.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of waves in the cosine schedule (default: 0.5 for half-cosine).
        last_epoch: The index of the last epoch when resuming training.
        min_lr_ratio: The minimum learning rate as a ratio of the initial lr (default: 0).
        
    Returns:
        LambdaLR scheduler with cosine annealing after warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_lr = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of
    the cosine function with several hard restarts, after a warmup period.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of hard restarts to use.
        last_epoch: The index of the last epoch when resuming training.
        min_lr_ratio: The minimum learning rate as a ratio of the initial lr.
        
    Returns:
        LambdaLR scheduler with cosine annealing and hard restarts.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        if progress >= 1.0:
            return min_lr_ratio
        cycle_progress = progress * num_cycles % 1.0
        cosine_lr = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases as a polynomial decay
    from the initial lr set in the optimizer to end lr, after a warmup period.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        lr_end: The end learning rate.
        power: The power of the polynomial (default: 1.0 for linear).
        last_epoch: The index of the last epoch when resuming training.
        
    Returns:
        LambdaLR scheduler with polynomial decay after warmup.
    """
    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step >= num_training_steps:
            return lr_end / lr_init  # Minimum learning rate
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            
            # Handle edge case where warmup_steps == num_training_steps
            if decay_steps == 0:
                return lr_end / lr_init
                
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # As a ratio of initial lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_exponential_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    gamma: float = 0.99,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with an exponential decay of learning rate after warmup.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        gamma: Multiplicative factor of learning rate decay.
        last_epoch: The index of the last epoch when resuming training.
        
    Returns:
        LambdaLR scheduler with exponential decay after warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return gamma ** (current_step - num_warmup_steps)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None
) -> Union[LambdaLR, None]:
    """
    Unified API to get any scheduler by name.
    
    Args:
        name: The name of the scheduler to use.
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        scheduler_specific_kwargs: Additional keyword arguments specific to the scheduler.
        
    Returns:
        The scheduler instance or None for constant schedule without warmup.
    """
    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}

    # Handle warmup configuration
    if num_warmup_steps is None:
        num_warmup_steps = 0
        
    if name == SchedulerType.CONSTANT:
        if num_warmup_steps > 0:
            return get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps
            )
        return None  # No scheduler needed for constant LR without warmup
        
    elif name == SchedulerType.LINEAR:
        if num_training_steps is None:
            raise ValueError("num_training_steps is required for linear schedule")
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
    elif name == SchedulerType.COSINE:
        if num_training_steps is None:
            raise ValueError("num_training_steps is required for cosine schedule")
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=scheduler_specific_kwargs.get("num_cycles", 0.5),
            min_lr_ratio=scheduler_specific_kwargs.get("min_lr_ratio", 0.0)
        )
        
    elif name == SchedulerType.COSINE_WITH_RESTARTS:
        if num_training_steps is None:
            raise ValueError("num_training_steps is required for cosine with restarts schedule")
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=scheduler_specific_kwargs.get("num_cycles", 1),
            min_lr_ratio=scheduler_specific_kwargs.get("min_lr_ratio", 0.0)
        )
        
    elif name == SchedulerType.POLYNOMIAL:
        if num_training_steps is None:
            raise ValueError("num_training_steps is required for polynomial schedule")
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_end=scheduler_specific_kwargs.get("lr_end", 1e-7),
            power=scheduler_specific_kwargs.get("power", 1.0)
        )
        
    elif name == SchedulerType.EXPONENTIAL:
        return get_exponential_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            gamma=scheduler_specific_kwargs.get("gamma", 0.99)
        )
        
    else:
        raise ValueError(
            f"Unknown scheduler type: {name}. "
            f"Available schedulers: {list(vars(SchedulerType).values())}"
        )


def compute_warmup_steps(
    num_training_steps: int,
    warmup_steps: Optional[int] = None,
    warmup_ratio: Optional[float] = None,
    warmup_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None
) -> int:
    """
    Compute the number of warmup steps based on various possible inputs.
    
    Args:
        num_training_steps: Total number of training steps.
        warmup_steps: Explicit number of warmup steps (highest priority).
        warmup_ratio: Ratio of training steps to use for warmup.
        warmup_epochs: Number of epochs to use for warmup.
        steps_per_epoch: Number of steps per epoch (required if warmup_epochs is set).
        
    Returns:
        The computed number of warmup steps.
        
    Raises:
        ValueError: If multiple warmup configurations are provided or required args are missing.
    """
    # Count how many warmup parameters were provided
    warmup_params = sum(x is not None for x in [warmup_steps, warmup_ratio, warmup_epochs])
    
    if warmup_params == 0:
        # No warmup specified
        return 0
    elif warmup_params > 1:
        raise ValueError(
            "Only one of warmup_steps, warmup_ratio, or warmup_epochs should be specified"
        )
    
    if warmup_steps is not None:
        return warmup_steps
    elif warmup_ratio is not None:
        if not 0.0 <= warmup_ratio <= 1.0:
            raise ValueError(f"warmup_ratio must be between 0 and 1, got {warmup_ratio}")
        return int(num_training_steps * warmup_ratio)
    elif warmup_epochs is not None:
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided when using warmup_epochs")
        return warmup_epochs * steps_per_epoch
    
    return 0


def get_scheduler_with_config(
    optimizer: Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    warmup_steps: Optional[int] = None,
    warmup_ratio: Optional[float] = None,
    warmup_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    **scheduler_kwargs
) -> Union[LambdaLR, None]:
    """
    High-level function to get a scheduler with automatic warmup configuration.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        scheduler_type: The type of scheduler to use.
        num_training_steps: Total number of training steps.
        warmup_steps: Explicit number of warmup steps.
        warmup_ratio: Ratio of training steps to use for warmup.
        warmup_epochs: Number of epochs to use for warmup.
        steps_per_epoch: Number of steps per epoch.
        **scheduler_kwargs: Additional scheduler-specific arguments.
        
    Returns:
        The configured scheduler instance.
    """
    # Compute warmup steps
    num_warmup_steps = compute_warmup_steps(
        num_training_steps=num_training_steps,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        warmup_epochs=warmup_epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=scheduler_kwargs
    )


# Maintain backward compatibility
TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.EXPONENTIAL: get_exponential_decay_schedule_with_warmup,
}


__all__ = [
    "SchedulerType",
    "get_constant_schedule",
    "get_constant_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
    "get_cosine_schedule_with_warmup",
    "get_cosine_with_hard_restarts_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "get_exponential_decay_schedule_with_warmup",
    "get_scheduler",
    "compute_warmup_steps",
    "get_scheduler_with_config",
]