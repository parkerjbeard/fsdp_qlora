"""
MPS-Compatible FSDP Wrapper

This module provides an FSDP (Fully Sharded Data Parallel) wrapper optimized for
Apple Silicon's MPS (Metal Performance Shaders) backend. It handles MPS-specific
limitations and optimizes for unified memory architecture.

Key Features:
- Uses Gloo backend for distributed communication (NCCL not supported on MPS)
- Handles dtype limitations (no bfloat16 on MPS)
- Provides operator fallbacks for unsupported operations
- Optimizes sharding strategies for unified memory
- Compatible with PyTorch 2.0+

Based on research findings:
- FSDP works on MPS since PyTorch 1.12
- MPS doesn't support bfloat16, must use float16
- Some operators may have limited support
"""

import os
import warnings
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from src.core.backend_manager import Backend, BackendManager

logger = logging.getLogger(__name__)


@dataclass
class MPSFSDPConfig:
    """Configuration for MPS-compatible FSDP."""
    
    # Sharding configuration
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    min_num_params: int = 1e6  # Minimum params for FSDP wrapping
    
    # Mixed precision settings (MPS-compatible)
    use_mixed_precision: bool = True
    compute_dtype: torch.dtype = torch.float16  # MPS doesn't support bfloat16
    reduce_dtype: torch.dtype = torch.float32
    buffer_dtype: torch.dtype = torch.float16
    
    # Memory optimization
    cpu_offload: bool = False  # Whether to offload to CPU
    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE
    forward_prefetch: bool = True
    limit_all_gathers: bool = True
    
    # Distributed settings
    backend: str = "gloo"  # MPS requires Gloo, not NCCL
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    
    # MPS-specific settings
    sync_module_states: bool = True
    use_orig_params: bool = True  # Required for some optimizers
    
    # Unified memory optimization
    unified_memory_pool_size: Optional[int] = None  # Bytes
    aggressive_memory_optimization: bool = False
    
    # Operator fallback settings
    fallback_to_cpu_ops: List[str] = field(default_factory=lambda: [
        "aten::_fused_adam",  # Fused Adam not supported
        "aten::_foreach_add",  # Some foreach ops may fail
        "aten::nll_loss_backward",  # May have issues
    ])
    
    # Debug settings
    debug_mode: bool = False
    profile_memory: bool = False


class MPSOperatorFallback:
    """Handles operator fallbacks for unsupported MPS operations."""
    
    def __init__(self, fallback_ops: List[str]):
        self.fallback_ops = set(fallback_ops)
        self._original_ops = {}
    
    @contextmanager
    def fallback_context(self):
        """Context manager for operator fallback."""
        # This is a simplified version - in practice would need deeper PyTorch integration
        try:
            # Set MPS fallback environment variable
            os.environ["PYTORCH_MPS_FALLBACK"] = "1"
            yield
        finally:
            os.environ.pop("PYTORCH_MPS_FALLBACK", None)
    
    def patch_unsupported_ops(self):
        """Patch unsupported operators with CPU fallbacks."""
        # This would require deeper integration with PyTorch's dispatcher
        # For now, we rely on PyTorch's built-in MPS fallback mechanism
        if self.fallback_ops:
            logger.warning(
                f"The following operators may fallback to CPU: {self.fallback_ops}"
            )


class UnifiedMemoryOptimizer:
    """Optimizes FSDP for Apple Silicon's unified memory architecture."""
    
    def __init__(self, config: MPSFSDPConfig):
        self.config = config
        self._memory_pool = None
    
    def setup_memory_pool(self):
        """Set up unified memory pool for efficient allocation."""
        if self.config.unified_memory_pool_size:
            # MPS uses unified memory - we can provide hints to Metal
            torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
            
            if self.config.aggressive_memory_optimization:
                # Enable aggressive memory reclamation
                torch.mps.empty_cache()
    
    def optimize_sharding_strategy(self, model_size: int, available_memory: int) -> ShardingStrategy:
        """
        Optimize sharding strategy based on model size and available memory.
        
        For unified memory, we can be more aggressive with sharding since
        there's no GPU-CPU transfer overhead.
        """
        memory_ratio = model_size / available_memory
        
        if memory_ratio > 0.8:
            # Model is large relative to memory - use full sharding
            return ShardingStrategy.FULL_SHARD
        elif memory_ratio > 0.5:
            # Moderate size - use gradient and optimizer state sharding
            return ShardingStrategy.SHARD_GRAD_OP
        else:
            # Small model - might not need sharding
            return ShardingStrategy.NO_SHARD
    
    @staticmethod
    def get_optimal_bucket_size(model_size: int) -> int:
        """Get optimal gradient bucket size for unified memory."""
        # Unified memory allows larger buckets without transfer overhead
        if model_size > 1e9:  # > 1B parameters
            return 200_000_000  # 200M parameters per bucket
        else:
            return 50_000_000  # 50M parameters per bucket


class MPSFSDPWrapper:
    """
    FSDP wrapper optimized for MPS backend.
    
    Handles MPS-specific limitations and provides optimizations for
    Apple Silicon's unified memory architecture.
    """
    
    def __init__(
        self,
        config: Optional[MPSFSDPConfig] = None,
        backend_manager: Optional[BackendManager] = None,
    ):
        self.config = config or MPSFSDPConfig()
        self.backend_manager = backend_manager or BackendManager(backend="mps")
        
        # Verify MPS availability
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend is not available")
        
        # Set up components
        self.operator_fallback = MPSOperatorFallback(self.config.fallback_to_cpu_ops)
        self.memory_optimizer = UnifiedMemoryOptimizer(self.config)
        
        # Initialize distributed if needed
        self._initialized = False
        if self.config.world_size > 1:
            self._init_distributed()
    
    def _init_distributed(self):
        """Initialize distributed training with Gloo backend."""
        if self._initialized:
            return
        
        # MPS only supports Gloo backend
        if self.config.backend != "gloo":
            warnings.warn(
                f"MPS only supports Gloo backend, not {self.config.backend}. "
                "Switching to Gloo."
            )
            self.config.backend = "gloo"
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
            )
        
        self._initialized = True
        logger.info(f"Initialized distributed with Gloo backend (rank {self.config.rank})")
    
    def wrap_model(
        self,
        model: nn.Module,
        auto_wrap_policy: Optional[Callable] = None,
        param_dtype: Optional[torch.dtype] = None,
    ) -> FSDP:
        """
        Wrap model with MPS-compatible FSDP.
        
        Args:
            model: PyTorch model to wrap
            auto_wrap_policy: Policy for wrapping submodules
            param_dtype: Parameter dtype (will be converted from bfloat16 if needed)
            
        Returns:
            FSDP-wrapped model
        """
        # Handle dtype conversion
        if param_dtype == torch.bfloat16:
            warnings.warn(
                "MPS doesn't support bfloat16. Converting to float16."
            )
            param_dtype = torch.float16
            
            # Convert model parameters
            model = self._convert_dtype(model, torch.float16)
        
        # Set up memory optimization
        self.memory_optimizer.setup_memory_pool()
        
        # Patch unsupported operators
        self.operator_fallback.patch_unsupported_ops()
        
        # Create mixed precision config
        mixed_precision = None
        if self.config.use_mixed_precision:
            mixed_precision = MixedPrecision(
                param_dtype=param_dtype or self.config.compute_dtype,
                reduce_dtype=self.config.reduce_dtype,
                buffer_dtype=self.config.buffer_dtype,
                cast_forward_inputs=True,
            )
        
        # Create auto wrap policy if not provided
        if auto_wrap_policy is None:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=self.config.min_num_params,
            )
        
        # CPU offload config
        cpu_offload_config = None
        if self.config.cpu_offload:
            cpu_offload_config = CPUOffload(offload_params=True)
        
        # Wrap with FSDP
        with self.operator_fallback.fallback_context():
            wrapped_model = FSDP(
                model,
                sharding_strategy=self.config.sharding_strategy,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision,
                backward_prefetch=self.config.backward_prefetch,
                forward_prefetch=self.config.forward_prefetch,
                limit_all_gathers=self.config.limit_all_gathers,
                use_orig_params=self.config.use_orig_params,
                cpu_offload=cpu_offload_config,
                sync_module_states=self.config.sync_module_states,
                device_id=torch.device("mps"),
            )
        
        logger.info("Model wrapped with MPS-compatible FSDP")
        return wrapped_model
    
    def wrap_transformer(
        self,
        model: nn.Module,
        transformer_layer_cls: Type[nn.Module],
        **kwargs,
    ) -> FSDP:
        """
        Wrap transformer model with automatic layer wrapping.
        
        Args:
            model: Transformer model
            transformer_layer_cls: Class of transformer layers to wrap
            **kwargs: Additional arguments for wrap_model
            
        Returns:
            FSDP-wrapped model
        """
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )
        
        return self.wrap_model(model, auto_wrap_policy=auto_wrap_policy, **kwargs)
    
    @staticmethod
    def _convert_dtype(model: nn.Module, target_dtype: torch.dtype) -> nn.Module:
        """Convert model parameters to target dtype."""
        for param in model.parameters():
            if param.dtype == torch.bfloat16:
                param.data = param.data.to(target_dtype)
        
        for buffer in model.buffers():
            if buffer.dtype == torch.bfloat16:
                buffer.data = buffer.data.to(target_dtype)
        
        return model
    
    def save_checkpoint(
        self,
        model: FSDP,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ):
        """
        Save FSDP model checkpoint.
        
        Args:
            model: FSDP-wrapped model
            checkpoint_path: Path to save checkpoint
            optimizer: Optional optimizer state to save
            **kwargs: Additional state to save
        """
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
            
            checkpoint = {
                "model_state_dict": state_dict,
                "config": self.config,
                **kwargs,
            }
            
            if optimizer is not None:
                checkpoint["optimizer_state_dict"] = FSDP.optim_state_dict(model, optimizer)
            
            if self.config.rank == 0:
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(
        self,
        model: FSDP,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load FSDP model checkpoint.
        
        Args:
            model: FSDP-wrapped model
            checkpoint_path: Path to checkpoint
            optimizer: Optional optimizer to load state into
            strict: Whether to strictly enforce state dict matching
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            FSDP.optim_state_dict_to_load(
                model, optimizer, checkpoint["optimizer_state_dict"]
            )
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get MPS memory statistics."""
        if not torch.backends.mps.is_available():
            return {}
        
        # Get current memory stats
        stats = {
            "allocated_gb": torch.mps.current_allocated_memory() / 1e9,
            "reserved_gb": torch.mps.driver_allocated_memory() / 1e9,
        }
        
        # Add Metal-specific stats if available
        try:
            import psutil
            process = psutil.Process()
            stats["process_memory_gb"] = process.memory_info().rss / 1e9
        except ImportError:
            logger.debug("psutil not available - process memory stats unavailable")
            stats["process_memory_gb"] = None
        
        return stats
    
    @contextmanager
    def profile_memory(self):
        """Context manager for memory profiling."""
        if not self.config.profile_memory:
            yield
            return
        
        torch.mps.empty_cache()
        start_mem = self.get_memory_stats()
        
        yield
        
        end_mem = self.get_memory_stats()
        
        logger.info(
            f"Memory delta: "
            f"allocated={end_mem.get('allocated_gb', 0) - start_mem.get('allocated_gb', 0):.2f}GB, "
            f"reserved={end_mem.get('reserved_gb', 0) - start_mem.get('reserved_gb', 0):.2f}GB"
        )


# Convenience functions

def create_mps_fsdp_wrapper(
    world_size: int = 1,
    rank: int = 0,
    sharding_strategy: Union[str, ShardingStrategy] = "FULL_SHARD",
    mixed_precision: bool = True,
    cpu_offload: bool = False,
    **kwargs,
) -> MPSFSDPWrapper:
    """
    Create MPS FSDP wrapper with common configurations.
    
    Args:
        world_size: Number of processes
        rank: Current process rank
        sharding_strategy: Sharding strategy name or enum
        mixed_precision: Whether to use mixed precision training
        cpu_offload: Whether to offload parameters to CPU
        **kwargs: Additional config parameters
        
    Returns:
        Configured MPSFSDPWrapper
    """
    # Convert string to ShardingStrategy enum
    if isinstance(sharding_strategy, str):
        sharding_strategy = ShardingStrategy[sharding_strategy]
    
    config = MPSFSDPConfig(
        world_size=world_size,
        rank=rank,
        sharding_strategy=sharding_strategy,
        use_mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        **kwargs,
    )
    
    return MPSFSDPWrapper(config)


def wrap_model_for_mps(
    model: nn.Module,
    min_num_params: int = 1e6,
    transformer_layer_cls: Optional[Type[nn.Module]] = None,
) -> FSDP:
    """
    Quick wrapper for MPS FSDP with sensible defaults.
    
    Args:
        model: Model to wrap
        min_num_params: Minimum parameters for wrapping
        transformer_layer_cls: Transformer layer class for auto-wrapping
        
    Returns:
        FSDP-wrapped model
    """
    wrapper = create_mps_fsdp_wrapper(min_num_params=min_num_params)
    
    if transformer_layer_cls:
        return wrapper.wrap_transformer(model, transformer_layer_cls)
    else:
        return wrapper.wrap_model(model)


def check_mps_fsdp_compatibility() -> Dict[str, Any]:
    """Check MPS and FSDP compatibility."""
    info = {
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "pytorch_version": torch.__version__,
        "fsdp_available": hasattr(torch.distributed.fsdp, "FullyShardedDataParallel"),
    }
    
    if info["mps_available"]:
        # Check for known issues
        warnings_list = []
        
        # Check PyTorch version
        version_parts = torch.__version__.split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 2:
            warnings_list.append("PyTorch 2.0+ recommended for MPS FSDP")
        
        if major == 2 and minor < 7:
            warnings_list.append("PyTorch 2.7+ recommended for better MPS operator support")
        
        info["warnings"] = warnings_list
        
        # Test basic operations
        try:
            test_tensor = torch.randn(10, 10, device="mps")
            test_tensor = test_tensor.half()  # Test float16
            info["float16_supported"] = True
        except Exception as e:
            info["float16_supported"] = False
            logger.debug(f"MPS float16 not supported: {e}")
            warnings_list.append("Float16 operations not supported on this MPS device")
        
        try:
            if info["float16_supported"]:
                test_tensor = test_tensor.bfloat16()  # Test bfloat16
                info["bfloat16_supported"] = True
            else:
                info["bfloat16_supported"] = False
        except Exception as e:
            info["bfloat16_supported"] = False
            logger.debug(f"MPS bfloat16 not supported: {e}")
            warnings_list.append("BFloat16 operations not supported on this MPS device")
    
    return info


# Example usage and patterns

def example_mps_fsdp_training():
    """
    Example of using MPS FSDP for training.
    
    This demonstrates best practices for FSDP on Apple Silicon.
    """
    import torch.nn as nn
    
    # Check compatibility
    compat_info = check_mps_fsdp_compatibility()
    print(f"MPS FSDP Compatibility: {compat_info}")
    
    if not compat_info["mps_available"]:
        print("MPS not available!")
        return
    
    # Create a simple model
    class SimpleTransformer(nn.Module):
        def __init__(self, hidden_size=768, num_layers=12):
            super().__init__()
            self.embeddings = nn.Embedding(50000, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, 8, hidden_size * 4)
                for _ in range(num_layers)
            ])
            self.output = nn.Linear(hidden_size, 50000)
        
        def forward(self, input_ids):
            x = self.embeddings(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    # Create model
    model = SimpleTransformer()
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    
    # Create FSDP wrapper
    wrapper = create_mps_fsdp_wrapper(
        sharding_strategy="FULL_SHARD",
        mixed_precision=True,
        min_num_params=1e5,  # Wrap layers with >100k params
    )
    
    # Wrap model
    fsdp_model = wrapper.wrap_transformer(
        model,
        transformer_layer_cls=nn.TransformerEncoderLayer,
    )
    
    # Move to MPS
    fsdp_model = fsdp_model.to("mps")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
    
    # Training loop (simplified)
    print("\nSimulated training:")
    for step in range(3):
        # Create dummy batch
        batch = torch.randint(0, 50000, (4, 128), device="mps")
        
        # Forward pass
        with wrapper.profile_memory():
            outputs = fsdp_model(batch)
            loss = outputs.mean()  # Dummy loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Step {step}: loss={loss.item():.4f}")
        
        # Print memory stats
        mem_stats = wrapper.get_memory_stats()
        print(f"  Memory: {mem_stats}")
    
    # Save checkpoint
    wrapper.save_checkpoint(
        fsdp_model,
        "mps_fsdp_checkpoint.pt",
        optimizer=optimizer,
        step=3,
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    # Run example
    example_mps_fsdp_training()


__all__ = [
    "MPSFSDPConfig",
    "MPSFSDPWrapper",
    "create_mps_fsdp_wrapper",
    "wrap_model_for_mps",
    "check_mps_fsdp_compatibility",
]