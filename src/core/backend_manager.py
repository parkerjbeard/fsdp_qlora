"""
Backend Manager Module for FSDP QLoRA

This module provides a unified interface for managing different compute backends (CUDA, MPS, MLX)
with automatic device detection and capability management.
"""

import enum
import platform
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


class Backend(enum.Enum):
    """Enumeration of supported compute backends."""

    CUDA = "cuda"
    MPS = "mps"
    MLX = "mlx"
    CPU = "cpu"  # Fallback option

    def __str__(self):
        """Return the string representation of the backend."""
        return self.value


@dataclass
class BackendCapabilities:
    """Defines the capabilities and limitations of each backend."""

    quantization_bits: list[int] = field(default_factory=list)
    supports_distributed: bool = False
    supports_fsdp: bool = False
    supports_bfloat16: bool = False
    supports_flash_attention: bool = False
    max_model_size: Optional[int] = None  # in billions of parameters
    distributed_backend: Optional[str] = None  # nccl, gloo, etc.
    notes: list[str] = field(default_factory=list)


class BackendManager:
    """Manages backend detection, selection, and capabilities."""

    # Define backend capabilities matrix
    CAPABILITIES: dict[Backend, BackendCapabilities] = {
        Backend.CUDA: BackendCapabilities(
            quantization_bits=[4, 8, 16],
            supports_distributed=True,
            supports_fsdp=True,
            supports_bfloat16=True,
            supports_flash_attention=True,
            max_model_size=None,  # Limited by GPU memory
            distributed_backend="nccl",
            notes=["Full feature support", "Requires NVIDIA GPU"],
        ),
        Backend.MPS: BackendCapabilities(
            quantization_bits=[8, 16],  # No 4-bit support yet
            supports_distributed=True,
            supports_fsdp=True,
            supports_bfloat16=False,  # MPS limitation
            supports_flash_attention=False,
            max_model_size=70,  # Reasonable limit for Apple Silicon
            distributed_backend="gloo",
            notes=[
                "No bfloat16 support",
                "Use float16 instead",
                "Limited operator support",
            ],
        ),
        Backend.MLX: BackendCapabilities(
            quantization_bits=[4, 8, 16],
            supports_distributed=False,  # MLX doesn't support distributed training yet
            supports_fsdp=False,
            supports_bfloat16=True,
            supports_flash_attention=True,  # MLX has efficient attention
            max_model_size=70,  # M3 Ultra can handle 70B models
            distributed_backend=None,
            notes=[
                "Optimized for Apple Silicon",
                "Unified memory architecture",
                "Requires MLX library",
            ],
        ),
        Backend.CPU: BackendCapabilities(
            quantization_bits=[8, 16],
            supports_distributed=True,
            supports_fsdp=True,
            supports_bfloat16=True,
            supports_flash_attention=False,
            max_model_size=7,  # Practical limit for CPU
            distributed_backend="gloo",
            notes=["Fallback option", "Very slow for training"],
        ),
    }

    def __init__(
        self, backend: Optional[Union[str, Backend]] = None, verbose: bool = True
    ):
        """
        Initialize the backend manager.

        Args:
            backend: Specific backend to use, or None for auto-detection
            verbose: Whether to print detection information
        """
        self.verbose = verbose
        self._available_backends = self._detect_available_backends()

        if backend is None:
            self.backend = self._auto_select_backend()
        else:
            if isinstance(backend, str):
                backend = Backend(backend.lower())
            self._validate_backend(backend)
            self.backend = backend

        self.device = self._get_device()
        self.capabilities = self.CAPABILITIES[self.backend]

        if self.verbose:
            self._print_backend_info()

    def _detect_available_backends(self) -> list[Backend]:
        """Detect which backends are available on the current system."""
        available = []

        # Check CUDA availability
        if torch.cuda.is_available():
            available.append(Backend.CUDA)

        # Check MPS availability (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available.append(Backend.MPS)

        # Check MLX availability
        if self._is_mlx_available():
            available.append(Backend.MLX)

        # CPU is always available
        available.append(Backend.CPU)

        return available

    def _is_mlx_available(self) -> bool:
        """Check if MLX is available."""
        import importlib.util

        # Check if MLX can be imported
        mlx_spec = importlib.util.find_spec("mlx")
        if mlx_spec is None:
            return False
        # Additional check for Apple Silicon
        return platform.system() == "Darwin" and platform.processor() == "arm"

    def _auto_select_backend(self) -> Backend:
        """Automatically select the best available backend."""
        # Priority order: CUDA > MLX > MPS > CPU
        priority_order = [Backend.CUDA, Backend.MLX, Backend.MPS, Backend.CPU]

        for backend in priority_order:
            if backend in self._available_backends:
                if self.verbose:
                    print(f"Auto-selected backend: {backend}")
                return backend

        # This should never happen as CPU is always available
        raise RuntimeError("No backend available")

    def _validate_backend(self, backend: Backend) -> None:
        """Validate that the requested backend is available."""
        if backend not in self._available_backends:
            available_str = ", ".join(str(b) for b in self._available_backends)
            raise ValueError(
                f"Backend '{backend}' is not available on this system. "
                f"Available backends: {available_str}"
            )

    def _get_device(self) -> torch.device:
        """Get the appropriate torch device for the selected backend."""
        if self.backend == Backend.CUDA:
            return torch.device("cuda")
        elif self.backend == Backend.MPS:
            return torch.device("mps")
        elif self.backend == Backend.CPU:
            return torch.device("cpu")
        elif self.backend == Backend.MLX:
            # MLX doesn't use torch devices, but we return CPU for compatibility
            return torch.device("cpu")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _print_backend_info(self) -> None:
        """Print information about the selected backend and its capabilities."""
        print(f"\n{'='*60}")
        print("Backend Manager - System Information")
        print(f"{'='*60}")
        print(f"Selected Backend: {self.backend}")
        print(
            f"Available Backends: {', '.join(str(b) for b in self._available_backends)}"
        )
        print(f"Device: {self.device}")
        print("\nCapabilities:")
        print(f"  - Quantization bits: {self.capabilities.quantization_bits}")
        print(f"  - Distributed training: {self.capabilities.supports_distributed}")
        print(f"  - FSDP support: {self.capabilities.supports_fsdp}")
        print(f"  - BFloat16 support: {self.capabilities.supports_bfloat16}")
        print(f"  - Flash attention: {self.capabilities.supports_flash_attention}")
        print(f"  - Max model size: {self.capabilities.max_model_size or 'No limit'} B")

        if self.capabilities.notes:
            print("\nNotes:")
            for note in self.capabilities.notes:
                print(f"  - {note}")

        # Print device-specific information
        if self.backend == Backend.CUDA:
            self._print_cuda_info()
        elif self.backend == Backend.MPS:
            self._print_mps_info()
        elif self.backend == Backend.MLX:
            self._print_mlx_info()

        print(f"{'='*60}\n")

    def _print_cuda_info(self) -> None:
        """Print CUDA-specific information."""
        if torch.cuda.is_available():
            print("\nCUDA Information:")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  - GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")

    def _print_mps_info(self) -> None:
        """Print MPS-specific information."""
        print("\nMPS Information:")
        print(f"  - Platform: {platform.platform()}")
        print(f"  - Processor: {platform.processor()}")

    def _print_mlx_info(self) -> None:
        """Print MLX-specific information."""
        print("\nMLX Information:")
        print(f"  - Platform: {platform.platform()}")
        print(f"  - Processor: {platform.processor()}")
        try:
            import mlx.core as mx

            print(f"  - Default device: {mx.default_device()}")
        except ImportError:
            print("  - MLX library not installed")
            logger.debug("MLX import failed - library not available")

    def supports_quantization(self, bits: int) -> bool:
        """Check if the backend supports a specific quantization bit width."""
        return bits in self.capabilities.quantization_bits

    def get_dtype(self, prefer_bfloat16: bool = True) -> torch.dtype:
        """Get the recommended dtype for the backend."""
        if prefer_bfloat16 and self.capabilities.supports_bfloat16:
            return torch.bfloat16
        else:
            return torch.float16

    def get_distributed_backend(self) -> Optional[str]:
        """Get the appropriate distributed backend for the current compute backend."""
        return self.capabilities.distributed_backend

    def validate_model_size(self, model_size_b: float) -> None:
        """
        Validate if the backend can handle a model of the given size.

        Args:
            model_size_b: Model size in billions of parameters

        Raises:
            ValueError: If the model is too large for the backend
        """
        if (
            self.capabilities.max_model_size
            and model_size_b > self.capabilities.max_model_size
        ):
            raise ValueError(
                f"Model size ({model_size_b}B parameters) exceeds the recommended "
                f"limit for {self.backend} ({self.capabilities.max_model_size}B parameters)"
            )

    def get_memory_info(self) -> dict[str, Union[int, float]]:
        """Get memory information for the current backend."""
        info = {}

        if self.backend == Backend.CUDA:
            if torch.cuda.is_available():
                info["total_memory"] = torch.cuda.get_device_properties(0).total_memory
                info["allocated_memory"] = torch.cuda.memory_allocated()
                info["reserved_memory"] = torch.cuda.memory_reserved()
                info["free_memory"] = info["total_memory"] - info["allocated_memory"]
        elif self.backend == Backend.MPS:
            # MPS doesn't provide detailed memory info through PyTorch yet
            info["note"] = "MPS memory info not available through PyTorch"
        elif self.backend == Backend.MLX:
            try:
                # MLX uses unified memory, so we report system memory
                import psutil

                mem = psutil.virtual_memory()
                info["total_memory"] = mem.total
                info["available_memory"] = mem.available
                info["used_memory"] = mem.used
                info["percent_used"] = mem.percent
            except ImportError:
                info["note"] = "Install psutil for memory info"

        return info

    def get_optimal_batch_size(
        self, model_size_b: float, sequence_length: int = 2048
    ) -> int:
        """
        Suggest an optimal batch size based on backend and model size.

        This is a heuristic and should be tuned for specific use cases.
        """
        # Base batch size recommendations
        base_batch_sizes = {
            7: {"cuda": 4, "mps": 2, "mlx": 8, "cpu": 1},
            13: {"cuda": 2, "mps": 1, "mlx": 4, "cpu": 1},
            70: {"cuda": 1, "mps": 1, "mlx": 2, "cpu": 1},
        }

        # Find closest model size
        closest_size = min(base_batch_sizes.keys(), key=lambda x: abs(x - model_size_b))

        # Get base batch size for backend
        backend_str = str(self.backend).lower()
        batch_size = base_batch_sizes[closest_size].get(backend_str, 1)

        # Adjust for sequence length
        if sequence_length > 2048:
            batch_size = max(1, batch_size // 2)

        return batch_size

    @classmethod
    def from_env(cls, verbose: bool = True) -> "BackendManager":
        """Create a BackendManager from environment variables."""
        import os

        backend_env = os.environ.get("FSDP_BACKEND", None)
        return cls(backend=backend_env, verbose=verbose)


# Convenience functions
def get_backend_manager(
    backend: Optional[Union[str, Backend]] = None, verbose: bool = True
) -> BackendManager:
    """Get a configured backend manager instance."""
    return BackendManager(backend=backend, verbose=verbose)


def detect_backend() -> Backend:
    """Quick function to detect the best available backend."""
    manager = BackendManager(verbose=False)
    return manager.backend


def get_device(backend: Optional[Union[str, Backend]] = None) -> torch.device:
    """Get the appropriate device for the given or auto-detected backend."""
    manager = BackendManager(backend=backend, verbose=False)
    return manager.device
