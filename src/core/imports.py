"""
Import Abstraction Layer for FSDP QLoRA

This module provides a conditional import system that handles backend-specific
dependencies gracefully. It includes:
- Try-except wrapped imports for optional dependencies
- Fallback mechanisms for missing libraries
- Backend-specific import registry
- Clear error reporting and validation
"""

import contextlib
import io
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ImportResult:
    """Result of an import attempt."""
    
    success: bool
    module: Optional[Any] = None
    error: Optional[Exception] = None
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)


class ImportRegistry:
    """Registry for backend-specific imports and their fallbacks."""
    
    def __init__(self):
        self._registry = {}
        self._fallbacks = {}
        self._validators = {}
        self._available_cache = {}
        
    def register(
        self,
        name: str,
        import_func: Callable[[], Any],
        fallback: Optional[Callable[[], Any]] = None,
        validator: Optional[Callable[[Any], bool]] = None,
        backends: Optional[list[str]] = None
    ):
        """Register an import with optional fallback and validator.
        
        Args:
            name: Name of the import (e.g., 'bitsandbytes')
            import_func: Function that performs the import
            fallback: Optional fallback function if import fails
            validator: Optional function to validate the imported module
            backends: List of backends this import is valid for (None = all)
        """
        self._registry[name] = {
            'import_func': import_func,
            'backends': backends or ['all']
        }
        if fallback:
            self._fallbacks[name] = fallback
        if validator:
            self._validators[name] = validator
    
    def get(self, name: str, backend: Optional[str] = None) -> ImportResult:
        """Get an import by name, with automatic fallback handling.
        
        Args:
            name: Name of the import
            backend: Current backend (cuda, mps, mlx, cpu)
            
        Returns:
            ImportResult with the module or fallback
        """
        # Check cache first
        cache_key = f"{name}_{backend or 'all'}"
        if cache_key in self._available_cache:
            return self._available_cache[cache_key]
        
        if name not in self._registry:
            error = ImportError(f"No import registered for '{name}'")
            result = ImportResult(success=False, error=error)
            self._available_cache[cache_key] = result
            return result
        
        entry = self._registry[name]
        
        # Check if import is valid for this backend
        if backend and 'all' not in entry['backends'] and backend not in entry['backends']:
            warning = f"Import '{name}' is not supported on backend '{backend}'"
            if name in self._fallbacks:
                try:
                    module = self._fallbacks[name]()
                    result = ImportResult(
                        success=True,
                        module=module,
                        fallback_used=True,
                        warnings=[warning]
                    )
                    self._available_cache[cache_key] = result
                    return result
                except Exception as e:
                    error = ImportError(f"Fallback for '{name}' failed: {e}")
                    result = ImportResult(success=False, error=error, warnings=[warning])
                    self._available_cache[cache_key] = result
                    return result
            else:
                result = ImportResult(
                    success=False,
                    error=ImportError(warning),
                    warnings=[warning]
                )
                self._available_cache[cache_key] = result
                return result
        
        # Try the main import
        try:
            module = entry['import_func']()
            
            # Validate if validator exists
            if name in self._validators:
                if not self._validators[name](module):
                    raise ImportError(f"Validation failed for '{name}'")
            
            result = ImportResult(success=True, module=module)
            self._available_cache[cache_key] = result
            return result
            
        except Exception as e:
            # Try fallback if available
            if name in self._fallbacks:
                try:
                    module = self._fallbacks[name]()
                    warnings_list = [f"Using fallback for '{name}': {str(e)}"]
                    result = ImportResult(
                        success=True,
                        module=module,
                        fallback_used=True,
                        warnings=warnings_list
                    )
                    self._available_cache[cache_key] = result
                    return result
                except Exception as fallback_error:
                    error = ImportError(
                        f"Import '{name}' failed: {e}\n"
                        f"Fallback also failed: {fallback_error}"
                    )
                    result = ImportResult(success=False, error=error)
                    self._available_cache[cache_key] = result
                    return result
            else:
                result = ImportResult(success=False, error=e)
                self._available_cache[cache_key] = result
                return result
    
    def check_availability(self, name: str, backend: Optional[str] = None) -> bool:
        """Check if an import is available without actually importing."""
        result = self.get(name, backend)
        return result.success


# Global import registry
_import_registry = ImportRegistry()


# Helper context managers
@contextlib.contextmanager
def suppress_output():
    """Suppress stdout/stderr temporarily."""
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


# Bitsandbytes imports
def _import_bitsandbytes():
    """Import bitsandbytes with output suppression."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*bitsandbytes.*GPU support.*")
        with suppress_output():
            import bitsandbytes as bnb
            
        # Patch missing attributes for non-CUDA systems
        if not hasattr(bnb.optim, "cadam32bit_grad_fp32"):
            class DummyOptimizer:
                cadam32bit_grad_fp32 = None
                cadam32bit = None
            
            if not hasattr(bnb, "optim"):
                bnb.optim = DummyOptimizer()
            else:
                bnb.optim.cadam32bit_grad_fp32 = None
                bnb.optim.cadam32bit = None
                
        # Also import specific classes
        from bitsandbytes.nn import Linear4bit, Params4bit
        bnb.Linear4bit = Linear4bit
        bnb.Params4bit = Params4bit
        
        return bnb


def _bitsandbytes_fallback():
    """Fallback for bitsandbytes when not available."""
    class DummyBitsandbytes:
        """Dummy bitsandbytes module for non-CUDA systems."""
        
        class nn:
            class Linear4bit:
                """Dummy Linear4bit class."""
                pass
            
            class Params4bit:
                """Dummy Params4bit class."""
                pass
        
        class optim:
            cadam32bit_grad_fp32 = None
            cadam32bit = None
        
        Linear4bit = nn.Linear4bit
        Params4bit = nn.Params4bit
        
        def __getattr__(self, name):
            raise ImportError(
                f"bitsandbytes.{name} is not available. "
                "Please install bitsandbytes or use a different quantization method."
            )
    
    return DummyBitsandbytes()


def _validate_bitsandbytes(bnb):
    """Validate bitsandbytes import."""
    required_attrs = ['nn', 'optim', 'Linear4bit', 'Params4bit']
    return all(hasattr(bnb, attr) for attr in required_attrs)


# HQQ imports
def _import_hqq():
    """Import HQQ quantization library."""
    from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
    
    class HQQModule:
        HQQLinear = HQQLinear
        HQQBackend = HQQBackend
        BaseQuantizeConfig = BaseQuantizeConfig
    
    return HQQModule()


def _hqq_fallback():
    """Fallback for HQQ when not available."""
    class DummyHQQ:
        """Dummy HQQ module."""
        
        class HQQLinear:
            """Dummy HQQLinear class."""
            pass
        
        class HQQBackend:
            """Dummy HQQBackend class."""
            PYTORCH = "pytorch"
            ATEN = "aten"
            
        class BaseQuantizeConfig:
            """Dummy BaseQuantizeConfig class."""
            pass
        
        def __getattr__(self, name):
            raise ImportError(
                f"hqq.{name} is not available. "
                "Please install HQQ or use a different quantization method."
            )
    
    return DummyHQQ()


# Wandb imports
def _import_wandb():
    """Import wandb for logging."""
    import wandb
    return wandb


def _wandb_fallback():
    """Fallback for wandb when not available."""
    class DummyWandb:
        """Dummy wandb module for when wandb is not installed."""
        
        @staticmethod
        def init(*args, **kwargs):
            """Dummy init function."""
            print("Warning: wandb not available, logging disabled")
            return None
        
        @staticmethod
        def log(*args, **kwargs):
            """Dummy log function."""
            pass
        
        @staticmethod
        def finish(*args, **kwargs):
            """Dummy finish function."""
            pass
        
        def __getattr__(self, name):
            # Return a no-op function for any attribute access
            return lambda *args, **kwargs: None
    
    return DummyWandb()


# Flash Attention imports
def _import_flash_attn():
    """Import flash attention."""
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    
    class FlashAttnModule:
        flash_attn_func = flash_attn_func
        flash_attn_varlen_func = flash_attn_varlen_func
    
    return FlashAttnModule()


def _flash_attn_fallback():
    """Fallback for flash attention."""
    class DummyFlashAttn:
        """Dummy flash attention module."""
        
        @staticmethod
        def flash_attn_func(*args, **kwargs):
            raise RuntimeError(
                "Flash Attention is not available. "
                "Please install flash-attn or disable flash attention."
            )
        
        flash_attn_varlen_func = flash_attn_func
    
    return DummyFlashAttn()


# Local module imports (DORA, LoRA)
def _import_dora():
    """Import DORA modules."""
    import dora
    
    class DORAModule:
        BNBDORA = dora.BNBDORA
        HQQDORA = dora.HQQDORA
        DORALayer = dora.DORALayer
        MagnitudeLayer = dora.MagnitudeLayer
    
    return DORAModule()


def _import_lora():
    """Import LoRA modules."""
    import lora
    
    class LORAModule:
        LORA = lora.LORA
    
    return LORAModule()


# Register all imports
def register_imports():
    """Register all conditional imports with the registry."""
    # Bitsandbytes (CUDA only)
    _import_registry.register(
        'bitsandbytes',
        _import_bitsandbytes,
        fallback=_bitsandbytes_fallback,
        validator=_validate_bitsandbytes,
        backends=['cuda']
    )
    
    # HQQ (all backends)
    _import_registry.register(
        'hqq',
        _import_hqq,
        fallback=_hqq_fallback,
        backends=['all']
    )
    
    # Wandb (all backends)
    _import_registry.register(
        'wandb',
        _import_wandb,
        fallback=_wandb_fallback,
        backends=['all']
    )
    
    # Flash Attention (CUDA only)
    _import_registry.register(
        'flash_attn',
        _import_flash_attn,
        fallback=_flash_attn_fallback,
        backends=['cuda']
    )
    
    # DORA (all backends)
    _import_registry.register(
        'dora',
        _import_dora,
        backends=['all']
    )
    
    # LoRA (all backends)
    _import_registry.register(
        'lora',
        _import_lora,
        backends=['all']
    )
    
    # MLX (Apple Silicon only)
    _import_registry.register(
        'mlx',
        _import_mlx,
        fallback=_mlx_fallback,
        backends=['mlx', 'mps']
    )


# MLX imports
def _import_mlx():
    """Import MLX framework."""
    import mlx
    import mlx.core as mx
    import mlx.nn as nn_mlx
    import mlx.optimizers as optim_mlx
    from mlx.utils import tree_unflatten, tree_flatten, tree_map
    
    class MLXModule:
        mlx = mlx
        mx = mx
        nn = nn_mlx
        optimizers = optim_mlx
        tree_unflatten = tree_unflatten
        tree_flatten = tree_flatten
        tree_map = tree_map
    
    return MLXModule()


def _mlx_fallback():
    """Fallback for MLX when not available."""
    class DummyMLX:
        """Dummy MLX module for when MLX is not installed."""
        
        def __getattr__(self, name):
            raise ImportError(
                f"mlx.{name} is not available. "
                "Please install MLX: pip install mlx mlx-lm"
            )
    
    return DummyMLX()


# Initialize registry
register_imports()


# Convenience functions
def get_module(name: str, backend: Optional[str] = None) -> Any:
    """Get a module with automatic fallback handling.
    
    Args:
        name: Module name (e.g., 'bitsandbytes', 'wandb')
        backend: Current backend (cuda, mps, mlx, cpu)
        
    Returns:
        The imported module or fallback
        
    Raises:
        ImportError: If import and fallback both fail
    """
    result = _import_registry.get(name, backend)
    if not result.success:
        raise result.error
    
    # Print warnings if any
    for warning in result.warnings:
        print(f"Import Warning: {warning}")
    
    return result.module


def check_import_availability(name: str, backend: Optional[str] = None) -> bool:
    """Check if an import is available.
    
    Args:
        name: Module name
        backend: Current backend
        
    Returns:
        True if import is available (including fallbacks)
    """
    return _import_registry.check_availability(name, backend)


def validate_imports(required_imports: list[str], backend: Optional[str] = None) -> dict[str, ImportResult]:
    """Validate a list of required imports.
    
    Args:
        required_imports: List of module names to validate
        backend: Current backend
        
    Returns:
        Dictionary mapping module names to ImportResults
    """
    results = {}
    for name in required_imports:
        results[name] = _import_registry.get(name, backend)
    return results


def report_import_status(backend: Optional[str] = None) -> str:
    """Generate a report of all registered imports and their status.
    
    Args:
        backend: Current backend
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "Import Status Report",
        "=" * 50,
        f"Backend: {backend or 'not specified'}",
        ""
    ]
    
    for name in sorted(_import_registry._registry.keys()):
        result = _import_registry.get(name, backend)
        status = "✓ Available" if result.success else "✗ Not Available"
        if result.fallback_used:
            status += " (fallback)"
        
        report_lines.append(f"{name:20} {status}")
        
        if result.warnings:
            for warning in result.warnings:
                report_lines.append(f"  ⚠ {warning}")
        
        if result.error and not result.success:
            report_lines.append(f"  ✗ {str(result.error)}")
    
    return "\n".join(report_lines)


# Export key functions and classes
__all__ = [
    'ImportResult',
    'ImportRegistry',
    'get_module',
    'check_import_availability',
    'validate_imports',
    'report_import_status',
    'suppress_output',
]