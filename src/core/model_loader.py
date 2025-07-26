"""
Model Loading Abstraction Layer for FSDP QLoRA

This module provides a unified interface for loading models across different backends
with support for various quantization methods and memory-efficient loading strategies.
"""

import enum
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from glob import glob

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Set up logging
logger = logging.getLogger(__name__)
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, hub
from accelerate import init_empty_weights
from fastcore.parallel import parallel
import safetensors.torch

from src.core.backend_manager import Backend, BackendManager
from src.core.quantization_wrapper import (
    QuantizationConfig,
    QuantizationMethod,
    create_quantization_adapter,
    get_recommended_config,
)
from src.core.imports import get_module, check_import_availability


class LoadingStrategy(enum.Enum):
    """Model loading strategies."""
    FULL = "full"           # Load full model to device
    LOW_MEMORY = "low_memory"  # Load to CPU/meta for memory efficiency
    UNIFIED_MEMORY = "unified_memory"  # For Apple Silicon unified memory
    STREAMING = "streaming"    # Load weights on demand


@dataclass
class ModelLoadingConfig:
    """Configuration for model loading."""
    
    model_name: str
    backend: Backend
    loading_strategy: LoadingStrategy = LoadingStrategy.FULL
    quantization_config: Optional[QuantizationConfig] = None
    dtype: torch.dtype = torch.float16
    device: Optional[Union[str, torch.device]] = None
    
    # Memory settings
    low_memory: bool = False
    offload_to_cpu: bool = False
    offload_to_disk: bool = False
    max_memory: Optional[Dict[str, str]] = None
    
    # Performance settings
    loading_workers: int = -1  # -1 for auto
    use_safetensors: bool = True
    use_fast_tokenizer: bool = True
    
    # Model settings
    use_cache: bool = False
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    
    # Quantization-specific
    skip_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    quant_method: Optional[str] = None  # "bnb" or "hqq"
    
    # Advanced settings
    llama_pro_path: Optional[str] = None
    rank: int = 0
    world_size: int = 1
    verbose: bool = False
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.device is None:
            if self.backend == Backend.CUDA:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif self.backend == Backend.MPS:
                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            elif self.backend == Backend.MLX:
                self.device = torch.device("cpu")  # MLX uses CPU device in PyTorch
            else:
                self.device = torch.device("cpu")
        
        # Auto-select loading strategy based on backend
        if self.backend in [Backend.MPS, Backend.MLX] and self.loading_strategy == LoadingStrategy.FULL:
            self.loading_strategy = LoadingStrategy.UNIFIED_MEMORY
        
        # Set quantization method from config if not specified
        if self.quantization_config and not self.quant_method:
            if self.quantization_config.method in [QuantizationMethod.BNB_NF4, QuantizationMethod.BNB_INT8]:
                self.quant_method = "bnb"
            elif self.quantization_config.method == QuantizationMethod.HQQ:
                self.quant_method = "hqq"


class ModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    def __init__(self, config: ModelLoadingConfig):
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self):
        """Validate configuration for this loader."""
        pass
    
    @abstractmethod
    def load_model(self) -> nn.Module:
        """Load the model according to configuration."""
        pass
    
    @abstractmethod
    def load_tokenizer(self) -> Any:
        """Load the tokenizer."""
        pass
    
    def _get_model_files(self) -> List[str]:
        """Get list of model weight files."""
        if self.config.llama_pro_path:
            # LLaMA Pro specific loading
            files = glob(str(Path(self.config.llama_pro_path) / "*.safetensors"))
        else:
            try:
                idx = hub.cached_file(self.config.model_name, SAFE_WEIGHTS_INDEX_NAME)
                files, _ = hub.get_checkpoint_shard_files(self.config.model_name, idx)
            except OSError:
                try:
                    # Model doesn't have index file (not sharded)
                    files = [hub.cached_file(self.config.model_name, SAFE_WEIGHTS_NAME)]
                except OSError as e:
                    # No safetensors file found
                    raise RuntimeError(f"Could not find model weights for {self.config.model_name}") from e
        
        return files
    
    def _get_loading_workers(self, param_count: int) -> int:
        """Determine number of parallel loading workers."""
        if self.config.loading_workers != -1:
            return self.config.loading_workers
        
        # Auto-determine based on backend and model size
        if self.config.backend == Backend.CUDA and torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if self.config.quant_method == "bnb":
                return 1  # bitsandbytes doesn't parallelize well
            elif mem_gb > 79:
                return 8
            elif mem_gb > 39:
                return 4
            else:
                return 2
        elif self.config.backend in [Backend.MPS, Backend.MLX]:
            # Apple Silicon has unified memory, can use more workers
            return 4
        else:
            # CPU loading
            return 2
    
    def _create_model_config(self) -> Any:
        """Create model configuration."""
        cfg = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        cfg.use_cache = self.config.use_cache
        cfg._attn_implementation = self.config.attn_implementation
        
        # Handle LLaMA Pro layer expansion
        if self.config.llama_pro_path:
            llama_pro_path = Path(self.config.llama_pro_path)
            num_original_layers, num_expanded_layers = llama_pro_path.name.split(
                "blk_exp-"
            )[1].split("-")
            num_original_layers = int(num_original_layers)
            num_expanded_layers = int(num_expanded_layers)
            cfg.num_hidden_layers = num_expanded_layers
        
        return cfg
    
    def _print_loading_info(self):
        """Print loading information if verbose."""
        if self.config.rank == 0 and self.config.verbose:
            print(f"Loading model: {self.config.model_name}")
            print(f"Backend: {self.config.backend}")
            print(f"Loading strategy: {self.config.loading_strategy}")
            if self.config.quantization_config:
                print(f"Quantization: {self.config.quantization_config.method}")


class StandardModelLoader(ModelLoader):
    """Standard model loader for full/LoRA training without quantization."""
    
    def _validate_config(self):
        """Validate configuration."""
        if self.config.quantization_config and self.config.quantization_config.method != QuantizationMethod.NONE:
            warnings.warn("StandardModelLoader doesn't support quantization, ignoring quantization config")
    
    def load_model(self) -> nn.Module:
        """Load model using standard transformers method."""
        self._print_loading_info()
        
        if self.config.loading_strategy == LoadingStrategy.LOW_MEMORY:
            if self.config.rank == 0:
                # Rank 0 loads to CPU
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    use_cache=self.config.use_cache,
                    torch_dtype=self.config.dtype,
                    _attn_implementation=self.config.attn_implementation,
                    trust_remote_code=self.config.trust_remote_code,
                )
                model.to(dtype=self.config.dtype, device="cpu")
            else:
                # Other ranks create empty model
                cfg = self._create_model_config()
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_config(
                        cfg,
                        torch_dtype=self.config.dtype,
                        trust_remote_code=self.config.trust_remote_code,
                    )
        else:
            # Standard loading
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                use_cache=self.config.use_cache,
                torch_dtype=self.config.dtype,
                _attn_implementation=self.config.attn_implementation,
                device_map={"": self.config.device} if self.config.device else None,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            if self.config.device and str(self.config.device) != "meta":
                model.to(self.config.device)
        
        return model
    
    def load_tokenizer(self) -> Any:
        """Load tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=self.config.use_fast_tokenizer,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer


class QuantizedModelLoader(ModelLoader):
    """Model loader with quantization support."""
    
    def _validate_config(self):
        """Validate quantization configuration."""
        if not self.config.quantization_config:
            raise ValueError("QuantizedModelLoader requires quantization_config")
        
        # Import required modules based on quantization method
        if self.config.quant_method == "bnb":
            if not check_import_availability('bitsandbytes', str(self.config.backend)):
                raise ImportError("bitsandbytes is required for BNB quantization")
        elif self.config.quant_method == "hqq":
            if not check_import_availability('hqq', str(self.config.backend)):
                warnings.warn("HQQ not available, model loading may fail")
    
    def load_model(self) -> nn.Module:
        """Load and quantize model."""
        self._print_loading_info()
        
        # Create model config
        cfg = self._create_model_config()
        
        # Create empty model
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                cfg,
                torch_dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
            )
        
        # Replace linear layers with quantized versions
        model = self._replace_with_quantized_layers(model)
        
        # Load and quantize weights
        self._load_quantized_weights(model)
        
        # Set quantization flag
        if hasattr(model, 'is_loaded_in_4bit'):
            model.is_loaded_in_4bit = True
        
        return model
    
    def _replace_with_quantized_layers(self, model: nn.Module) -> nn.Module:
        """Replace linear layers with quantized versions."""
        # Use quantization_wrapper for this
        from train import replace_linear
        
        if self.config.quant_method == "hqq":
            hqq = get_module('hqq', str(self.config.backend))
            quant_config = hqq.BaseQuantizeConfig(
                nbits=self.config.quantization_config.bits,
                group_size=self.config.quantization_config.group_size,
                quant_zero=self.config.quantization_config.quant_zero,
                quant_scale=self.config.quantization_config.quant_scale,
                offload_meta=True,
                view_as_float=True,
            )
            model.model = replace_linear(
                model.model,
                hqq.HQQLinear,
                quant_config,
                device=self.config.device,
                compute_dtype=self.config.quantization_config.compute_dtype,
                del_orig=True,
                initialize=False,
                skip_modules=self.config.skip_modules,
            )
            hqq.HQQLinear.set_backend(hqq.HQQBackend.ATEN_BACKPROP)
        else:  # bnb
            bnb = get_module('bitsandbytes', str(self.config.backend))
            model.model = replace_linear(
                model.model,
                bnb.Linear4bit,
                compute_dtype=self.config.quantization_config.compute_dtype,
                quant_type=self.config.quantization_config.quant_type,
                quant_storage=self.config.dtype,
                skip_modules=self.config.skip_modules,
            )
        
        return model
    
    def _load_quantized_weights(self, model: nn.Module):
        """Load and quantize weights in parallel."""
        from train import load_and_quantize
        
        files = self._get_model_files()
        param_count = sum(p.numel() for p in model.named_parameters())
        n_workers = self._get_loading_workers(param_count)
        
        if self.config.rank == 0 and self.config.verbose:
            print(f"Total model params: {param_count:,}")
            print(f"Using {n_workers} workers for loading")
        
        # Skip loading these parameter names
        skip_names = ["inv_freq"]
        
        def load_and_quantize_parallel(name_param, model, **kwargs):
            name, param = name_param
            load_and_quantize(model, name, param, **kwargs)
        
        start_time = time.time()
        
        for filename in tqdm(
            files,
            desc="Loading & Quantizing Model Shards",
            disable=self.config.rank != 0,
        ):
            weights = safetensors.torch.load_file(filename)
            parallel(
                load_and_quantize_parallel,
                iter(weights.items()),
                n_workers=n_workers,
                threadpool=True,
                model=model,
                dtype=self.config.dtype,
                device=self.config.device,
                skip_names=skip_names,
                to_cpu=(self.config.low_memory and self.config.rank == 0),
                to_meta=(self.config.low_memory and self.config.rank != 0),
                verbose=self.config.verbose,
                quant_method=self.config.quant_method,
                is_dora=False,  # Set based on train_type
            )
        
        if self.config.rank == 0 and self.config.verbose:
            print(f"Loaded model weights in {time.time() - start_time:.3f} seconds")
        
        # Clear cache after loading
        if self.config.backend == Backend.CUDA and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_tokenizer(self) -> Any:
        """Load tokenizer."""
        return StandardModelLoader(self.config).load_tokenizer()


class CUDAModelLoader(QuantizedModelLoader):
    """CUDA-specific model loader with optimizations."""
    
    def _validate_config(self):
        """Validate CUDA-specific configuration."""
        super()._validate_config()
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Validate device
        if isinstance(self.config.device, int):
            if self.config.device >= torch.cuda.device_count():
                raise ValueError(f"Invalid CUDA device {self.config.device}")


class MPSModelLoader(QuantizedModelLoader):
    """MPS (Apple Silicon) specific model loader."""
    
    def _validate_config(self):
        """Validate MPS-specific configuration."""
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        
        # MPS doesn't support bfloat16
        if self.config.dtype == torch.bfloat16:
            warnings.warn("MPS doesn't support bfloat16, using float16 instead")
            self.config.dtype = torch.float16
        
        # Check quantization compatibility
        if self.config.quantization_config:
            if self.config.quantization_config.compute_dtype == torch.bfloat16:
                self.config.quantization_config.compute_dtype = torch.float16
    
    def load_model(self) -> nn.Module:
        """Load model with MPS optimizations."""
        # Use unified memory loading strategy
        self.config.loading_strategy = LoadingStrategy.UNIFIED_MEMORY
        
        # MPS benefits from different loading approach
        if self.config.quantization_config:
            # For quantized models, load to CPU first then move
            original_device = self.config.device
            self.config.device = "cpu"
            model = super().load_model()
            
            # Move to MPS after quantization
            model = model.to(original_device)
            return model
        else:
            return super().load_model()


class MLXModelLoader(ModelLoader):
    """MLX-specific model loader for Apple Silicon."""
    
    def _validate_config(self):
        """Validate MLX configuration."""
        if self.config.backend != Backend.MLX:
            raise ValueError("MLXModelLoader requires MLX backend")
        
        if not check_import_availability('mlx', 'mlx'):
            raise ImportError("MLX is not available")
        
        # MLX has its own model format
        warnings.warn("MLX uses its own model format. PyTorch models will need conversion.")
    
    def load_model(self) -> nn.Module:
        """Load model for MLX (returns PyTorch model for compatibility)."""
        # For now, return a standard PyTorch model
        # In a full implementation, this would convert to MLX format
        loader = StandardModelLoader(self.config)
        return loader.load_model()
    
    def load_tokenizer(self) -> Any:
        """Load tokenizer."""
        return StandardModelLoader(self.config).load_tokenizer()


class CPUModelLoader(QuantizedModelLoader):
    """CPU-specific model loader with memory optimizations."""
    
    def _validate_config(self):
        """Validate CPU configuration."""
        # CPU supports most operations but may be slow
        if self.config.quantization_config:
            if self.config.quant_method == "bnb":
                warnings.warn("bitsandbytes may not work well on CPU, consider using HQQ")
    
    def _get_loading_workers(self, param_count: int) -> int:
        """CPU can benefit from more parallelism."""
        if self.config.loading_workers != -1:
            return self.config.loading_workers
        
        import multiprocessing
        # Use half the CPU cores for loading
        return max(1, multiprocessing.cpu_count() // 2)


class ModelLoaderFactory:
    """Factory for creating appropriate model loaders."""
    
    @staticmethod
    def create_loader(
        config: ModelLoadingConfig,
        backend_manager: Optional[BackendManager] = None
    ) -> ModelLoader:
        """
        Create a model loader based on configuration and backend.
        
        Args:
            config: Model loading configuration
            backend_manager: Optional backend manager for auto-detection
            
        Returns:
            Appropriate ModelLoader instance
        """
        # Auto-detect backend if needed
        if backend_manager and config.backend is None:
            config.backend = backend_manager.backend
        
        # Determine if quantization is needed
        needs_quantization = (
            config.quantization_config and 
            config.quantization_config.method != QuantizationMethod.NONE
        )
        
        # Select loader based on backend and requirements
        if config.backend == Backend.CUDA:
            if needs_quantization:
                return CUDAModelLoader(config)
            else:
                return StandardModelLoader(config)
        
        elif config.backend == Backend.MPS:
            if needs_quantization:
                return MPSModelLoader(config)
            else:
                # Use MPS loader even for standard models for optimizations
                loader = MPSModelLoader(config)
                loader.config.quantization_config = None
                return loader
        
        elif config.backend == Backend.MLX:
            return MLXModelLoader(config)
        
        elif config.backend == Backend.CPU:
            if needs_quantization:
                return CPUModelLoader(config)
            else:
                return StandardModelLoader(config)
        
        else:
            # Fallback to standard loader
            warnings.warn(f"Unknown backend {config.backend}, using standard loader")
            return StandardModelLoader(config)


# Convenience functions

def load_model_and_tokenizer(
    model_name: str,
    backend: Optional[Union[str, Backend]] = None,
    quantization_config: Optional[QuantizationConfig] = None,
    **kwargs
) -> Tuple[nn.Module, Any]:
    """
    Convenience function to load model and tokenizer.
    
    Args:
        model_name: Name or path of the model
        backend: Backend to use (auto-detected if None)
        quantization_config: Optional quantization configuration
        **kwargs: Additional arguments for ModelLoadingConfig
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Convert string backend to enum
    if isinstance(backend, str):
        backend = Backend(backend)
    
    # Auto-detect backend if not specified
    if backend is None:
        backend_manager = BackendManager(verbose=kwargs.get('verbose', False))
        backend = backend_manager.backend
    
    # Create configuration
    config = ModelLoadingConfig(
        model_name=model_name,
        backend=backend,
        quantization_config=quantization_config,
        **kwargs
    )
    
    # Create loader and load
    loader = ModelLoaderFactory.create_loader(config)
    model = loader.load_model()
    tokenizer = loader.load_tokenizer()
    
    return model, tokenizer


def get_recommended_loader_config(
    model_name: str,
    backend: Backend,
    available_memory_gb: Optional[float] = None,
    **kwargs
) -> ModelLoadingConfig:
    """
    Get recommended loader configuration based on model and system.
    
    Args:
        model_name: Name of the model
        backend: Target backend
        available_memory_gb: Available memory in GB
        **kwargs: Additional configuration options
        
    Returns:
        Recommended ModelLoadingConfig
    """
    # Estimate model size
    # This is a simplified heuristic - in practice would query model info
    model_size_b = 7.0  # Default to 7B
    if "70b" in model_name.lower():
        model_size_b = 70.0
    elif "13b" in model_name.lower():
        model_size_b = 13.0
    elif "3b" in model_name.lower():
        model_size_b = 3.0
    
    # Get memory if not provided
    if available_memory_gb is None:
        if backend == Backend.CUDA and torch.cuda.is_available():
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            # Rough estimate for other backends
            available_memory_gb = 16.0
    
    # Get recommended quantization
    quant_config = get_recommended_config(backend, model_size_b, available_memory_gb)
    
    # Determine loading strategy
    if available_memory_gb < model_size_b * 2:  # Need at least 2x model size
        loading_strategy = LoadingStrategy.LOW_MEMORY
    elif backend in [Backend.MPS, Backend.MLX]:
        loading_strategy = LoadingStrategy.UNIFIED_MEMORY
    else:
        loading_strategy = LoadingStrategy.FULL
    
    # Create config
    config = ModelLoadingConfig(
        model_name=model_name,
        backend=backend,
        loading_strategy=loading_strategy,
        quantization_config=quant_config if quant_config.method != QuantizationMethod.NONE else None,
        low_memory=(loading_strategy == LoadingStrategy.LOW_MEMORY),
        **kwargs
    )
    
    return config


__all__ = [
    'LoadingStrategy',
    'ModelLoadingConfig',
    'ModelLoader',
    'StandardModelLoader',
    'QuantizedModelLoader',
    'CUDAModelLoader',
    'MPSModelLoader',
    'MLXModelLoader',
    'CPUModelLoader',
    'ModelLoaderFactory',
    'load_model_and_tokenizer',
    'get_recommended_loader_config',
]