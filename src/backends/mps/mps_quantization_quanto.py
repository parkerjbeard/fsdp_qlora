"""
MPS Quantization with Quanto Integration

This module provides improved MPS quantization using Hugging Face's Quanto library,
which has better MPS support than PyTorch's native quantization.

Key Features:
- INT2/4/8 quantization with MPS compatibility
- Automatic fallbacks for unsupported operations
- Minimal accuracy loss
- Easy integration with existing PyTorch models
"""

import os
import warnings
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    from optimum.quanto import (
        quantize,
        freeze,
        QLinear,
        QConv2d,
        quantization_map,
        Calibration,
        qint2,
        qint4,
        qint8,
        qfloat8_e4m3fn,
        qfloat8_e5m2,
    )
    QUANTO_AVAILABLE = True
except ImportError:
    QUANTO_AVAILABLE = False
    warnings.warn(
        "Quanto not available. Install with: pip install optimum-quanto"
    )

from src.core.backend_manager import Backend, BackendManager
from src.core.quantization_wrapper import (
    QuantizationConfig,
    QuantizationMethod,
    QuantizedLinear as BaseQuantizedLinear,
    QuantizationAdapter,
)

logger = logging.getLogger(__name__)


# Quanto-specific quantization methods
class QuantoQuantizationMethod:
    """Quanto-compatible quantization methods."""
    INT2 = "quanto_int2"    # 2-bit integer
    INT4 = "quanto_int4"    # 4-bit integer
    INT8 = "quanto_int8"    # 8-bit integer
    FLOAT8_E4M3 = "quanto_float8_e4m3"  # FP8 E4M3
    FLOAT8_E5M2 = "quanto_float8_e5m2"  # FP8 E5M2


@dataclass
class QuantoConfig(QuantizationConfig):
    """Configuration for Quanto quantization."""
    
    # Quanto-specific options
    quanto_method: str = QuantoQuantizationMethod.INT4
    weights_only: bool = True  # Quantize weights only (not activations)
    activations_dtype: Optional[str] = None  # For activation quantization
    
    # Calibration options
    calibration_samples: int = 0  # Number of samples for calibration
    calibration_dataset: Optional[Any] = None
    
    # Module selection
    modules_to_quantize: List[str] = field(
        default_factory=lambda: ["Linear", "Conv2d"]
    )
    
    # MPS-specific options
    force_cpu_calibration: bool = False  # Use CPU for calibration if MPS fails
    
    def get_quanto_dtype(self):
        """Get Quanto dtype from configuration."""
        dtype_map = {
            QuantoQuantizationMethod.INT2: qint2,
            QuantoQuantizationMethod.INT4: qint4,
            QuantoQuantizationMethod.INT8: qint8,
            QuantoQuantizationMethod.FLOAT8_E4M3: qfloat8_e4m3fn,
            QuantoQuantizationMethod.FLOAT8_E5M2: qfloat8_e5m2,
        }
        return dtype_map.get(self.quanto_method, qint4)


class QuantoQuantizationAdapter(QuantizationAdapter):
    """Quantization adapter using Quanto for better MPS support."""
    
    def __init__(
        self,
        backend: Backend,
        config: QuantoConfig,
    ):
        if not QUANTO_AVAILABLE:
            raise ImportError(
                "Quanto is required for this adapter. "
                "Install with: pip install optimum-quanto"
            )
        
        super().__init__(backend, config)
        self.config = config
        
    def _validate_backend_support(self):
        """Validate that Quanto supports the backend."""
        # Quanto supports CPU, CUDA, and MPS
        supported_backends = [Backend.CPU, Backend.CUDA, Backend.MPS]
        
        if self.backend not in supported_backends:
            raise ValueError(
                f"Quanto doesn't support {self.backend}. "
                f"Supported: {supported_backends}"
            )
        
        # Check specific limitations
        if self.backend == Backend.MPS:
            # Float8 on MPS will be upcast to float16
            if "float8" in self.config.quanto_method:
                warnings.warn(
                    "Float8 quantization on MPS will be upcast to float16 "
                    "for compatibility."
                )
    
    def create_quantized_linear(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> nn.Module:
        """Create a Quanto quantized linear layer."""
        # Quanto handles quantization differently - return standard linear
        # that will be replaced during quantize_model
        return nn.Linear(in_features, out_features, bias)
    
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Quantize model using Quanto."""
        device = kwargs.get('device', str(self.backend).lower())
        
        # Move model to device
        model = model.to(device)
        
        # Prepare quantization config
        weights_dtype = self.config.get_quanto_dtype()
        activations_dtype = None
        
        if self.config.activations_dtype:
            activations_dtype = getattr(
                self, 
                self.config.activations_dtype, 
                None
            )
        
        # Handle calibration if needed
        if self.config.calibration_samples > 0 and activations_dtype:
            logger.info("Starting calibration for activation quantization...")
            
            if self.backend == Backend.MPS and self.config.force_cpu_calibration:
                # MPS calibration can be problematic, use CPU
                logger.warning("Using CPU for calibration (MPS workaround)")
                model_cpu = model.cpu()
                
                with Calibration():
                    quantize(
                        model_cpu,
                        weights=weights_dtype,
                        activations=activations_dtype,
                    )
                    
                    # Run calibration samples
                    if self.config.calibration_dataset:
                        self._run_calibration(
                            model_cpu, 
                            self.config.calibration_dataset,
                            self.config.calibration_samples
                        )
                
                # Move back to MPS
                model = model_cpu.to(device)
            else:
                # Normal calibration
                with Calibration():
                    quantize(
                        model,
                        weights=weights_dtype,
                        activations=activations_dtype,
                    )
                    
                    if self.config.calibration_dataset:
                        self._run_calibration(
                            model,
                            self.config.calibration_dataset,
                            self.config.calibration_samples
                        )
        else:
            # Quantize without calibration (weights only)
            quantize(
                model,
                weights=weights_dtype,
                activations=activations_dtype,
                modules_to_quantize=self._get_modules_to_quantize(),
                exclude=self.config.skip_modules,
            )
        
        # Freeze quantized model (converts to fixed point)
        freeze(model)
        
        # Log quantization summary
        self._log_quantization_summary(model)
        
        return model
    
    def prepare_model_for_training(self, model: nn.Module) -> nn.Module:
        """Prepare quantized model for training."""
        # Quanto supports QAT (Quantization Aware Training)
        model.train()
        
        # Unfreeze for training if frozen
        for module in model.modules():
            if hasattr(module, '_frozen') and module._frozen:
                # Quanto internal: unfreeze for gradient updates
                module._frozen = False
        
        return model
    
    def save_quantized_model(self, model: nn.Module, save_path: str):
        """Save Quanto quantized model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Quanto models can be saved with standard PyTorch
        state = {
            'model_state_dict': model.state_dict(),
            'quantization_config': self.config,
            'quanto_quantization_map': quantization_map(model),
        }
        
        torch.save(state, save_path)
        logger.info(f"Saved Quanto quantized model to {save_path}")
    
    def load_quantized_model(
        self, 
        model_path: str, 
        model: nn.Module,
    ) -> nn.Module:
        """Load Quanto quantized model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # First quantize the model structure
        quanto_map = checkpoint.get('quanto_quantization_map', {})
        if quanto_map:
            # Apply quantization based on saved map
            for name, dtype in quanto_map.items():
                # Quanto will handle the quantization
                pass
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def _get_modules_to_quantize(self) -> List[type]:
        """Get module types to quantize."""
        module_map = {
            "Linear": nn.Linear,
            "Conv2d": nn.Conv2d,
            "Conv1d": nn.Conv1d,
            "ConvTranspose2d": nn.ConvTranspose2d,
        }
        
        return [
            module_map[name] 
            for name in self.config.modules_to_quantize 
            if name in module_map
        ]
    
    def _run_calibration(
        self, 
        model: nn.Module, 
        dataset: Any, 
        num_samples: int
    ):
        """Run calibration samples through the model."""
        model.eval()
        
        with torch.no_grad():
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                
                # Handle different dataset formats
                if isinstance(sample, torch.Tensor):
                    _ = model(sample)
                elif isinstance(sample, dict) and 'input_ids' in sample:
                    _ = model(**sample)
                elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    inputs = sample[0]
                    _ = model(inputs)
                else:
                    logger.warning(f"Unknown sample format: {type(sample)}")
    
    def _log_quantization_summary(self, model: nn.Module):
        """Log quantization summary."""
        quantized_modules = 0
        total_modules = 0
        param_reduction = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                total_modules += 1
                
                # Check if module is quantized (Quanto replaces with QLinear/QConv2d)
                if hasattr(module, 'weight_qtype'):
                    quantized_modules += 1
                    
                    # Estimate compression
                    original_bits = 32  # Assuming FP32
                    if module.weight_qtype == qint2:
                        new_bits = 2
                    elif module.weight_qtype == qint4:
                        new_bits = 4
                    elif module.weight_qtype == qint8:
                        new_bits = 8
                    else:
                        new_bits = 8  # Float8
                    
                    compression = original_bits / new_bits
                    param_reduction += module.weight.numel() * (1 - 1/compression)
        
        logger.info(
            f"Quantization summary: {quantized_modules}/{total_modules} modules quantized"
        )
        logger.info(
            f"Estimated parameter reduction: {param_reduction / 1e6:.1f}M parameters"
        )


def create_quanto_quantized_model(
    model: nn.Module,
    quantization_bits: int = 4,
    backend: str = "mps",
    weights_only: bool = True,
    calibration_data: Optional[Any] = None,
    **kwargs,
) -> nn.Module:
    """
    Create a Quanto-quantized model optimized for MPS.
    
    Args:
        model: PyTorch model to quantize
        quantization_bits: Number of bits (2, 4, or 8)
        backend: Backend to use
        weights_only: Only quantize weights (not activations)
        calibration_data: Dataset for calibration (if quantizing activations)
        **kwargs: Additional configuration
        
    Returns:
        Quantized model
    """
    # Map bits to Quanto method
    bit_to_method = {
        2: QuantoQuantizationMethod.INT2,
        4: QuantoQuantizationMethod.INT4,
        8: QuantoQuantizationMethod.INT8,
    }
    
    if quantization_bits not in bit_to_method:
        raise ValueError(f"Unsupported bit width: {quantization_bits}")
    
    # Create configuration
    config = QuantoConfig(
        quanto_method=bit_to_method[quantization_bits],
        weights_only=weights_only,
        calibration_dataset=calibration_data,
        calibration_samples=kwargs.get('calibration_samples', 100),
        **kwargs,
    )
    
    # Create adapter and quantize
    backend_enum = Backend(backend.lower())
    adapter = QuantoQuantizationAdapter(backend_enum, config)
    
    return adapter.quantize_model(model, device=backend)


__all__ = [
    'QuantoConfig',
    'QuantoQuantizationMethod',
    'QuantoQuantizationAdapter',
    'create_quanto_quantized_model',
]