"""
Unit tests for the unified quantization implementation.

Tests cover:
- Quantization and dequantization methods
- Multi-bit support (4, 8, 16)
- Group-wise quantization
- Scale/zero-point computation
- Backend selection and fallback mechanisms
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.utils.unified_quantization import (
    QuantizationBackend,
    UnifiedQuantizationConfig,
    UnifiedQuantizer,
    BackendSelector,
    quantize_model,
    compare_backends,
)


class TestUnifiedQuantizationConfig:
    """Test the configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = UnifiedQuantizationConfig()
        
        assert config.backend == QuantizationBackend.AUTO
        assert config.bits == 4
        assert config.group_size == 64
        assert config.skip_modules == ["lm_head"]
        assert config.embedding_bits == 8  # Set in __post_init__
        assert config.output_bits == 8  # Set in __post_init__
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = UnifiedQuantizationConfig(
            bits=8,
            group_size=128,
            embedding_bits=16,
            layer_bits={"transformer.h.0": 2},
            enable_lora=True,
            lora_rank=32,
        )
        
        assert config.bits == 8
        assert config.group_size == 128
        assert config.embedding_bits == 16
        assert config.layer_bits == {"transformer.h.0": 2}
        assert config.enable_lora is True
        assert config.lora_rank == 32


class TestBackendSelector:
    """Test backend selection logic."""
    
    @patch('src.utils.unified_quantization.torch.backends.mps.is_available')
    @patch('src.utils.unified_quantization.platform.system')
    @patch('src.utils.unified_quantization.platform.machine')
    def test_apple_silicon_mlx_available(self, mock_machine, mock_system, mock_mps):
        """Test selection on Apple Silicon with MLX available."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        mock_mps.return_value = True
        
        # Mock mlx module being available by mocking the import check in detect_hardware
        with patch.object(BackendSelector, 'detect_hardware') as mock_detect:
            mock_detect.return_value = {
                "platform": "Darwin",
                "machine": "arm64",
                "processor": "arm",
                "is_apple_silicon": True,
                "cuda_available": False,
                "mps_available": True,
                "mlx_available": True,
                "quanto_available": False,
                "bitsandbytes_available": False,
                "hqq_available": False,
            }
            
            config = UnifiedQuantizationConfig()
            backend = BackendSelector.select_backend(config)
            
            assert backend == QuantizationBackend.MLX
    
    @patch('src.utils.unified_quantization.torch.backends.mps.is_available')
    @patch('src.utils.unified_quantization.platform.system')
    @patch('src.utils.unified_quantization.platform.machine')
    def test_apple_silicon_quanto_fallback(self, mock_machine, mock_system, mock_mps):
        """Test falling back to Quanto when MLX not available."""
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        mock_mps.return_value = True
        
        # Mock MLX not available but Quanto is
        with patch.dict('sys.modules', {'mlx': None}), \
             patch.dict('sys.modules', {'optimum.quanto': MagicMock()}):
            
            hardware = BackendSelector.detect_hardware()
            assert hardware["mlx_available"] is False
            assert hardware["quanto_available"] is True
            
            config = UnifiedQuantizationConfig()
            backend = BackendSelector.select_backend(config)
            
            assert backend == QuantizationBackend.QUANTO
    
    def test_manual_backend_selection(self):
        """Test that manual backend selection is respected."""
        config = UnifiedQuantizationConfig(backend=QuantizationBackend.MPS_CUSTOM)
        backend = BackendSelector.select_backend(config)
        
        assert backend == QuantizationBackend.MPS_CUSTOM


class TestUnifiedQuantizer:
    """Test the main quantizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test model
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        
        # Create test tensors
        self.test_tensor = torch.randn(64, 128)
        self.test_weight = torch.randn(256, 128)
    
    @patch('src.utils.unified_quantization.BackendSelector.select_backend')
    def test_quantizer_initialization(self, mock_select_backend):
        """Test quantizer initialization."""
        mock_select_backend.return_value = QuantizationBackend.MPS_CUSTOM
        
        config = UnifiedQuantizationConfig(bits=8)
        quantizer = UnifiedQuantizer(config)
        
        assert quantizer.config == config
        assert quantizer.backend == QuantizationBackend.MPS_CUSTOM
        assert quantizer._framework == "pytorch"
    
    def test_quantize_symmetric(self):
        """Test symmetric quantization."""
        config = UnifiedQuantizationConfig(backend=QuantizationBackend.MPS_CUSTOM)
        quantizer = UnifiedQuantizer(config)
        
        # Test 8-bit symmetric quantization
        quantized, params = quantizer.quantize(
            self.test_weight,
            bits=8,
            symmetric=True,
        )
        
        assert quantized.dtype == torch.int8
        assert params["scales"] is not None
        assert params["zero_points"] is None  # Should be None for symmetric
        assert params["bits"] == 8
        assert params["symmetric"] is True
        
        # Check quantization range
        assert quantized.min() >= -128
        assert quantized.max() <= 127
    
    def test_quantize_asymmetric(self):
        """Test asymmetric quantization."""
        config = UnifiedQuantizationConfig(backend=QuantizationBackend.MPS_CUSTOM)
        quantizer = UnifiedQuantizer(config)
        
        # Test 8-bit asymmetric quantization
        quantized, params = quantizer.quantize(
            self.test_weight,
            bits=8,
            symmetric=False,
        )
        
        assert quantized.dtype == torch.uint8
        assert params["scales"] is not None
        assert params["zero_points"] is not None
        assert params["bits"] == 8
        assert params["symmetric"] is False
        
        # Check quantization range
        assert quantized.min() >= 0
        assert quantized.max() <= 255
    
    def test_quantize_multiple_bits(self):
        """Test quantization with different bit widths."""
        config = UnifiedQuantizationConfig(backend=QuantizationBackend.MPS_CUSTOM)
        quantizer = UnifiedQuantizer(config)
        
        for bits in [4, 8, 16]:
            quantized, params = quantizer.quantize(
                self.test_weight,
                bits=bits,
                symmetric=True,
            )
            
            assert params["bits"] == bits
            
            # Check dtype
            if bits == 8:
                assert quantized.dtype == torch.int8
            elif bits == 16:
                assert quantized.dtype == torch.int16
    
    def test_quantize_invalid_bits(self):
        """Test that invalid bit widths raise an error."""
        config = UnifiedQuantizationConfig(backend=QuantizationBackend.MPS_CUSTOM)
        quantizer = UnifiedQuantizer(config)
        
        with pytest.raises(ValueError, match="Unsupported bit width"):
            quantizer.quantize(self.test_weight, bits=3)
    
    def test_dequantize(self):
        """Test dequantization."""
        config = UnifiedQuantizationConfig(backend=QuantizationBackend.MPS_CUSTOM)
        quantizer = UnifiedQuantizer(config)
        
        # Quantize first
        quantized, params = quantizer.quantize(
            self.test_weight,
            bits=8,
            symmetric=True,
        )
        
        # Dequantize
        dequantized = quantizer.dequantize(quantized, params)
        
        assert dequantized.dtype == torch.float32
        assert dequantized.shape == self.test_weight.shape
        
        # Check that values are close to original (allowing for quantization error)
        max_error = torch.max(torch.abs(dequantized - self.test_weight))
        assert max_error < 0.1  # Reasonable error for 8-bit quantization
    
    def test_group_wise_quantization(self):
        """Test group-wise quantization."""
        config = UnifiedQuantizationConfig(backend=QuantizationBackend.MPS_CUSTOM)
        quantizer = UnifiedQuantizer(config)
        
        # Create a larger tensor for group-wise quantization
        large_tensor = torch.randn(512, 512)
        
        quantized, params = quantizer.quantize(
            large_tensor,
            bits=8,
            group_size=64,
            symmetric=True,
        )
        
        assert "group_size" in params
        assert params["group_size"] == 64
        assert params["scales"].numel() > 1  # Multiple scales for groups
        
        # Dequantize and check
        dequantized = quantizer.dequantize(quantized, params)
        assert dequantized.shape == large_tensor.shape
    
    def test_quantize_model_fallback(self):
        """Test model quantization with fallback implementation."""
        config = UnifiedQuantizationConfig(
            backend=QuantizationBackend.MPS_CUSTOM,
            bits=8,
        )
        quantizer = UnifiedQuantizer(config)
        
        # Create a mock adapter without quantize_model method
        mock_adapter = MagicMock(spec=[])  # Empty spec means no methods
        mock_adapter.backend = quantizer._adapter.backend
        mock_adapter.config = quantizer._adapter.config
        
        # Replace the adapter temporarily
        original_adapter = quantizer._adapter
        quantizer._adapter = mock_adapter
        
        try:
            quantized_model = quantizer.quantize_model(self.model)
            
            # The fallback implementation should return a model with quantization applied
            assert quantized_model is not None
            assert isinstance(quantized_model, nn.Module)
            
            # Check that linear layers have quantized weights
            has_quantized_layers = False
            for name, module in quantized_model.named_modules():
                if isinstance(module, nn.Linear):
                    assert hasattr(module, 'quantized_weight')
                    has_quantized_layers = True
            assert has_quantized_layers
            
        finally:
            # Restore the adapter
            quantizer._adapter = original_adapter
    
    def test_save_load_model(self):
        """Test saving and loading quantized models."""
        config = UnifiedQuantizationConfig(
            backend=QuantizationBackend.MPS_CUSTOM,
            bits=8,
        )
        quantizer = UnifiedQuantizer(config)
        
        # Quantize model
        quantized_model = quantizer.quantize_model(self.model)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "quantized_model")
            quantizer.save_model(quantized_model, save_path)
            
            # Check that save path exists (could be file or directory)
            assert os.path.exists(save_path) or os.path.exists(save_path + '.pt')
            
            # If it's a directory, check for files inside
            if os.path.isdir(save_path):
                files = os.listdir(save_path)
                assert len(files) > 0  # At least one file should be created
                
                # Check for model file (could be model.pt, quantized_model.pt, or model.pth)
                model_files = [f for f in files if f.endswith(('.pt', '.pth', '.bin'))]
                assert len(model_files) > 0
            
            # Load model
            loaded_model = quantizer.load_model(
                save_path,
                model_class=lambda: nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                ),
            )
            
            # Check that loaded model has same structure
            assert len(list(loaded_model.modules())) == len(list(quantized_model.modules()))
    
    def test_quantization_with_skip_modules(self):
        """Test that skip modules are not quantized."""
        # Create a custom model with an lm_head
        class ModelWithLMHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                )
                self.lm_head = nn.Linear(128, 1000)
            
            def forward(self, x):
                x = self.encoder(x)
                return self.lm_head(x)
        
        model = ModelWithLMHead()
        
        config = UnifiedQuantizationConfig(
            backend=QuantizationBackend.MPS_CUSTOM,
            bits=8,
            skip_modules=["lm_head"],
        )
        quantizer = UnifiedQuantizer(config)
        
        # Create a mock adapter without quantize_model method to force fallback
        mock_adapter = MagicMock(spec=[])  # Empty spec means no methods
        mock_adapter.backend = quantizer._adapter.backend
        mock_adapter.config = quantizer._adapter.config
        
        # Replace the adapter temporarily
        original_adapter = quantizer._adapter
        quantizer._adapter = mock_adapter
        
        try:
            quantized_model = quantizer.quantize_model(model)
            
            # Check that lm_head was not quantized
            assert not hasattr(quantized_model.lm_head, 'quantized_weight')
            
            # Check that encoder layers were quantized
            for name, module in quantized_model.named_modules():
                if isinstance(module, nn.Linear) and 'encoder' in name:
                    assert hasattr(module, 'quantized_weight'), f"Layer {name} should be quantized"
            
            # Also check that we have exactly 2 quantized layers (the encoder layers)
            quantized_count = sum(1 for _, m in quantized_model.named_modules() 
                                if isinstance(m, nn.Linear) and hasattr(m, 'quantized_weight'))
            assert quantized_count == 2  # Should have 2 linear layers in encoder
        finally:
            # Restore the adapter
            quantizer._adapter = original_adapter
    
    def test_layer_specific_bits(self):
        """Test layer-specific bit configuration."""
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        
        config = UnifiedQuantizationConfig(
            backend=QuantizationBackend.MPS_CUSTOM,
            bits=4,  # Default 4-bit
            layer_bits={"0": 8, "2": 16},  # First and last linear layers
        )
        quantizer = UnifiedQuantizer(config)
        
        # Create a mock adapter without quantize_model method to force fallback
        mock_adapter = MagicMock(spec=[])  # Empty spec means no methods
        mock_adapter.backend = quantizer._adapter.backend
        mock_adapter.config = quantizer._adapter.config
        
        # Replace the adapter temporarily
        original_adapter = quantizer._adapter
        quantizer._adapter = mock_adapter
        
        try:
            quantized_model = quantizer.quantize_model(model)
            
            # Check bit configuration - fallback stores bits as buffer
            assert quantized_model[0].quant_bits.item() == 8  # First linear
            assert quantized_model[2].quant_bits.item() == 16  # Last linear
        finally:
            # Restore the adapter
            quantizer._adapter = original_adapter


class TestQuantizeModelFunction:
    """Test the high-level quantize_model function."""
    
    @patch('src.utils.unified_quantization.BackendSelector.select_backend')
    def test_quantize_model_function(self, mock_select_backend):
        """Test the quantize_model convenience function."""
        mock_select_backend.return_value = QuantizationBackend.MPS_CUSTOM
        
        model = nn.Linear(128, 256)
        
        quantized_model, quantizer = quantize_model(
            model,
            bits=8,
            backend="auto",
            enable_lora=False,
        )
        
        assert quantizer.config.bits == 8
        assert quantizer.backend == QuantizationBackend.MPS_CUSTOM
        
    @patch('src.utils.unified_quantization.BackendSelector.select_backend')
    def test_quantize_model_from_string(self, mock_select_backend):
        """Test quantizing a model from HuggingFace ID."""
        mock_select_backend.return_value = QuantizationBackend.MPS_CUSTOM
        
        # Mock the transformers import
        mock_model = nn.Linear(128, 256)
        mock_automodel = MagicMock()
        mock_automodel.from_pretrained.return_value = mock_model
        
        with patch.dict('sys.modules', {'transformers': MagicMock(AutoModel=mock_automodel)}):
            quantized_model, quantizer = quantize_model(
                "meta-llama/Llama-2-7b-hf",
                bits=4,
            )
            
            mock_automodel.from_pretrained.assert_called_once_with("meta-llama/Llama-2-7b-hf")


class TestCompareBackends:
    """Test backend comparison functionality."""
    
    @patch('src.utils.unified_quantization.BackendSelector.detect_hardware')
    def test_compare_backends(self, mock_detect_hardware):
        """Test comparing different backends."""
        mock_detect_hardware.return_value = {
            "mlx_available": False,
            "quanto_available": False,
            "mps_available": True,
        }
        
        model = nn.Linear(128, 256)
        input_shape = (1, 128)
        
        results = compare_backends(model, input_shape, bits=8)
        
        # Should only test MPS_CUSTOM since others are not available
        assert QuantizationBackend.MPS_CUSTOM.value in results
        assert QuantizationBackend.MLX.value not in results
        assert QuantizationBackend.QUANTO.value not in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])