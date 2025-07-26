"""
Tests for the quantization abstraction layer.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend
from src.core.quantization_wrapper import (
    QuantizationMethod,
    QuantizationConfig,
    QuantizedLinear,
    QuantizationAdapter,
    BitsAndBytesAdapter,
    HQQAdapter,
    MLXAdapter,
    FallbackAdapter,
    create_quantization_adapter,
    validate_quantization_config,
    get_recommended_config,
    replace_linear_with_quantized,
    BNBLinear4bit,
    BNBLinear8bit,
    HQQLinearWrapper,
    MLXLinearWrapper,
)


class TestQuantizationConfig(unittest.TestCase):
    """Test QuantizationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = QuantizationConfig()
        
        self.assertEqual(config.method, QuantizationMethod.BNB_NF4)
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.group_size, 64)
        self.assertTrue(config.double_quant)
        self.assertEqual(config.quant_type, "nf4")
        self.assertEqual(config.skip_modules, ["lm_head"])
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid bit widths
        for bits in [2, 4, 8, 16]:
            config = QuantizationConfig(bits=bits)
            self.assertEqual(config.bits, bits)
        
        # Invalid bit width
        with self.assertRaises(ValueError):
            QuantizationConfig(bits=3)
        
        with self.assertRaises(ValueError):
            QuantizationConfig(bits=32)
    
    def test_compute_dtype_defaults(self):
        """Test compute dtype defaults."""
        with patch('torch.cuda.is_bf16_supported', return_value=True):
            config = QuantizationConfig()
            self.assertEqual(config.compute_dtype, torch.bfloat16)
        
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            config = QuantizationConfig()
            self.assertEqual(config.compute_dtype, torch.float16)
    
    def test_storage_dtype_defaults(self):
        """Test storage dtype defaults."""
        # 4-bit quantization
        config = QuantizationConfig(bits=4)
        self.assertEqual(config.storage_dtype, torch.uint8)
        
        # 16-bit quantization
        config = QuantizationConfig(bits=16)
        self.assertEqual(config.storage_dtype, torch.float16)


class TestQuantizedLinear(unittest.TestCase):
    """Test QuantizedLinear base class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Can't instantiate abstract class
        with self.assertRaises(TypeError):
            QuantizedLinear(128, 256)
    
    def test_initialization(self):
        """Test initialization of concrete subclass."""
        class ConcreteQuantizedLinear(QuantizedLinear):
            def forward(self, x):
                return x
            
            def quantize_weights(self, weights):
                return weights
            
            def dequantize_weights(self):
                return torch.randn(self.out_features, self.in_features)
        
        layer = ConcreteQuantizedLinear(128, 256, bias=True)
        self.assertEqual(layer.in_features, 128)
        self.assertEqual(layer.out_features, 256)
        self.assertIsNotNone(layer.config)


class TestBitsAndBytesAdapter(unittest.TestCase):
    """Test BitsAndBytesAdapter."""
    
    @patch('quantization_wrapper.check_import_availability')
    def test_backend_validation(self, mock_check):
        """Test that BNB adapter only works with CUDA."""
        mock_check.return_value = True
        
        # Should work with CUDA
        config = QuantizationConfig()
        adapter = BitsAndBytesAdapter(Backend.CUDA, config)
        self.assertEqual(adapter.backend, Backend.CUDA)
        
        # Should fail with other backends
        with self.assertRaises(ValueError):
            BitsAndBytesAdapter(Backend.MPS, config)
        
        with self.assertRaises(ValueError):
            BitsAndBytesAdapter(Backend.CPU, config)
    
    @patch('quantization_wrapper.check_import_availability')
    def test_import_availability_check(self, mock_check):
        """Test that adapter checks for bitsandbytes availability."""
        mock_check.return_value = False
        
        config = QuantizationConfig()
        with self.assertRaises(ImportError):
            BitsAndBytesAdapter(Backend.CUDA, config)
    
    @patch('quantization_wrapper.get_module')
    @patch('quantization_wrapper.check_import_availability')
    def test_create_quantized_linear(self, mock_check, mock_get_module):
        """Test creating quantized linear layers."""
        mock_check.return_value = True
        
        # Mock bitsandbytes module
        mock_bnb = MagicMock()
        mock_bnb.Linear4bit = MagicMock()
        mock_bnb.Linear8bitLt = MagicMock()
        mock_get_module.return_value = mock_bnb
        
        adapter = BitsAndBytesAdapter(Backend.CUDA, QuantizationConfig(bits=4))
        
        # Create 4-bit layer
        layer = adapter.create_quantized_linear(128, 256)
        self.assertIsInstance(layer, BNBLinear4bit)
        
        # Create 8-bit layer
        adapter.config.bits = 8
        layer = adapter.create_quantized_linear(128, 256)
        self.assertIsInstance(layer, BNBLinear8bit)
        
        # Unsupported bit width
        adapter.config.bits = 16
        with self.assertRaises(ValueError):
            adapter.create_quantized_linear(128, 256)


class TestHQQAdapter(unittest.TestCase):
    """Test HQQAdapter."""
    
    @patch('quantization_wrapper.check_import_availability')
    def test_backend_flexibility(self, mock_check):
        """Test that HQQ adapter works with any backend."""
        mock_check.return_value = True
        
        config = QuantizationConfig(method=QuantizationMethod.HQQ)
        
        # Should work with all backends
        for backend in [Backend.CUDA, Backend.MPS, Backend.CPU, Backend.MLX]:
            adapter = HQQAdapter(backend, config)
            self.assertEqual(adapter.backend, backend)
    
    @patch('quantization_wrapper.get_module')
    @patch('quantization_wrapper.check_import_availability')
    def test_create_quantized_linear(self, mock_check, mock_get_module):
        """Test creating HQQ quantized layers."""
        mock_check.return_value = True
        
        # Mock HQQ module
        mock_hqq = MagicMock()
        mock_hqq.BaseQuantizeConfig = MagicMock()
        mock_hqq.HQQLinear = MagicMock()
        mock_get_module.return_value = mock_hqq
        
        adapter = HQQAdapter(Backend.CUDA, QuantizationConfig(method=QuantizationMethod.HQQ))
        layer = adapter.create_quantized_linear(128, 256)
        self.assertIsInstance(layer, HQQLinearWrapper)
    
    @patch('quantization_wrapper.get_module')
    @patch('quantization_wrapper.check_import_availability')
    def test_fallback_when_unavailable(self, mock_check, mock_get_module):
        """Test fallback to standard linear when HQQ unavailable."""
        mock_check.return_value = False
        mock_get_module.side_effect = ImportError("HQQ not available")
        
        adapter = HQQAdapter(Backend.CUDA, QuantizationConfig(method=QuantizationMethod.HQQ))
        
        with self.assertWarns(UserWarning):
            layer = adapter.create_quantized_linear(128, 256)
            self.assertIsInstance(layer, nn.Linear)


class TestMLXAdapter(unittest.TestCase):
    """Test MLXAdapter."""
    
    @patch('quantization_wrapper.check_import_availability')
    def test_backend_validation(self, mock_check):
        """Test that MLX adapter only works with Apple Silicon backends."""
        mock_check.return_value = True
        
        config = QuantizationConfig(method=QuantizationMethod.MLX_INT4)
        
        # Should work with MLX and MPS
        adapter = MLXAdapter(Backend.MLX, config)
        self.assertEqual(adapter.backend, Backend.MLX)
        
        adapter = MLXAdapter(Backend.MPS, config)
        self.assertEqual(adapter.backend, Backend.MPS)
        
        # Should fail with other backends
        with self.assertRaises(ValueError):
            MLXAdapter(Backend.CUDA, config)
        
        with self.assertRaises(ValueError):
            MLXAdapter(Backend.CPU, config)
    
    @patch('quantization_wrapper.check_import_availability')
    def test_create_quantized_linear(self, mock_check):
        """Test creating MLX quantized layers."""
        mock_check.return_value = True
        
        adapter = MLXAdapter(Backend.MLX, QuantizationConfig(method=QuantizationMethod.MLX_INT4))
        layer = adapter.create_quantized_linear(128, 256)
        self.assertIsInstance(layer, MLXLinearWrapper)


class TestFallbackAdapter(unittest.TestCase):
    """Test FallbackAdapter."""
    
    def test_always_works(self):
        """Test that fallback adapter works with any backend."""
        config = QuantizationConfig()
        
        for backend in [Backend.CUDA, Backend.MPS, Backend.CPU, Backend.MLX]:
            with self.assertWarns(UserWarning):
                adapter = FallbackAdapter(backend, config)
                self.assertEqual(adapter.backend, backend)
    
    def test_no_quantization(self):
        """Test that fallback adapter doesn't quantize."""
        adapter = FallbackAdapter(Backend.CPU, QuantizationConfig())
        
        # Creates standard linear layer
        layer = adapter.create_quantized_linear(128, 256)
        self.assertIsInstance(layer, nn.Linear)
        
        # Returns model unchanged
        model = nn.Sequential(nn.Linear(128, 256))
        with self.assertWarns(UserWarning):
            quantized = adapter.quantize_model(model)
        self.assertIs(quantized, model)


class TestFactoryFunction(unittest.TestCase):
    """Test create_quantization_adapter factory function."""
    
    @patch('quantization_wrapper.check_import_availability')
    def test_cuda_bnb_selection(self, mock_check):
        """Test CUDA + BNB selection."""
        mock_check.return_value = True
        
        # BNB 4-bit
        adapter = create_quantization_adapter(
            Backend.CUDA,
            QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        )
        self.assertIsInstance(adapter, BitsAndBytesAdapter)
        
        # BNB 8-bit
        adapter = create_quantization_adapter(
            Backend.CUDA,
            QuantizationConfig(method=QuantizationMethod.BNB_INT8)
        )
        self.assertIsInstance(adapter, BitsAndBytesAdapter)
    
    @patch('quantization_wrapper.check_import_availability')
    def test_hqq_selection(self, mock_check):
        """Test HQQ selection."""
        mock_check.return_value = True
        
        adapter = create_quantization_adapter(
            Backend.CUDA,
            QuantizationConfig(method=QuantizationMethod.HQQ)
        )
        self.assertIsInstance(adapter, HQQAdapter)
    
    @patch('quantization_wrapper.check_import_availability')
    def test_mlx_selection(self, mock_check):
        """Test MLX selection."""
        mock_check.return_value = True
        
        adapter = create_quantization_adapter(
            Backend.MLX,
            QuantizationConfig(method=QuantizationMethod.MLX_INT4)
        )
        self.assertIsInstance(adapter, MLXAdapter)
    
    def test_fallback_selection(self):
        """Test fallback selection for unsupported combinations."""
        # CPU + BNB -> Fallback
        adapter = create_quantization_adapter(
            Backend.CPU,
            QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        )
        self.assertIsInstance(adapter, FallbackAdapter)
        
        # CUDA + MLX -> Fallback
        adapter = create_quantization_adapter(
            Backend.CUDA,
            QuantizationConfig(method=QuantizationMethod.MLX_INT4)
        )
        self.assertIsInstance(adapter, FallbackAdapter)
    
    def test_string_backend(self):
        """Test using string backend."""
        adapter = create_quantization_adapter(
            'cuda',
            QuantizationConfig(method=QuantizationMethod.HQQ)
        )
        self.assertIsInstance(adapter, HQQAdapter)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation functions."""
    
    def test_validate_quantization_config(self):
        """Test config validation for different backends."""
        # Valid: CUDA + BNB
        config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        issues = validate_quantization_config(config, Backend.CUDA)
        self.assertEqual(len(issues), 0)
        
        # Invalid: MPS + BNB
        issues = validate_quantization_config(config, Backend.MPS)
        self.assertGreater(len(issues), 0)
        self.assertIn("bitsandbytes", issues[0])
        
        # Warning: MPS + 4-bit (but not MLX method)
        config = QuantizationConfig(bits=4, method=QuantizationMethod.HQQ)
        issues = validate_quantization_config(config, Backend.MPS)
        self.assertGreater(len(issues), 0)
        self.assertIn("4-bit", issues[0])
        
        # Warning: MPS + bfloat16
        config = QuantizationConfig(method=QuantizationMethod.HQQ, compute_dtype=torch.bfloat16)
        issues = validate_quantization_config(config, Backend.MPS)
        self.assertGreater(len(issues), 0)
        self.assertIn("bfloat16", issues[0])
    
    def test_get_recommended_config(self):
        """Test recommended configuration generation."""
        # Small model, limited memory -> 4-bit
        config = get_recommended_config(Backend.CUDA, model_size_b=7.0, available_memory_gb=8.0)
        self.assertEqual(config.method, QuantizationMethod.BNB_NF4)
        self.assertEqual(config.bits, 4)
        
        # Medium memory -> 8-bit
        config = get_recommended_config(Backend.CUDA, model_size_b=7.0, available_memory_gb=16.0)
        self.assertEqual(config.method, QuantizationMethod.BNB_INT8)
        self.assertEqual(config.bits, 8)
        
        # Large memory -> No quantization
        config = get_recommended_config(Backend.CUDA, model_size_b=7.0, available_memory_gb=32.0)
        self.assertEqual(config.method, QuantizationMethod.NONE)
        
        # MPS backend
        config = get_recommended_config(Backend.MPS, model_size_b=7.0, available_memory_gb=8.0)
        self.assertEqual(config.method, QuantizationMethod.MLX_INT4)
        self.assertEqual(config.compute_dtype, torch.float16)  # Not bfloat16
        
        # CPU backend - needs less memory to trigger quantization
        config = get_recommended_config(Backend.CPU, model_size_b=7.0, available_memory_gb=12.0)
        self.assertEqual(config.method, QuantizationMethod.HQQ)


class TestModelQuantization(unittest.TestCase):
    """Test model quantization functions."""
    
    def setUp(self):
        """Create test model."""
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
        # Name the last layer 'lm_head' for skip testing
        self.model.add_module('lm_head', nn.Linear(10, 10))
    
    @patch('quantization_wrapper.check_import_availability')
    def test_replace_linear_with_quantized(self, mock_check):
        """Test replacing linear layers with quantized versions."""
        mock_check.return_value = True
        
        # Create adapter
        adapter = FallbackAdapter(Backend.CPU, QuantizationConfig())
        
        # Count linear layers before
        linear_count_before = sum(1 for m in self.model.modules() if isinstance(m, nn.Linear))
        self.assertEqual(linear_count_before, 4)  # 3 in Sequential + 1 lm_head
        
        # Replace layers
        quantized_model = replace_linear_with_quantized(self.model, adapter)
        
        # Should skip lm_head
        self.assertIsInstance(self.model.lm_head, nn.Linear)
        
        # Other layers should be replaced (in this case with nn.Linear from fallback)
        replaced_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name:
                replaced_count += 1
        
        self.assertEqual(replaced_count, 3)
    
    def test_weight_copying(self):
        """Test that weights are copied during replacement."""
        # Set some weights
        original_weight = self.model[0].weight.data.clone()
        
        # Create adapter
        adapter = FallbackAdapter(Backend.CPU, QuantizationConfig())
        
        # Replace layers
        quantized_model = replace_linear_with_quantized(self.model, adapter)
        
        # Check weights were copied
        new_weight = self.model[0].weight.data
        self.assertTrue(torch.allclose(original_weight, new_weight))


class TestQuantizedLayers(unittest.TestCase):
    """Test specific quantized layer implementations."""
    
    @patch('quantization_wrapper.get_module')
    def test_bnb_linear_4bit(self, mock_get_module):
        """Test BNBLinear4bit layer."""
        # Mock bitsandbytes
        mock_bnb = MagicMock()
        mock_linear4bit = MagicMock()
        mock_params4bit = MagicMock()
        
        mock_bnb.Linear4bit = MagicMock(return_value=mock_linear4bit)
        mock_bnb.Params4bit = MagicMock(return_value=mock_params4bit)
        
        config = QuantizationConfig(bits=4)
        layer = BNBLinear4bit(128, 256, True, config, mock_bnb)
        
        # Test initialization
        self.assertEqual(layer.in_features, 128)
        self.assertEqual(layer.out_features, 256)
        mock_bnb.Linear4bit.assert_called_once()
        
        # Test forward
        x = torch.randn(32, 128)
        layer.forward(x)
        mock_linear4bit.assert_called_with(x)
        
        # Test quantize_weights
        weights = torch.randn(256, 128)
        layer.quantize_weights(weights)
        mock_bnb.Params4bit.assert_called()
    
    @patch('quantization_wrapper.get_module')
    def test_hqq_linear_wrapper(self, mock_get_module):
        """Test HQQLinearWrapper."""
        # Mock HQQ
        mock_hqq = MagicMock()
        mock_hqq.BaseQuantizeConfig = MagicMock()
        mock_hqq.HQQLinear = MagicMock()
        
        config = QuantizationConfig(method=QuantizationMethod.HQQ)
        layer = HQQLinearWrapper(128, 256, True, config, mock_hqq)
        
        # Test initialization
        self.assertEqual(layer.in_features, 128)
        self.assertEqual(layer.out_features, 256)
        self.assertIsInstance(layer.linear_layer, nn.Linear)
        
        # Test forward (uses linear_layer before quantization)
        x = torch.randn(32, 128)
        output = layer.forward(x)
        self.assertEqual(output.shape, (32, 256))
    
    def test_mlx_linear_wrapper(self):
        """Test MLXLinearWrapper."""
        config = QuantizationConfig(method=QuantizationMethod.MLX_INT4)
        layer = MLXLinearWrapper(128, 256, True, config)
        
        # Test initialization
        self.assertEqual(layer.in_features, 128)
        self.assertEqual(layer.out_features, 256)
        self.assertFalse(layer.quantized)
        
        # Test forward (uses standard linear)
        x = torch.randn(32, 128)
        output = layer.forward(x)
        self.assertEqual(output.shape, (32, 256))
        
        # Test quantize_weights
        weights = torch.randn(256, 128)
        layer.quantize_weights(weights)
        self.assertTrue(layer.quantized)


if __name__ == '__main__':
    unittest.main()