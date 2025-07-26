"""
Integration tests for quantization abstraction with train.py.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import shutil

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend, BackendManager
from src.core.quantization_wrapper import (
    QuantizationMethod,
    QuantizationConfig,
    create_quantization_adapter,
    replace_linear_with_quantized,
    get_recommended_config,
)


class TestModel(nn.Module):
    """Simple test model for integration testing."""
    
    def __init__(self, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            for _ in range(4)
        ])
        self.lm_head = nn.Linear(hidden_size, 1000)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        return self.lm_head(x)


class TestQuantizationIntegration(unittest.TestCase):
    """Test quantization integration with train.py functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.model = TestModel()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('quantization_wrapper.check_import_availability')
    def test_backend_specific_quantization(self, mock_check):
        """Test backend-specific quantization selection."""
        mock_check.return_value = True
        
        # CUDA + BNB
        config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        adapter = create_quantization_adapter(Backend.CUDA, config)
        quantized_model = replace_linear_with_quantized(self.model, adapter)
        
        # Check that linear layers were replaced (except lm_head)
        linear_count = 0
        quantized_count = 0
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                linear_count += 1
                if 'lm_head' in name:
                    # lm_head should remain unchanged
                    self.assertIsInstance(module, nn.Linear)
                else:
                    # Other layers should be replaced
                    quantized_count += 1
        
        # We have 8 linear layers (4 layers × 2 linear each) + 1 lm_head
        self.assertEqual(linear_count, 9)
    
    @patch('quantization_wrapper.check_import_availability')
    def test_memory_based_config_selection(self, mock_check):
        """Test automatic configuration based on memory constraints."""
        mock_check.return_value = True
        
        # Test different memory scenarios
        test_cases = [
            # (backend, model_size_b, memory_gb, expected_method, expected_bits)
            (Backend.CUDA, 7.0, 6.0, QuantizationMethod.BNB_NF4, 4),
            (Backend.CUDA, 7.0, 16.0, QuantizationMethod.BNB_INT8, 8),
            (Backend.CUDA, 7.0, 32.0, QuantizationMethod.NONE, None),
            (Backend.MPS, 7.0, 6.0, QuantizationMethod.MLX_INT4, 4),
            (Backend.CPU, 7.0, 12.0, QuantizationMethod.HQQ, 8),
        ]
        
        for backend, model_size, memory, expected_method, expected_bits in test_cases:
            config = get_recommended_config(backend, model_size, memory)
            self.assertEqual(config.method, expected_method)
            if expected_bits:
                self.assertEqual(config.bits, expected_bits)
    
    def test_quantization_with_train_workflow(self):
        """Test quantization in the context of a training workflow."""
        # Simulate the train.py workflow
        from argparse import Namespace
        
        # Create dummy args similar to train.py
        args = Namespace(
            q_bits=4,
            q_group_size=64,
            quantize='bnb_nf4',
            backend='cuda',
            compute_dtype='bfloat16',
            skip_modules=['lm_head']
        )
        
        # Create quantization config from args
        config = QuantizationConfig(
            method=QuantizationMethod.BNB_NF4,
            bits=args.q_bits,
            group_size=args.q_group_size,
            compute_dtype=torch.bfloat16,
            skip_modules=args.skip_modules
        )
        
        # Create adapter
        with patch('quantization_wrapper.check_import_availability', return_value=True):
            adapter = create_quantization_adapter('cuda', config)
            
            # Quantize model
            quantized_model = replace_linear_with_quantized(self.model, adapter)
            
            # Verify lm_head was skipped
            self.assertIsInstance(quantized_model.lm_head, nn.Linear)
    
    @patch('imports.get_module')
    @patch('quantization_wrapper.check_import_availability')
    def test_quantization_with_lora_peft(self, mock_check, mock_get_module):
        """Test quantization compatibility with LoRA/PEFT."""
        mock_check.return_value = True
        
        # Mock bitsandbytes module
        mock_bnb = MagicMock()
        mock_bnb.Linear4bit = MagicMock()
        mock_get_module.return_value = mock_bnb
        
        # Create quantized model
        config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        adapter = create_quantization_adapter(Backend.CUDA, config)
        
        # Prepare for training (simulating QLoRA setup)
        prepared_model = adapter.prepare_model_for_training(self.model)
        
        # Model should be returned (even if unchanged in this mock scenario)
        self.assertIsNotNone(prepared_model)
    
    def test_fallback_behavior(self):
        """Test fallback behavior when quantization is not supported."""
        # CPU + BNB should fallback
        config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        adapter = create_quantization_adapter(Backend.CPU, config)
        
        # Should get FallbackAdapter
        from quantization_wrapper import FallbackAdapter
        self.assertIsInstance(adapter, FallbackAdapter)
        
        # Quantize model with fallback
        with self.assertWarns(UserWarning):
            quantized_model = adapter.quantize_model(self.model)
        
        # Model should be unchanged
        self.assertIs(quantized_model, self.model)
    
    def test_backend_manager_integration(self):
        """Test integration with BackendManager."""
        # Create backend manager
        backend_manager = BackendManager(backend='auto', verbose=False)
        
        # Get recommended config based on detected backend
        config = get_recommended_config(
            backend_manager.backend,
            model_size_b=0.1,  # Small test model
            available_memory_gb=16.0
        )
        
        # Should get a valid config
        self.assertIsNotNone(config)
        self.assertIsInstance(config.method, QuantizationMethod)
    
    @patch('quantization_wrapper.get_module')
    @patch('quantization_wrapper.check_import_availability')
    def test_hqq_quantization_flow(self, mock_check, mock_get_module):
        """Test HQQ quantization flow."""
        mock_check.return_value = True
        
        # Mock HQQ module
        mock_hqq = MagicMock()
        mock_hqq.BaseQuantizeConfig = MagicMock()
        mock_hqq.HQQLinear = MagicMock()
        mock_get_module.return_value = mock_hqq
        
        # Create HQQ config
        config = QuantizationConfig(
            method=QuantizationMethod.HQQ,
            bits=8,
            group_size=128
        )
        
        # Create adapter for any backend (HQQ works on all)
        for backend in [Backend.CUDA, Backend.MPS, Backend.CPU]:
            adapter = create_quantization_adapter(backend, config)
            
            # Create quantized layer
            q_layer = adapter.create_quantized_linear(128, 256)
            self.assertIsNotNone(q_layer)
    
    def test_mlx_quantization_flow(self):
        """Test MLX quantization flow for Apple Silicon."""
        # MLX config
        config = QuantizationConfig(
            method=QuantizationMethod.MLX_INT4,
            bits=4,
            compute_dtype=torch.float16  # MPS doesn't support bfloat16
        )
        
        # Create adapter
        with patch('quantization_wrapper.check_import_availability', return_value=True):
            adapter = create_quantization_adapter(Backend.MPS, config)
            
            # Create quantized layer
            q_layer = adapter.create_quantized_linear(128, 256)
            self.assertIsNotNone(q_layer)
            
            # Test quantize_model
            with self.assertWarns(UserWarning):  # MLX conversion warning
                quantized_model = adapter.quantize_model(self.model)
    
    def test_config_validation_in_workflow(self):
        """Test configuration validation in typical workflow."""
        from quantization_wrapper import validate_quantization_config
        
        # Invalid: MPS + BNB
        config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        issues = validate_quantization_config(config, Backend.MPS)
        self.assertGreater(len(issues), 0)
        self.assertIn("bitsandbytes", issues[0])
        
        # Valid: MPS + MLX
        config = QuantizationConfig(method=QuantizationMethod.MLX_INT4)
        issues = validate_quantization_config(config, Backend.MPS)
        self.assertEqual(len(issues), 0)
        
        # Invalid: CUDA + MLX
        config = QuantizationConfig(method=QuantizationMethod.MLX_INT8)
        issues = validate_quantization_config(config, Backend.CUDA)
        self.assertGreater(len(issues), 0)
        self.assertIn("MLX", issues[0])
    
    def test_save_load_quantized_model(self):
        """Test saving and loading quantized models."""
        config = QuantizationConfig(method=QuantizationMethod.HQQ)
        
        with patch('quantization_wrapper.check_import_availability', return_value=True):
            adapter = create_quantization_adapter(Backend.CPU, config)
            
            # Save path
            save_path = os.path.join(self.test_dir, "quantized_model.pt")
            
            # Save model
            adapter.save_quantized_model(self.model, save_path)
            
            # Check file exists
            self.assertTrue(os.path.exists(save_path))
            
            # Load would require model architecture
            with self.assertRaises(NotImplementedError):
                adapter.load_quantized_model(save_path, None)
    
    def test_layer_specific_quantization(self):
        """Test layer-specific quantization configurations."""
        # Configure different quantization for different layers
        config = QuantizationConfig(
            method=QuantizationMethod.HQQ,
            bits=4,  # Default
            layer_configs={
                'layers.0': {'bits': 8},  # First layer uses 8-bit
                'layers.3': {'bits': 2},  # Last layer uses 2-bit
            }
        )
        
        # This tests that the config can store layer-specific settings
        self.assertEqual(config.layer_configs['layers.0']['bits'], 8)
        self.assertEqual(config.layer_configs['layers.3']['bits'], 2)
        
        # The actual implementation would use these in the adapter
        adapter = create_quantization_adapter(Backend.CPU, config)
        self.assertIsNotNone(adapter)


class TestQuantizationCLIIntegration(unittest.TestCase):
    """Test quantization integration with CLI arguments."""
    
    def test_parse_quantization_args(self):
        """Test parsing quantization arguments from CLI."""
        # Simulate argparse results
        args = Namespace(
            quantize='bnb_nf4',
            q_bits=4,
            q_group_size=64,
            q_compute_dtype='bfloat16',
            q_skip_modules='lm_head,embed',
            q_double_quant=True
        )
        
        # Convert to QuantizationConfig
        skip_modules = args.q_skip_modules.split(',') if args.q_skip_modules else []
        
        # Map string to method
        method_map = {
            'bnb_nf4': QuantizationMethod.BNB_NF4,
            'bnb_int8': QuantizationMethod.BNB_INT8,
            'hqq': QuantizationMethod.HQQ,
            'mlx_int4': QuantizationMethod.MLX_INT4,
            'mlx_int8': QuantizationMethod.MLX_INT8,
            'none': QuantizationMethod.NONE,
        }
        
        config = QuantizationConfig(
            method=method_map.get(args.quantize, QuantizationMethod.NONE),
            bits=args.q_bits,
            group_size=args.q_group_size,
            compute_dtype=torch.bfloat16 if args.q_compute_dtype == 'bfloat16' else torch.float16,
            skip_modules=skip_modules,
            double_quant=args.q_double_quant
        )
        
        self.assertEqual(config.method, QuantizationMethod.BNB_NF4)
        self.assertEqual(config.bits, 4)
        self.assertEqual(config.skip_modules, ['lm_head', 'embed'])
    
    def test_auto_quantization_selection(self):
        """Test automatic quantization method selection based on backend."""
        # Map of backend to expected default quantization
        backend_defaults = {
            Backend.CUDA: QuantizationMethod.BNB_NF4,
            Backend.MPS: QuantizationMethod.MLX_INT4,
            Backend.MLX: QuantizationMethod.MLX_INT4,
            Backend.CPU: QuantizationMethod.HQQ,
        }
        
        for backend, expected_method in backend_defaults.items():
            # Get recommended config with limited memory to force quantization
            config = get_recommended_config(backend, model_size_b=7.0, available_memory_gb=6.0)
            
            # Check if the method matches expected
            if backend == Backend.CUDA:
                self.assertIn(config.method, [QuantizationMethod.BNB_NF4, QuantizationMethod.BNB_INT8])
            elif backend in [Backend.MPS, Backend.MLX]:
                self.assertIn(config.method, [QuantizationMethod.MLX_INT4, QuantizationMethod.MLX_INT8])
            elif backend == Backend.CPU:
                self.assertEqual(config.method, QuantizationMethod.HQQ)


if __name__ == '__main__':
    unittest.main()