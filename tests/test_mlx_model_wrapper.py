"""
Tests for the MLX model wrapper.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import shutil
import json
import numpy as np

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend, BackendManager
from src.core.quantization_wrapper import QuantizationConfig, QuantizationMethod

# Import with MLX mocking
with patch.dict('sys.modules', {
    'mlx': MagicMock(),
    'mlx.core': MagicMock(),
    'mlx.nn': MagicMock(),
    'mlx.optimizers': MagicMock(),
    'mlx.utils': MagicMock(),
}):
    from src.backends.mlx.mlx_model_wrapper import (
        MLXConfig,
        MLXLinear,
        LoRALinear,
        MLXModel,
        PyTorchToMLXConverter,
        MLXModelWrapper,
        UnifiedMemoryOptimizer,
        create_mlx_model,
        convert_pytorch_to_mlx,
    )


class TestMLXConfig(unittest.TestCase):
    """Test MLXConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MLXConfig(
            model_name="test-model",
            model_type="llama"
        )
        
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.model_type, "llama")
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.quantization_bits, 4)
        self.assertTrue(config.use_quantization)
        self.assertFalse(config.use_lora)
        self.assertEqual(config.lora_rank, 16)
        self.assertEqual(config.lora_target_modules, ["q_proj", "v_proj"])
    
    def test_from_huggingface_config(self):
        """Test creating config from HuggingFace format."""
        hf_config = {
            "_name_or_path": "meta-llama/Llama-2-7b-hf",
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 11008,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
        }
        
        config = MLXConfig.from_huggingface_config(hf_config)
        
        self.assertEqual(config.model_name, "meta-llama/Llama-2-7b-hf")
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.num_hidden_layers, 32)
    
    def test_lora_configuration(self):
        """Test LoRA-specific configuration."""
        config = MLXConfig(
            model_name="test",
            use_lora=True,
            lora_rank=8,
            lora_alpha=16.0,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        
        self.assertTrue(config.use_lora)
        self.assertEqual(config.lora_rank, 8)
        self.assertEqual(config.lora_alpha, 16.0)
        self.assertEqual(len(config.lora_target_modules), 4)


class TestMLXLinear(unittest.TestCase):
    """Test MLXLinear layer."""
    
    @patch('mlx_model_wrapper.mx')
    def test_standard_linear(self, mock_mx):
        """Test standard linear layer creation."""
        # Mock MLX arrays
        mock_mx.random.uniform.return_value = MagicMock()
        mock_mx.zeros.return_value = MagicMock()
        
        layer = MLXLinear(
            input_dims=128,
            output_dims=256,
            bias=True,
            quantized=False
        )
        
        self.assertFalse(layer.quantized)
        self.assertIsNotNone(layer.weight)
        self.assertIsNotNone(layer.bias)
        
        # Check initialization calls
        mock_mx.random.uniform.assert_called_once()
        mock_mx.zeros.assert_called_once_with((256,))
    
    @patch('mlx_model_wrapper.mx')
    def test_quantized_linear(self, mock_mx):
        """Test quantized linear layer creation."""
        mock_mx.zeros.return_value = MagicMock()
        mock_mx.ones_like.return_value = MagicMock()
        
        layer = MLXLinear(
            input_dims=128,
            output_dims=256,
            bias=True,
            quantized=True,
            bits=4,
            group_size=64
        )
        
        self.assertTrue(layer.quantized)
        self.assertEqual(layer.bits, 4)
        self.assertEqual(layer.group_size, 64)
        self.assertIsNotNone(layer.scales)
        self.assertIsNotNone(layer.q_weight)
    
    @patch('mlx_model_wrapper.mx')
    def test_forward_pass(self, mock_mx):
        """Test forward pass through linear layer."""
        # Create mock arrays with @ operator
        mock_weight = MagicMock()
        mock_weight.T = MagicMock()
        mock_input = MagicMock()
        mock_output = MagicMock()
        
        # Set up matrix multiplication
        mock_input.__matmul__.return_value = mock_output
        
        layer = MLXLinear(128, 256, bias=False, quantized=False)
        layer.weight = mock_weight
        
        result = layer(mock_input)
        
        # Check that matrix multiplication was performed
        mock_input.__matmul__.assert_called_once()


class TestLoRALinear(unittest.TestCase):
    """Test LoRALinear adapter."""
    
    @patch('mlx_model_wrapper.mx')
    @patch('mlx_model_wrapper.nn_mlx')
    def test_lora_initialization(self, mock_nn_mlx, mock_mx):
        """Test LoRA adapter initialization."""
        # Mock base layer
        base_layer = MagicMock()
        base_layer.weight = MagicMock(shape=(256, 128))
        base_layer.quantized = False
        
        # Mock MLX arrays
        mock_mx.random.normal.return_value = MagicMock()
        mock_mx.zeros.return_value = MagicMock()
        
        lora = LoRALinear(
            base_layer,
            rank=16,
            alpha=32.0,
            dropout=0.1
        )
        
        self.assertEqual(lora.rank, 16)
        self.assertEqual(lora.alpha, 32.0)
        self.assertEqual(lora.scaling, 32.0 / 16)  # alpha / rank
        self.assertIsNotNone(lora.lora_a)
        self.assertIsNotNone(lora.lora_b)
    
    @patch('mlx_model_wrapper.mx')
    def test_lora_forward(self, mock_mx):
        """Test LoRA forward pass."""
        # Mock base layer
        base_layer = MagicMock()
        base_output = MagicMock()
        base_layer.return_value = base_output
        base_layer.weight = MagicMock(shape=(256, 128))
        
        # Mock LoRA computation
        lora_a = MagicMock()
        lora_a.T = MagicMock()
        lora_b = MagicMock()
        lora_b.T = MagicMock()
        
        mock_input = MagicMock()
        lora_intermediate = MagicMock()
        lora_output = MagicMock()
        
        # Set up chained matrix multiplication
        mock_input.__matmul__.return_value = lora_intermediate
        lora_intermediate.__matmul__.return_value = lora_output
        base_output.__add__.return_value = MagicMock()
        
        lora = LoRALinear(base_layer, rank=16)
        lora.lora_a = lora_a
        lora.lora_b = lora_b
        lora.lora_dropout = None
        
        result = lora(mock_input)
        
        # Check that base layer was called
        base_layer.assert_called_once_with(mock_input)
        
        # Check that LoRA computation was performed
        mock_input.__matmul__.assert_called_once()


class TestPyTorchToMLXConverter(unittest.TestCase):
    """Test PyTorch to MLX conversion."""
    
    @patch('mlx_model_wrapper.mx')
    def test_convert_tensor(self, mock_mx):
        """Test tensor conversion."""
        # Create PyTorch tensor
        torch_tensor = torch.randn(10, 20)
        
        # Mock MLX array creation
        mock_array = MagicMock()
        mock_mx.array.return_value = mock_array
        
        # Convert
        mlx_array = PyTorchToMLXConverter.convert_tensor(torch_tensor)
        
        # Check conversion
        mock_mx.array.assert_called_once()
        call_args = mock_mx.array.call_args[0][0]
        self.assertEqual(call_args.shape, (10, 20))
        self.assertEqual(mlx_array, mock_array)
    
    @patch('mlx_model_wrapper.mx')
    @patch('mlx_model_wrapper.nn_mlx')
    def test_convert_linear_layer(self, mock_nn_mlx, mock_mx):
        """Test linear layer conversion."""
        # Create PyTorch linear layer
        pytorch_layer = nn.Linear(128, 256, bias=True)
        
        # Mock MLX linear creation
        mock_mlx_linear = MagicMock()
        mock_nn_mlx.Linear.return_value = mock_mlx_linear
        
        # Convert without quantization
        mlx_layer = PyTorchToMLXConverter.convert_linear_layer(
            pytorch_layer,
            quantize=False
        )
        
        # Check creation
        mock_nn_mlx.Linear.assert_called_once_with(128, 256, bias=True)
        
        # Check weight conversion
        self.assertEqual(mlx_layer, mock_mlx_linear)
    
    @patch('mlx_model_wrapper.MLXLinear')
    @patch('mlx_model_wrapper.mx')
    def test_convert_linear_layer_quantized(self, mock_mx, mock_mlx_linear_cls):
        """Test quantized linear layer conversion."""
        # Create PyTorch linear layer
        pytorch_layer = nn.Linear(128, 256, bias=False)
        
        # Mock quantized MLX linear
        mock_mlx_layer = MagicMock()
        mock_mlx_linear_cls.return_value = mock_mlx_layer
        
        # Convert with quantization
        mlx_layer = PyTorchToMLXConverter.convert_linear_layer(
            pytorch_layer,
            quantize=True,
            bits=4,
            group_size=64
        )
        
        # Check creation
        mock_mlx_linear_cls.assert_called_once_with(
            128, 256, bias=False, quantized=True, bits=4, group_size=64
        )
        
        # Check quantization was called
        mock_mlx_layer.quantize_weights.assert_called_once()
    
    @patch('mlx_model_wrapper.mx')
    @patch('mlx_model_wrapper.nn_mlx')
    def test_convert_embedding(self, mock_nn_mlx, mock_mx):
        """Test embedding layer conversion."""
        # Create PyTorch embedding
        pytorch_emb = nn.Embedding(1000, 128)
        
        # Mock MLX embedding
        mock_mlx_emb = MagicMock()
        mock_nn_mlx.Embedding.return_value = mock_mlx_emb
        
        # Convert
        mlx_emb = PyTorchToMLXConverter.convert_embedding(pytorch_emb)
        
        # Check creation
        mock_nn_mlx.Embedding.assert_called_once_with(1000, 128)
        self.assertEqual(mlx_emb, mock_mlx_emb)


class TestMLXModelWrapper(unittest.TestCase):
    """Test MLXModelWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_mlx_model = MagicMock(spec=MLXModel)
        self.mock_tokenizer = MagicMock()
        self.mock_backend_manager = MagicMock(spec=BackendManager)
        self.mock_backend_manager.get_device.return_value = torch.device("cpu")
        
        self.wrapper = MLXModelWrapper(
            self.mock_mlx_model,
            self.mock_tokenizer,
            self.mock_backend_manager
        )
    
    @patch('mlx_model_wrapper.PyTorchToMLXConverter')
    def test_forward_pass(self, mock_converter):
        """Test forward pass through wrapper."""
        # Mock inputs
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        
        # Mock conversion
        mock_mlx_input = MagicMock()
        mock_converter.convert_tensor.return_value = mock_mlx_input
        
        # Mock model output
        mock_mlx_output = MagicMock()
        self.mock_mlx_model.return_value = mock_mlx_output
        
        # Mock conversion back
        mock_logits = torch.randn(2, 10, 1000)
        self.wrapper._mlx_to_torch = MagicMock(return_value=mock_logits)
        
        # Forward pass
        outputs = self.wrapper.forward(input_ids=input_ids, labels=labels)
        
        # Check model was called
        self.mock_mlx_model.assert_called_once()
        
        # Check outputs
        self.assertIn("logits", outputs)
        self.assertIn("loss", outputs)
        self.assertIsNotNone(outputs["loss"])
    
    def test_train_eval_mode(self):
        """Test train/eval mode setting."""
        # MLX doesn't have train/eval modes, so these should be no-ops
        self.wrapper.train()
        self.wrapper.eval()
        # No assertions needed - just checking they don't error
    
    @patch('mlx_model_wrapper.mx')
    def test_save_pretrained(self, mock_mx):
        """Test saving model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock model parameters
            mock_params = {"weight": MagicMock()}
            with patch('mlx_model_wrapper.tree_flatten', return_value=mock_params):
                self.wrapper.save_pretrained(tmpdir)
            
            # Check save was called
            mock_mx.save.assert_called_once()
            
            # Check config was saved
            config_path = os.path.join(tmpdir, "config.json")
            self.assertTrue(os.path.exists(config_path))
            
            # Check tokenizer save was called
            self.mock_tokenizer.save_pretrained.assert_called_once_with(tmpdir)
    
    def test_parameters_collection(self):
        """Test parameter collection for LoRA."""
        # Create mock LoRA layers
        mock_lora = MagicMock(spec=LoRALinear)
        mock_lora.lora_a = MagicMock()
        mock_lora.lora_b = MagicMock()
        
        # Mock model structure with LoRA
        self.mock_mlx_model.layer1 = mock_lora
        
        # Mock tensor conversion
        mock_tensor_a = torch.randn(16, 128)
        mock_tensor_b = torch.randn(256, 16)
        
        def mock_mlx_to_torch(array):
            if array == mock_lora.lora_a:
                return mock_tensor_a
            elif array == mock_lora.lora_b:
                return mock_tensor_b
            return torch.randn(10, 10)
        
        self.wrapper._mlx_to_torch = mock_mlx_to_torch
        
        # Collect parameters
        params = self.wrapper.parameters()
        
        # Should have LoRA parameters
        self.assertGreater(len(params), 0)


class TestUnifiedMemoryOptimizer(unittest.TestCase):
    """Test UnifiedMemoryOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_wrapper = MagicMock(spec=MLXModelWrapper)
        self.mock_wrapper._input_cache = {"test": MagicMock()}
        self.mock_wrapper._output_cache = {"test": MagicMock()}
        
        self.optimizer = UnifiedMemoryOptimizer(self.mock_wrapper)
    
    def test_optimize_memory_layout(self):
        """Test memory optimization."""
        with patch('gc.collect') as mock_gc:
            self.optimizer.optimize_memory_layout()
            
            # Check caches were cleared
            self.assertEqual(len(self.mock_wrapper._input_cache), 0)
            self.assertEqual(len(self.mock_wrapper._output_cache), 0)
            
            # Check garbage collection was called
            mock_gc.assert_called_once()
    
    def test_enable_gradient_checkpointing(self):
        """Test gradient checkpointing enablement."""
        # Should warn that it's not implemented
        with self.assertWarns(UserWarning):
            self.optimizer.enable_gradient_checkpointing()
    
    @patch('mlx_model_wrapper.psutil')
    def test_profile_memory_usage(self, mock_psutil):
        """Test memory profiling."""
        # Mock process and memory info
        mock_process = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 8 * 1e9  # 8GB
        mock_memory_info.vms = 16 * 1e9  # 16GB
        
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 50.0
        
        mock_psutil.Process.return_value = mock_process
        mock_psutil.virtual_memory.return_value.available = 32 * 1e9  # 32GB
        
        # Profile memory
        stats = self.optimizer.profile_memory_usage()
        
        # Check stats
        self.assertEqual(stats["rss_gb"], 8.0)
        self.assertEqual(stats["vms_gb"], 16.0)
        self.assertEqual(stats["available_gb"], 32.0)
        self.assertEqual(stats["percent"], 50.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    @patch('mlx_model_wrapper.MLX_AVAILABLE', False)
    def test_create_mlx_model_no_mlx(self):
        """Test model creation when MLX is not available."""
        with self.assertRaises(ImportError):
            create_mlx_model("test-model")
    
    @patch('mlx_model_wrapper.MLX_AVAILABLE', True)
    def test_create_mlx_model_not_implemented(self):
        """Test model creation raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            create_mlx_model("test-model")
    
    @patch('mlx_model_wrapper.MLX_AVAILABLE', False)
    def test_convert_pytorch_to_mlx_no_mlx(self):
        """Test conversion when MLX is not available."""
        mock_model = MagicMock()
        config = MLXConfig("test")
        
        with self.assertRaises(ImportError):
            convert_pytorch_to_mlx(mock_model, config)
    
    @patch('mlx_model_wrapper.MLX_AVAILABLE', True)
    def test_convert_pytorch_to_mlx_not_implemented(self):
        """Test conversion raises NotImplementedError."""
        mock_model = MagicMock()
        config = MLXConfig("test")
        
        with self.assertRaises(NotImplementedError):
            convert_pytorch_to_mlx(mock_model, config)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios."""
    
    @patch('mlx_model_wrapper.mx')
    @patch('mlx_model_wrapper.nn_mlx')
    def test_quantized_lora_setup(self, mock_nn_mlx, mock_mx):
        """Test setting up quantized model with LoRA."""
        # Create config
        config = MLXConfig(
            model_name="test-model",
            use_quantization=True,
            quantization_bits=4,
            use_lora=True,
            lora_rank=8,
            lora_target_modules=["q_proj", "v_proj"]
        )
        
        # Create mock model with linear layers
        mock_model = MagicMock(spec=MLXModel)
        mock_model.config = config
        
        # Mock linear layer
        mock_linear = MagicMock()
        mock_linear.weight = MagicMock(shape=(256, 128))
        mock_model.q_proj = mock_linear
        mock_model.v_proj = mock_linear
        
        # Apply LoRA (simplified test)
        def mock_apply(fn):
            for name in ["q_proj", "v_proj"]:
                if hasattr(mock_model, name):
                    new_layer = fn(getattr(mock_model, name), name)
                    if new_layer is not None:
                        setattr(mock_model, name, new_layer)
        
        mock_model._apply_to_modules = mock_apply
        
        # Apply LoRA
        mock_model.apply_lora()
        
        # Both layers should be wrapped
        # In real implementation, they would be LoRALinear instances


if __name__ == '__main__':
    unittest.main()