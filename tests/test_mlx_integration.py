"""
Integration tests for MLX model wrapper with the training pipeline.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from argparse import Namespace
import tempfile
import shutil

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend, BackendManager
from src.core.quantization_wrapper import QuantizationConfig, QuantizationMethod
from src.core.model_loader import ModelLoadingConfig

# Mock MLX imports
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
        MLXModelWrapper,
        UnifiedMemoryOptimizer,
        PyTorchToMLXConverter,
    )


class TestMLXTrainingIntegration(unittest.TestCase):
    """Test MLX integration with training workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_train_args(self, **overrides):
        """Create training arguments for MLX."""
        args = {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "train_type": "qlora",
            "backend": "mlx",
            "precision": "fp16",  # MLX doesn't support bf16
            "batch_size": 2,
            "context_length": 512,
            "gradient_accumulation_steps": 4,
            "num_epochs": 1,
            "dataset": "alpaca_sample",
            "use_gradient_checkpointing": False,  # Not yet supported in MLX
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "v_proj"],
            "q_bits": 4,
            "q_group_size": 64,
            "learning_rate": 5e-4,
            "verbose": True,
        }
        args.update(overrides)
        return Namespace(**args)
    
    @patch('mlx_model_wrapper.mx')
    def test_mlx_backend_detection(self, mock_mx):
        """Test MLX backend detection and setup."""
        backend_manager = BackendManager(backend="mlx", verbose=True)
        
        # Should detect MLX backend
        self.assertEqual(backend_manager.backend, Backend.MLX)
        
        # Device should be CPU (MLX uses unified memory)
        device = backend_manager.get_device()
        self.assertEqual(device.type, "cpu")
    
    def test_mlx_config_from_train_args(self):
        """Test creating MLX config from training arguments."""
        args = self.create_train_args()
        
        # Create MLX config
        mlx_config = MLXConfig(
            model_name=args.model_name,
            use_quantization=True,
            quantization_bits=args.q_bits,
            quantization_group_size=args.q_group_size,
            use_lora=True,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
        )
        
        self.assertEqual(mlx_config.model_name, args.model_name)
        self.assertTrue(mlx_config.use_quantization)
        self.assertEqual(mlx_config.quantization_bits, 4)
        self.assertTrue(mlx_config.use_lora)
        self.assertEqual(mlx_config.lora_rank, 16)
    
    def test_quantization_config_for_mlx(self):
        """Test quantization configuration for MLX backend."""
        from quantization_wrapper import get_recommended_config
        
        # Get recommended config for MLX with limited memory
        config = get_recommended_config(
            Backend.MLX,
            model_size_b=7.0,
            available_memory_gb=8.0
        )
        
        # Should recommend MLX quantization
        self.assertIn(config.method, [QuantizationMethod.MLX_INT4, QuantizationMethod.MLX_INT8])
        self.assertEqual(config.compute_dtype, torch.float16)  # MLX doesn't support bf16
    
    @patch('mlx_model_wrapper.MLXModel')
    @patch('mlx_model_wrapper.mx')
    def test_mlx_model_wrapper_training_loop(self, mock_mx, mock_mlx_model_cls):
        """Test MLX model wrapper in training loop."""
        # Create mock MLX model
        mock_mlx_model = MagicMock()
        mock_mlx_model_cls.return_value = mock_mlx_model
        
        # Create wrapper
        wrapper = MLXModelWrapper(
            mock_mlx_model,
            tokenizer=MagicMock(),
            backend_manager=BackendManager(backend="mlx")
        )
        
        # Simulate training batch
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128),
            "labels": torch.randint(0, 1000, (2, 128)),
        }
        
        # Mock MLX output
        mock_mlx_output = MagicMock()
        mock_mlx_model.return_value = mock_mlx_output
        
        # Mock tensor conversion
        with patch.object(wrapper, '_torch_to_mlx') as mock_to_mlx:
            with patch.object(wrapper, '_mlx_to_torch') as mock_to_torch:
                mock_to_torch.return_value = torch.randn(2, 128, 32000)
                
                # Forward pass
                outputs = wrapper.forward(**batch)
                
                # Check conversions were called
                mock_to_mlx.assert_called()
                mock_to_torch.assert_called()
                
                # Check outputs
                self.assertIn("loss", outputs)
                self.assertIn("logits", outputs)
    
    def test_unified_memory_optimization(self):
        """Test unified memory optimization for Apple Silicon."""
        # Create wrapper with mock model
        mock_model = MagicMock()
        wrapper = MLXModelWrapper(mock_model, MagicMock(), BackendManager(backend="mlx"))
        
        # Create optimizer
        optimizer = UnifiedMemoryOptimizer(wrapper)
        
        # Add some dummy cache entries
        wrapper._input_cache = {1: "data1", 2: "data2"}
        wrapper._output_cache = {3: "data3", 4: "data4"}
        
        # Optimize memory
        with patch('gc.collect') as mock_gc:
            optimizer.optimize_memory_layout()
            
            # Caches should be cleared
            self.assertEqual(len(wrapper._input_cache), 0)
            self.assertEqual(len(wrapper._output_cache), 0)
            
            # GC should be called
            mock_gc.assert_called_once()
    
    @patch('mlx_model_wrapper.tree_flatten')
    @patch('mlx_model_wrapper.mx')
    def test_save_and_load_mlx_model(self, mock_mx, mock_tree_flatten):
        """Test saving and loading MLX models."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.config = MLXConfig("test-model")
        
        wrapper = MLXModelWrapper(mock_model, MagicMock())
        
        # Mock parameters
        mock_params = {"layer.weight": MagicMock()}
        mock_tree_flatten.return_value = mock_params
        
        # Save model
        save_path = os.path.join(self.test_dir, "mlx_model")
        wrapper.save_pretrained(save_path)
        
        # Check save was called
        mock_mx.save.assert_called_once()
        
        # Check config was saved
        config_path = os.path.join(save_path, "config.json")
        self.assertTrue(os.path.exists(config_path))
    
    def test_pytorch_to_mlx_conversion_flow(self):
        """Test PyTorch to MLX model conversion flow."""
        # Create PyTorch model components
        pytorch_linear = nn.Linear(128, 256)
        pytorch_embedding = nn.Embedding(1000, 128)
        
        # Test tensor conversion
        torch_tensor = torch.randn(10, 20)
        with patch('mlx_model_wrapper.mx.array') as mock_array:
            mlx_array = PyTorchToMLXConverter.convert_tensor(torch_tensor)
            mock_array.assert_called_once()
        
        # Test linear layer conversion
        with patch('mlx_model_wrapper.nn_mlx.Linear') as mock_mlx_linear:
            with patch('mlx_model_wrapper.mx'):
                mlx_linear = PyTorchToMLXConverter.convert_linear_layer(
                    pytorch_linear,
                    quantize=False
                )
                mock_mlx_linear.assert_called_once_with(128, 256, bias=True)
        
        # Test quantized linear conversion
        with patch('mlx_model_wrapper.MLXLinear') as mock_mlx_quantized:
            with patch('mlx_model_wrapper.mx'):
                mlx_quantized = PyTorchToMLXConverter.convert_linear_layer(
                    pytorch_linear,
                    quantize=True,
                    bits=4
                )
                mock_mlx_quantized.assert_called_once()
    
    def test_lora_parameter_updates(self):
        """Test LoRA parameter updates during training."""
        # Create mock base layer
        mock_base = MagicMock()
        mock_base.weight = MagicMock(shape=(256, 128))
        
        # Create LoRA layer
        with patch('mlx_model_wrapper.mx.random.normal') as mock_normal:
            with patch('mlx_model_wrapper.mx.zeros') as mock_zeros:
                mock_normal.return_value = MagicMock()
                mock_zeros.return_value = MagicMock()
                
                lora = LoRALinear(
                    mock_base,
                    rank=8,
                    alpha=16.0,
                    dropout=0.0
                )
                
                # Check LoRA matrices were initialized
                self.assertIsNotNone(lora.lora_a)
                self.assertIsNotNone(lora.lora_b)
                self.assertEqual(lora.scaling, 16.0 / 8)
    
    def test_mlx_training_memory_profile(self):
        """Test memory profiling during MLX training."""
        wrapper = MLXModelWrapper(MagicMock(), MagicMock())
        optimizer = UnifiedMemoryOptimizer(wrapper)
        
        with patch('mlx_model_wrapper.psutil') as mock_psutil:
            # Mock memory info
            mock_process = MagicMock()
            mock_process.memory_info.return_value = MagicMock(
                rss=16 * 1e9,  # 16GB
                vms=24 * 1e9   # 24GB
            )
            mock_process.memory_percent.return_value = 40.0
            
            mock_psutil.Process.return_value = mock_process
            mock_psutil.virtual_memory.return_value = MagicMock(available=64 * 1e9)
            
            # Profile memory
            stats = optimizer.profile_memory_usage()
            
            # Check stats are reasonable
            self.assertEqual(stats["rss_gb"], 16.0)
            self.assertEqual(stats["vms_gb"], 24.0)
            self.assertEqual(stats["available_gb"], 64.0)
            self.assertLess(stats["percent"], 100.0)


class TestMLXQuantizationIntegration(unittest.TestCase):
    """Test MLX quantization integration."""
    
    @patch('mlx_model_wrapper.mx')
    def test_mlx_4bit_quantization(self, mock_mx):
        """Test 4-bit quantization in MLX."""
        # Mock MLX arrays
        mock_mx.zeros.return_value = MagicMock()
        mock_mx.ones_like.return_value = MagicMock()
        
        # Create quantized layer
        layer = MLXLinear(
            input_dims=256,
            output_dims=512,
            quantized=True,
            bits=4,
            group_size=64
        )
        
        self.assertTrue(layer.quantized)
        self.assertEqual(layer.bits, 4)
        
        # Check quantization structures
        self.assertIsNotNone(layer.scales)
        self.assertIsNotNone(layer.biases)
        self.assertIsNotNone(layer.q_weight)
    
    @patch('mlx_model_wrapper.mx')
    def test_mlx_8bit_quantization(self, mock_mx):
        """Test 8-bit quantization in MLX."""
        mock_mx.zeros.return_value = MagicMock()
        mock_mx.ones_like.return_value = MagicMock()
        
        layer = MLXLinear(
            input_dims=256,
            output_dims=512,
            quantized=True,
            bits=8,
            group_size=128
        )
        
        self.assertEqual(layer.bits, 8)
        self.assertEqual(layer.group_size, 128)
    
    def test_quantized_lora_combination(self):
        """Test combining quantization with LoRA."""
        # Create quantized base layer
        with patch('mlx_model_wrapper.mx'):
            base_layer = MLXLinear(
                128, 256,
                quantized=True,
                bits=4
            )
            base_layer.q_weight = MagicMock(shape=(256, 64))  # Compressed weight
            
            # Wrap with LoRA
            with patch('mlx_model_wrapper.mx.random.normal'):
                with patch('mlx_model_wrapper.mx.zeros'):
                    lora_layer = LoRALinear(
                        base_layer,
                        rank=8,
                        alpha=16.0
                    )
                    
                    # LoRA should work with quantized base
                    self.assertEqual(lora_layer.rank, 8)
                    self.assertIsNotNone(lora_layer.lora_a)
                    self.assertIsNotNone(lora_layer.lora_b)


class TestMLXModelLoaderIntegration(unittest.TestCase):
    """Test MLX integration with model loader."""
    
    def test_mlx_model_loader_config(self):
        """Test MLXModelLoader configuration."""
        from model_loader import ModelLoadingConfig, LoadingStrategy
        
        config = ModelLoadingConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            backend=Backend.MLX,
            quantization_config=QuantizationConfig(
                method=QuantizationMethod.MLX_INT4,
                bits=4,
                compute_dtype=torch.float16
            ),
            dtype=torch.float16,
            loading_strategy=LoadingStrategy.UNIFIED_MEMORY
        )
        
        # MLX should use unified memory
        self.assertEqual(config.loading_strategy, LoadingStrategy.UNIFIED_MEMORY)
        # Device should be CPU for MLX
        self.assertEqual(config.device.type, "cpu")
    
    @patch('model_loader.check_import_availability')
    def test_mlx_model_loader_validation(self, mock_check):
        """Test MLXModelLoader validation."""
        from model_loader import MLXModelLoader, ModelLoadingConfig
        
        # MLX not available
        mock_check.return_value = False
        config = ModelLoadingConfig("test", Backend.MLX)
        
        with self.assertRaises(ImportError):
            MLXModelLoader(config)
        
        # MLX available
        mock_check.return_value = True
        with self.assertWarns(UserWarning):  # Warns about format conversion
            loader = MLXModelLoader(config)
            self.assertIsNotNone(loader)


class TestMLXPerformanceOptimizations(unittest.TestCase):
    """Test MLX-specific performance optimizations."""
    
    def test_batch_size_recommendations(self):
        """Test batch size recommendations for MLX."""
        backend_manager = BackendManager(backend="mlx")
        
        # Get recommended batch sizes
        recommendations = backend_manager.get_batch_size_recommendation(
            model_size_b=7.0,
            available_memory_gb=16.0
        )
        
        # MLX can handle larger batches due to unified memory
        self.assertIn("batch_size", recommendations)
        self.assertGreater(recommendations["batch_size"], 0)
    
    def test_mlx_parallelization(self):
        """Test MLX parallelization settings."""
        from model_loader import ModelLoadingConfig
        
        config = ModelLoadingConfig(
            "test-model",
            Backend.MLX,
            loading_workers=-1  # Auto
        )
        
        # MLX should use multiple workers for loading
        from model_loader import MLXModelLoader
        with patch('model_loader.check_import_availability', return_value=True):
            with self.assertWarns(UserWarning):
                loader = MLXModelLoader(config)
                workers = loader._get_loading_workers(7e9)  # 7B params
                self.assertEqual(workers, 4)  # MLX uses 4 workers


if __name__ == '__main__':
    unittest.main()