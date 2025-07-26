"""
Tests for the model loading abstraction layer.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
import tempfile
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend
from src.core.model_loader import (
    LoadingStrategy,
    ModelLoadingConfig,
    ModelLoader,
    StandardModelLoader,
    QuantizedModelLoader,
    CUDAModelLoader,
    MPSModelLoader,
    MLXModelLoader,
    CPUModelLoader,
    ModelLoaderFactory,
    load_model_and_tokenizer,
    get_recommended_loader_config,
)
from src.core.quantization_wrapper import QuantizationConfig, QuantizationMethod


class TestModelLoadingConfig(unittest.TestCase):
    """Test ModelLoadingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelLoadingConfig(
            model_name="test-model",
            backend=Backend.CUDA
        )
        
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.backend, Backend.CUDA)
        self.assertEqual(config.loading_strategy, LoadingStrategy.FULL)
        self.assertIsNone(config.quantization_config)
        self.assertEqual(config.dtype, torch.float16)
        self.assertFalse(config.low_memory)
        self.assertEqual(config.skip_modules, ["lm_head"])
    
    def test_auto_device_selection(self):
        """Test automatic device selection based on backend."""
        # CUDA
        with patch('torch.cuda.is_available', return_value=True):
            config = ModelLoadingConfig("test", Backend.CUDA)
            self.assertEqual(config.device.type, "cuda")
        
        # MPS
        with patch('torch.backends.mps.is_available', return_value=True):
            config = ModelLoadingConfig("test", Backend.MPS)
            self.assertEqual(config.device.type, "mps")
        
        # CPU
        config = ModelLoadingConfig("test", Backend.CPU)
        self.assertEqual(config.device.type, "cpu")
    
    def test_unified_memory_auto_selection(self):
        """Test automatic unified memory selection for Apple Silicon."""
        config = ModelLoadingConfig("test", Backend.MPS)
        self.assertEqual(config.loading_strategy, LoadingStrategy.UNIFIED_MEMORY)
        
        config = ModelLoadingConfig("test", Backend.MLX)
        self.assertEqual(config.loading_strategy, LoadingStrategy.UNIFIED_MEMORY)
    
    def test_quant_method_from_config(self):
        """Test quantization method detection from config."""
        # BNB
        quant_config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        config = ModelLoadingConfig("test", Backend.CUDA, quantization_config=quant_config)
        self.assertEqual(config.quant_method, "bnb")
        
        # HQQ
        quant_config = QuantizationConfig(method=QuantizationMethod.HQQ)
        config = ModelLoadingConfig("test", Backend.CPU, quantization_config=quant_config)
        self.assertEqual(config.quant_method, "hqq")


class TestModelLoader(unittest.TestCase):
    """Test base ModelLoader functionality."""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        config = ModelLoadingConfig("test", Backend.CPU)
        
        # Can't instantiate abstract class
        with self.assertRaises(TypeError):
            ModelLoader(config)
    
    @patch('model_loader.hub')
    def test_get_model_files(self, mock_hub):
        """Test getting model files from hub."""
        config = ModelLoadingConfig("test-model", Backend.CPU)
        
        # Test sharded model
        mock_hub.cached_file.return_value = "index.json"
        mock_hub.get_checkpoint_shard_files.return_value = (
            ["shard1.safetensors", "shard2.safetensors"],
            {}
        )
        
        loader = StandardModelLoader(config)
        files = loader._get_model_files()
        
        self.assertEqual(len(files), 2)
        self.assertIn("shard1.safetensors", files)
    
    @patch('model_loader.hub')
    def test_get_model_files_single(self, mock_hub):
        """Test getting single model file."""
        config = ModelLoadingConfig("test-model", Backend.CPU)
        
        # First call fails (no index), second succeeds (single file)
        mock_hub.cached_file.side_effect = [
            OSError("No index"),
            "model.safetensors"
        ]
        
        loader = StandardModelLoader(config)
        files = loader._get_model_files()
        
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], "model.safetensors")
    
    def test_get_loading_workers(self):
        """Test worker calculation for parallel loading."""
        config = ModelLoadingConfig("test", Backend.CUDA)
        loader = StandardModelLoader(config)
        
        # Manual setting
        config.loading_workers = 4
        self.assertEqual(loader._get_loading_workers(1e9), 4)
        
        # Auto for CUDA
        config.loading_workers = -1
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                # Large GPU
                mock_props.return_value.total_memory = 80e9
                self.assertEqual(loader._get_loading_workers(1e9), 8)
                
                # Medium GPU
                mock_props.return_value.total_memory = 40e9
                self.assertEqual(loader._get_loading_workers(1e9), 4)


class TestStandardModelLoader(unittest.TestCase):
    """Test StandardModelLoader."""
    
    @patch('model_loader.AutoModelForCausalLM')
    @patch('model_loader.AutoTokenizer')
    def test_load_model_standard(self, mock_tokenizer_cls, mock_model_cls):
        """Test standard model loading."""
        config = ModelLoadingConfig("test-model", Backend.CPU)
        loader = StandardModelLoader(config)
        
        # Mock model
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        
        model = loader.load_model()
        
        mock_model_cls.from_pretrained.assert_called_once_with(
            "test-model",
            use_cache=False,
            torch_dtype=torch.float16,
            _attn_implementation="sdpa",
            device_map={"": torch.device("cpu")},
            trust_remote_code=False,
        )
        self.assertEqual(model, mock_model)
    
    @patch('model_loader.AutoModelForCausalLM')
    @patch('model_loader.AutoConfig')
    @patch('model_loader.init_empty_weights')
    def test_load_model_low_memory(self, mock_init_empty, mock_config_cls, mock_model_cls):
        """Test low memory model loading."""
        # Rank 0 - loads to CPU
        config = ModelLoadingConfig(
            "test-model",
            Backend.CUDA,
            loading_strategy=LoadingStrategy.LOW_MEMORY,
            rank=0
        )
        loader = StandardModelLoader(config)
        
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        
        model = loader.load_model()
        mock_model.to.assert_called_with(dtype=torch.float16, device="cpu")
        
        # Rank 1 - creates empty model
        config.rank = 1
        mock_config = MagicMock()
        mock_config_cls.from_pretrained.return_value = mock_config
        
        model = loader.load_model()
        mock_model_cls.from_config.assert_called_with(
            mock_config,
            torch_dtype=torch.float16,
            trust_remote_code=False,
        )
    
    @patch('model_loader.AutoTokenizer')
    def test_load_tokenizer(self, mock_tokenizer_cls):
        """Test tokenizer loading."""
        config = ModelLoadingConfig("test-model", Backend.CPU)
        loader = StandardModelLoader(config)
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = loader.load_tokenizer()
        
        mock_tokenizer_cls.from_pretrained.assert_called_with(
            "test-model",
            use_fast=True,
            trust_remote_code=False,
        )
        self.assertEqual(tokenizer.pad_token_id, 1)


class TestQuantizedModelLoader(unittest.TestCase):
    """Test QuantizedModelLoader."""
    
    def test_validation_requires_config(self):
        """Test that quantization config is required."""
        config = ModelLoadingConfig("test", Backend.CUDA)
        
        with self.assertRaises(ValueError):
            QuantizedModelLoader(config)
    
    @patch('model_loader.check_import_availability')
    def test_validation_checks_imports(self, mock_check):
        """Test import validation."""
        quant_config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        config = ModelLoadingConfig(
            "test",
            Backend.CUDA,
            quantization_config=quant_config,
            quant_method="bnb"
        )
        
        # BNB not available
        mock_check.return_value = False
        with self.assertRaises(ImportError):
            QuantizedModelLoader(config)
        
        # BNB available
        mock_check.return_value = True
        loader = QuantizedModelLoader(config)
        self.assertIsNotNone(loader)
    
    @patch('model_loader.safetensors.torch.load_file')
    @patch('model_loader.parallel')
    @patch('model_loader.get_module')
    @patch('model_loader.check_import_availability')
    @patch('model_loader.AutoConfig')
    @patch('model_loader.AutoModelForCausalLM')
    @patch('model_loader.init_empty_weights')
    def test_load_quantized_model(
        self,
        mock_init_empty,
        mock_model_cls,
        mock_config_cls,
        mock_check,
        mock_get_module,
        mock_parallel,
        mock_load_file
    ):
        """Test loading and quantizing a model."""
        mock_check.return_value = True
        
        # Setup mocks
        mock_config = MagicMock()
        mock_config_cls.from_pretrained.return_value = mock_config
        
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("test", torch.randn(10, 10))]
        mock_model_cls.from_config.return_value = mock_model
        
        mock_bnb = MagicMock()
        mock_get_module.return_value = mock_bnb
        
        # Create loader
        quant_config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        config = ModelLoadingConfig(
            "test-model",
            Backend.CUDA,
            quantization_config=quant_config,
            quant_method="bnb"
        )
        
        with patch.object(QuantizedModelLoader, '_get_model_files', return_value=["test.safetensors"]):
            loader = QuantizedModelLoader(config)
            
            # Mock weight loading
            mock_load_file.return_value = {"weight": torch.randn(10, 10)}
            
            model = loader.load_model()
            
            # Check model was created and quantized
            self.assertTrue(mock_model_cls.from_config.called)
            self.assertTrue(hasattr(model, 'is_loaded_in_4bit'))


class TestBackendSpecificLoaders(unittest.TestCase):
    """Test backend-specific model loaders."""
    
    @patch('torch.cuda.is_available')
    def test_cuda_loader_validation(self, mock_cuda):
        """Test CUDA loader validation."""
        mock_cuda.return_value = False
        
        config = ModelLoadingConfig("test", Backend.CUDA)
        with self.assertRaises(RuntimeError):
            CUDAModelLoader(config)
        
        mock_cuda.return_value = True
        loader = CUDAModelLoader(config)
        self.assertIsNotNone(loader)
    
    @patch('torch.backends.mps.is_available')
    def test_mps_loader_validation(self, mock_mps):
        """Test MPS loader validation."""
        mock_mps.return_value = False
        
        config = ModelLoadingConfig("test", Backend.MPS)
        with self.assertRaises(RuntimeError):
            MPSModelLoader(config)
        
        mock_mps.return_value = True
        config = ModelLoadingConfig("test", Backend.MPS, dtype=torch.bfloat16)
        
        with self.assertWarns(UserWarning):
            loader = MPSModelLoader(config)
        
        # Should convert bfloat16 to float16
        self.assertEqual(loader.config.dtype, torch.float16)
    
    @patch('model_loader.check_import_availability')
    def test_mlx_loader_validation(self, mock_check):
        """Test MLX loader validation."""
        mock_check.return_value = False
        
        config = ModelLoadingConfig("test", Backend.MLX)
        with self.assertRaises(ImportError):
            MLXModelLoader(config)
        
        mock_check.return_value = True
        with self.assertWarns(UserWarning):  # MLX format warning
            loader = MLXModelLoader(config)
        self.assertIsNotNone(loader)
    
    def test_cpu_loader_warnings(self):
        """Test CPU loader warnings."""
        quant_config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        config = ModelLoadingConfig(
            "test",
            Backend.CPU,
            quantization_config=quant_config,
            quant_method="bnb"
        )
        
        with self.assertWarns(UserWarning):
            loader = CPUModelLoader(config)
        self.assertIsNotNone(loader)


class TestModelLoaderFactory(unittest.TestCase):
    """Test ModelLoaderFactory."""
    
    def test_create_standard_loaders(self):
        """Test creating standard loaders."""
        config = ModelLoadingConfig("test", Backend.CPU)
        loader = ModelLoaderFactory.create_loader(config)
        self.assertIsInstance(loader, StandardModelLoader)
    
    def test_create_quantized_loaders(self):
        """Test creating quantized loaders."""
        quant_config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        
        # CUDA
        config = ModelLoadingConfig("test", Backend.CUDA, quantization_config=quant_config)
        loader = ModelLoaderFactory.create_loader(config)
        self.assertIsInstance(loader, CUDAModelLoader)
        
        # CPU
        config = ModelLoadingConfig("test", Backend.CPU, quantization_config=quant_config)
        loader = ModelLoaderFactory.create_loader(config)
        self.assertIsInstance(loader, CPUModelLoader)
    
    @patch('model_loader.BackendManager')
    def test_auto_backend_detection(self, mock_backend_manager_cls):
        """Test automatic backend detection."""
        mock_manager = MagicMock()
        mock_manager.backend = Backend.MPS
        mock_backend_manager_cls.return_value = mock_manager
        
        config = ModelLoadingConfig("test", Backend.CPU)  # Will be overridden
        loader = ModelLoaderFactory.create_loader(config, mock_manager)
        
        # MPS loader even for non-quantized
        self.assertIsInstance(loader, MPSModelLoader)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    @patch('model_loader.ModelLoaderFactory.create_loader')
    def test_load_model_and_tokenizer(self, mock_create_loader):
        """Test load_model_and_tokenizer function."""
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        mock_loader.load_model.return_value = mock_model
        mock_loader.load_tokenizer.return_value = mock_tokenizer
        mock_create_loader.return_value = mock_loader
        
        model, tokenizer = load_model_and_tokenizer(
            "test-model",
            backend="cuda",
            low_memory=True
        )
        
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)
        
        # Check config was created correctly
        call_args = mock_create_loader.call_args[0][0]
        self.assertEqual(call_args.model_name, "test-model")
        self.assertEqual(call_args.backend, Backend.CUDA)
        self.assertTrue(call_args.low_memory)
    
    @patch('model_loader.get_recommended_config')
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_get_recommended_loader_config(self, mock_props, mock_cuda, mock_get_config):
        """Test recommended configuration generation."""
        mock_cuda.return_value = True
        mock_props.return_value.total_memory = 16e9
        
        mock_quant_config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        mock_get_config.return_value = mock_quant_config
        
        # Test with 70B model
        config = get_recommended_loader_config(
            "meta-llama/Llama-2-70b-hf",
            Backend.CUDA
        )
        
        # Should recommend low memory for 70B on 16GB
        self.assertEqual(config.loading_strategy, LoadingStrategy.LOW_MEMORY)
        self.assertTrue(config.low_memory)
        self.assertEqual(config.quantization_config, mock_quant_config)
        
        # Test with small model
        config = get_recommended_loader_config(
            "meta-llama/Llama-2-3b-hf",
            Backend.MPS,
            available_memory_gb=32.0
        )
        
        # Should use unified memory for MPS
        self.assertEqual(config.loading_strategy, LoadingStrategy.UNIFIED_MEMORY)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete integration scenarios."""
    
    @patch('model_loader.AutoModelForCausalLM')
    @patch('model_loader.AutoTokenizer')
    @patch('model_loader.hub')
    def test_full_loading_scenario(self, mock_hub, mock_tokenizer_cls, mock_model_cls):
        """Test a complete model loading scenario."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            "meta-llama/Llama-2-7b-hf",
            backend=Backend.CPU,
            dtype=torch.float32,
            use_cache=True
        )
        
        # Verify calls
        mock_model_cls.from_pretrained.assert_called()
        mock_tokenizer_cls.from_pretrained.assert_called()
        
        # Check configuration was applied
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs['torch_dtype'], torch.float32)
        self.assertTrue(call_kwargs['use_cache'])


if __name__ == '__main__':
    unittest.main()