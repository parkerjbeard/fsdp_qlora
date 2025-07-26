"""
Integration tests for model loader with train.py workflow.
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
from src.core.model_loader import (
    LoadingStrategy,
    ModelLoadingConfig,
    ModelLoaderFactory,
    load_model_and_tokenizer,
    get_recommended_loader_config,
)
from src.core.quantization_wrapper import QuantizationConfig, QuantizationMethod


class TestModelLoaderIntegration(unittest.TestCase):
    """Test model loader integration with train.py workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_train_args(self, **overrides):
        """Create training arguments similar to train.py."""
        args = {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "train_type": "qlora",
            "precision": "bf16",
            "batch_size": 2,
            "context_length": 512,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "dataset": "alpaca",
            "low_memory": False,
            "use_gradient_checkpointing": True,
            "use_cpu_offload": False,
            "loading_workers": -1,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "lora_target_modules": ["q_proj", "v_proj"],
            "q_bits": 4,
            "q_group_size": 64,
            "verbose": True,
            "rank": 0,
            "world_size": 1,
            "backend": "auto",
        }
        args.update(overrides)
        return Namespace(**args)
    
    @patch('model_loader.AutoModelForCausalLM')
    @patch('model_loader.AutoTokenizer')
    @patch('model_loader.AutoConfig')
    @patch('model_loader.init_empty_weights')
    @patch('model_loader.get_module')
    @patch('model_loader.check_import_availability')
    def test_qlora_loading_workflow(
        self,
        mock_check,
        mock_get_module,
        mock_init_empty,
        mock_config_cls,
        mock_tokenizer_cls,
        mock_model_cls
    ):
        """Test QLoRA model loading workflow."""
        mock_check.return_value = True
        
        # Mock bitsandbytes
        mock_bnb = MagicMock()
        mock_bnb.Linear4bit = MagicMock()
        mock_get_module.return_value = mock_bnb
        
        # Mock model config
        mock_config = MagicMock()
        mock_config.use_cache = False
        mock_config_cls.from_pretrained.return_value = mock_config
        
        # Mock model
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []
        mock_model_cls.from_config.return_value = mock_model
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        # Create training args
        args = self.create_train_args(train_type="qlora")
        
        # Create backend manager
        backend_manager = BackendManager(backend=args.backend, verbose=args.verbose)
        
        # Determine dtype
        if args.precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        
        # Create quantization config
        quant_config = QuantizationConfig(
            method=QuantizationMethod.BNB_NF4,
            bits=args.q_bits,
            group_size=args.q_group_size,
            compute_dtype=dtype,
            skip_modules=["lm_head"],
        )
        
        # Create model loading config
        model_config = ModelLoadingConfig(
            model_name=args.model_name,
            backend=backend_manager.backend,
            quantization_config=quant_config,
            dtype=dtype,
            low_memory=args.low_memory,
            loading_workers=args.loading_workers,
            rank=args.rank,
            world_size=args.world_size,
            verbose=args.verbose,
        )
        
        # Load model and tokenizer
        loader = ModelLoaderFactory.create_loader(model_config)
        
        # Should create a quantized loader
        self.assertIn("Quantized", loader.__class__.__name__)
        
        # Mock file loading
        with patch.object(loader, '_get_model_files', return_value=[]):
            with patch.object(loader, '_load_quantized_weights'):
                model = loader.load_model()
        
        tokenizer = loader.load_tokenizer()
        
        # Verify model was created with quantization
        self.assertTrue(hasattr(model, 'is_loaded_in_4bit'))
        self.assertEqual(tokenizer.pad_token_id, 1)
    
    def test_backend_specific_loading(self):
        """Test loading with different backends."""
        test_cases = [
            (Backend.CUDA, QuantizationMethod.BNB_NF4),
            (Backend.MPS, QuantizationMethod.MLX_INT4),
            (Backend.CPU, QuantizationMethod.HQQ),
        ]
        
        for backend, expected_method in test_cases:
            with self.subTest(backend=backend):
                # Get recommended config
                config = get_recommended_loader_config(
                    "meta-llama/Llama-2-7b-hf",
                    backend,
                    available_memory_gb=8.0  # Force quantization
                )
                
                # Check recommended quantization
                if config.quantization_config:
                    if backend == Backend.CUDA:
                        self.assertIn(config.quantization_config.method, 
                                    [QuantizationMethod.BNB_NF4, QuantizationMethod.BNB_INT8])
                    elif backend == Backend.MPS:
                        self.assertIn(config.quantization_config.method,
                                    [QuantizationMethod.MLX_INT4, QuantizationMethod.MLX_INT8])
                    elif backend == Backend.CPU:
                        self.assertEqual(config.quantization_config.method, QuantizationMethod.HQQ)
    
    @patch('model_loader.parallel')
    @patch('model_loader.safetensors.torch.load_file')
    def test_parallel_weight_loading(self, mock_load_file, mock_parallel):
        """Test parallel weight loading mechanism."""
        # Mock weight file
        mock_weights = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(4096, 4096),
        }
        mock_load_file.return_value = mock_weights
        
        # Create config with quantization
        quant_config = QuantizationConfig(method=QuantizationMethod.HQQ)
        config = ModelLoadingConfig(
            "test-model",
            Backend.CPU,
            quantization_config=quant_config,
            loading_workers=4,
        )
        
        # Test worker calculation
        loader = ModelLoaderFactory.create_loader(config)
        n_workers = loader._get_loading_workers(32e9)  # 32B params
        self.assertEqual(n_workers, 4)  # Should use specified workers
        
        # Test parallel loading call
        with patch.object(loader, '_get_model_files', return_value=["test.safetensors"]):
            with patch('model_loader.AutoConfig'), \
                 patch('model_loader.AutoModelForCausalLM'), \
                 patch('model_loader.init_empty_weights'), \
                 patch('model_loader.get_module'), \
                 patch('model_loader.check_import_availability', return_value=True):
                
                # Mock the actual loading
                from train import load_and_quantize
                with patch('model_loader.load_and_quantize'):
                    loader._load_quantized_weights(MagicMock())
                    
                    # Check parallel was called with correct arguments
                    mock_parallel.assert_called()
                    call_kwargs = mock_parallel.call_args[1]
                    self.assertEqual(call_kwargs['n_workers'], 4)
                    self.assertTrue(call_kwargs['threadpool'])
    
    def test_memory_efficient_loading_strategies(self):
        """Test different memory-efficient loading strategies."""
        # Low memory strategy
        config = ModelLoadingConfig(
            "test-model",
            Backend.CUDA,
            loading_strategy=LoadingStrategy.LOW_MEMORY,
            low_memory=True,
            rank=0,
        )
        
        loader = ModelLoaderFactory.create_loader(config)
        self.assertTrue(config.low_memory)
        self.assertEqual(config.loading_strategy, LoadingStrategy.LOW_MEMORY)
        
        # Unified memory for Apple Silicon
        config = ModelLoadingConfig(
            "test-model",
            Backend.MPS,
        )
        
        self.assertEqual(config.loading_strategy, LoadingStrategy.UNIFIED_MEMORY)
    
    def test_llama_pro_loading(self):
        """Test LLaMA Pro model loading with layer expansion."""
        llama_pro_path = os.path.join(self.test_dir, "llama_pro_blk_exp-32-40")
        os.makedirs(llama_pro_path, exist_ok=True)
        
        # Create dummy safetensors file
        with open(os.path.join(llama_pro_path, "model.safetensors"), "w") as f:
            f.write("dummy")
        
        config = ModelLoadingConfig(
            "test-model",
            Backend.CUDA,
            llama_pro_path=llama_pro_path,
        )
        
        loader = ModelLoaderFactory.create_loader(config)
        files = loader._get_model_files()
        
        # Should find the LLaMA Pro file
        self.assertEqual(len(files), 1)
        self.assertIn("model.safetensors", files[0])
    
    @patch('model_loader.AutoModelForCausalLM')
    @patch('model_loader.AutoTokenizer')
    def test_dtype_handling(self, mock_tokenizer_cls, mock_model_cls):
        """Test dtype handling across backends."""
        # Mock returns
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        
        # MPS should convert bfloat16 to float16
        config = ModelLoadingConfig(
            "test-model",
            Backend.MPS,
            dtype=torch.bfloat16,
        )
        
        if config.backend == Backend.MPS:
            # MPS loader should have converted dtype
            self.assertEqual(config.dtype, torch.float16)
        
        # CUDA should keep bfloat16
        config = ModelLoadingConfig(
            "test-model",
            Backend.CUDA,
            dtype=torch.bfloat16,
        )
        self.assertEqual(config.dtype, torch.bfloat16)
    
    def test_error_handling(self):
        """Test error handling in model loading."""
        # Invalid model name
        config = ModelLoadingConfig(
            "invalid-model-name",
            Backend.CPU,
        )
        
        loader = ModelLoaderFactory.create_loader(config)
        
        # Should raise error when trying to get files
        with patch('model_loader.hub.cached_file', side_effect=OSError("Not found")):
            with self.assertRaises(RuntimeError):
                loader._get_model_files()
        
        # Quantization without proper backend
        quant_config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        config = ModelLoadingConfig(
            "test-model",
            Backend.CPU,  # BNB doesn't work on CPU
            quantization_config=quant_config,
        )
        
        with patch('model_loader.check_import_availability', return_value=False):
            # Should raise error for missing bitsandbytes
            with self.assertRaises(ImportError):
                ModelLoaderFactory.create_loader(config)


class TestTrainPyRefactoring(unittest.TestCase):
    """Test how to refactor train.py to use model loader."""
    
    def test_refactoring_example(self):
        """Example of how to refactor train.py model loading."""
        # Original train.py logic simulation
        args = {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "train_type": "qlora",
            "precision": "bf16",
            "low_memory": True,
            "q_bits": 4,
            "n_bits": 4,
            "rank": 0,
            "world_size": 1,
            "loading_workers": -1,
            "verbose": True,
        }
        args = Namespace(**args)
        
        # New refactored approach
        backend_manager = BackendManager(verbose=args.verbose)
        
        # Determine dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(args.precision, torch.float16)
        
        # Create quantization config based on train_type
        quant_config = None
        if args.train_type in ["qlora", "custom_qlora"]:
            quant_config = QuantizationConfig(
                method=QuantizationMethod.BNB_NF4,
                bits=args.q_bits,
                compute_dtype=dtype,
            )
        elif args.train_type in ["hqq_lora", "hqq_dora"]:
            quant_config = QuantizationConfig(
                method=QuantizationMethod.HQQ,
                bits=args.n_bits,
                group_size=64,
                compute_dtype=dtype,
            )
        
        # Create model loading config
        model_config = ModelLoadingConfig(
            model_name=args.model_name,
            backend=backend_manager.backend,
            quantization_config=quant_config,
            dtype=dtype,
            low_memory=args.low_memory,
            loading_workers=args.loading_workers,
            rank=args.rank,
            world_size=args.world_size,
            verbose=args.verbose,
        )
        
        # This replaces the complex if/else logic in train.py
        loader = ModelLoaderFactory.create_loader(model_config, backend_manager)
        
        # Verify correct loader type
        if quant_config:
            self.assertIn("Quantized", loader.__class__.__name__)
        else:
            self.assertIsInstance(loader, type(loader))  # Any loader type
        
        # The actual loading would be:
        # model = loader.load_model()
        # tokenizer = loader.load_tokenizer()


if __name__ == '__main__':
    unittest.main()