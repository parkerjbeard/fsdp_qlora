"""
Integration tests for MPS FSDP wrapper.

These tests demonstrate real-world usage scenarios including:
- Training with FSDP on MPS
- Memory optimization strategies
- Multi-process training simulation
- Checkpoint save/load workflows
- Integration with quantization
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import shutil
import warnings
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend, BackendManager
from src.backends.mps.mps_fsdp_wrapper import (
    MPSFSDPConfig,
    MPSFSDPWrapper,
    create_mps_fsdp_wrapper,
    wrap_model_for_mps,
    check_mps_fsdp_compatibility,
)
from src.core.quantization_wrapper import QuantizationConfig, QuantizationMethod


class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, size=100, seq_length=128, vocab_size=1000):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_length,)),
            "labels": torch.randint(0, self.vocab_size, (self.seq_length,)),
        }


class LlamaBlock(nn.Module):
    """Simplified LLaMA-style transformer block."""
    
    def __init__(self, hidden_size=768, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
    
    def forward(self, x):
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleLlamaModel(nn.Module):
    """Simplified LLaMA model for testing."""
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=2048,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LlamaBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        
        # Store config for FSDP
        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        }
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.output(x)


class TestMPSFSDPTraining(unittest.TestCase):
    """Test FSDP training workflows on MPS."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Mock MPS availability
        self.mps_patcher = patch('torch.backends.mps.is_available', return_value=True)
        self.mps_patcher.start()
        
        # Mock distributed
        self.dist_init_patcher = patch('torch.distributed.is_initialized', return_value=False)
        self.dist_init_patcher.start()
    
    def tearDown(self):
        """Clean up test environment."""
        self.mps_patcher.stop()
        self.dist_init_patcher.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    def test_simple_training_workflow(self, mock_fsdp_class):
        """Test simple training workflow with FSDP on MPS."""
        # Create model
        model = SimpleLlamaModel(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
        )
        
        # Calculate model size
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        # Create FSDP wrapper
        wrapper = create_mps_fsdp_wrapper(
            sharding_strategy="FULL_SHARD",
            mixed_precision=True,
            min_num_params=10_000,  # Wrap layers with >10k params
        )
        
        # Mock FSDP wrapping
        mock_fsdp = MagicMock()
        mock_fsdp_class.return_value = mock_fsdp
        
        # Mock forward/backward
        mock_fsdp.return_value = torch.randn(2, 128, 1000)  # Mock output
        
        # Wrap model
        fsdp_model = wrapper.wrap_model(model)
        
        # Verify FSDP was called with correct config
        mock_fsdp_class.assert_called_once()
        call_kwargs = mock_fsdp_class.call_args[1]
        
        # Check MPS-specific settings
        self.assertEqual(call_kwargs['device_id'], torch.device('mps'))
        self.assertIsNotNone(call_kwargs['mixed_precision'])
        
        # Check mixed precision uses float16, not bfloat16
        mp_config = call_kwargs['mixed_precision']
        if hasattr(mp_config, 'param_dtype'):
            self.assertNotEqual(mp_config.param_dtype, torch.bfloat16)
    
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    @patch('torch.distributed.init_process_group')
    def test_multi_process_setup(self, mock_init_pg, mock_fsdp):
        """Test multi-process FSDP setup with Gloo."""
        # Create config for multi-process
        config = MPSFSDPConfig(
            world_size=2,
            rank=0,
            backend="gloo",  # Must use Gloo for MPS
        )
        
        wrapper = MPSFSDPWrapper(config)
        
        # Verify Gloo was used
        mock_init_pg.assert_called_once()
        init_kwargs = mock_init_pg.call_args[1]
        self.assertEqual(init_kwargs['backend'], 'gloo')
        
        # Create and wrap model
        model = SimpleLlamaModel(hidden_size=128, num_layers=2)
        mock_fsdp.return_value = MagicMock()
        
        fsdp_model = wrapper.wrap_transformer(model, LlamaBlock)
        
        # Verify transformer auto-wrap policy was used
        call_kwargs = mock_fsdp.call_args[1]
        self.assertIsNotNone(call_kwargs['auto_wrap_policy'])
    
    def test_memory_optimization_strategies(self):
        """Test memory optimization for different model sizes."""
        test_cases = [
            # (model_size_gb, available_gb, expected_strategy)
            (0.5, 16, ShardingStrategy.NO_SHARD),  # Small model
            (8, 16, ShardingStrategy.SHARD_GRAD_OP),  # Medium model
            (14, 16, ShardingStrategy.FULL_SHARD),  # Large model
        ]
        
        wrapper = MPSFSDPWrapper()
        
        for model_gb, available_gb, expected_strategy in test_cases:
            model_size = int(model_gb * 1e9)
            available_memory = int(available_gb * 1e9)
            
            strategy = wrapper.memory_optimizer.optimize_sharding_strategy(
                model_size, available_memory
            )
            
            self.assertEqual(
                strategy, expected_strategy,
                f"Failed for {model_gb}GB model with {available_gb}GB memory"
            )
    
    @patch('torch.save')
    @patch('torch.load')
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    def test_checkpoint_workflow(self, mock_fsdp_class, mock_load, mock_save):
        """Test checkpoint save/load workflow."""
        # Create wrapper
        wrapper = MPSFSDPWrapper()
        wrapper.config.rank = 0  # Set as primary rank
        
        # Create model
        model = SimpleLlamaModel(hidden_size=128, num_layers=2)
        
        # Mock FSDP model
        mock_fsdp_model = MagicMock()
        mock_fsdp_model.state_dict.return_value = {
            "embedding.weight": torch.randn(1000, 128),
            "output.weight": torch.randn(1000, 128),
        }
        mock_fsdp_class.return_value = mock_fsdp_model
        
        # Wrap model
        fsdp_model = wrapper.wrap_model(model)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.test_dir, "checkpoint.pt")
        
        with patch('torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type'):
            wrapper.save_checkpoint(
                fsdp_model,
                checkpoint_path,
                optimizer=optimizer,
                epoch=5,
                global_step=1000,
            )
        
        # Verify save was called
        mock_save.assert_called_once()
        saved_checkpoint = mock_save.call_args[0][1]
        
        # Check checkpoint contents
        self.assertIn("model_state_dict", saved_checkpoint)
        self.assertIn("config", saved_checkpoint)
        self.assertEqual(saved_checkpoint["epoch"], 5)
        self.assertEqual(saved_checkpoint["global_step"], 1000)
        
        # Test loading
        mock_load.return_value = saved_checkpoint
        
        with patch('torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type'):
            loaded_checkpoint = wrapper.load_checkpoint(
                fsdp_model,
                checkpoint_path,
                optimizer=optimizer,
            )
        
        # Verify load was called
        mock_load.assert_called_once_with(checkpoint_path, map_location="cpu")
        mock_fsdp_model.load_state_dict.assert_called_once()


class TestMPSFSDPWithQuantization(unittest.TestCase):
    """Test FSDP with quantization on MPS."""
    
    def setUp(self):
        """Set up test environment."""
        self.mps_patcher = patch('torch.backends.mps.is_available', return_value=True)
        self.mps_patcher.start()
    
    def tearDown(self):
        """Clean up."""
        self.mps_patcher.stop()
    
    def test_dtype_compatibility_with_quantization(self):
        """Test dtype handling with quantization."""
        # Create config that would use bfloat16 on CUDA
        wrapper = MPSFSDPWrapper()
        
        # Create model with mixed precision that includes bfloat16
        model = SimpleLlamaModel(hidden_size=128, num_layers=2)
        
        # Simulate quantization that might use bfloat16
        for name, param in model.named_parameters():
            if "embedding" in name:
                param.data = param.data.to(torch.bfloat16)
        
        # Convert model
        converted_model = wrapper._convert_dtype(model, torch.float16)
        
        # Verify all bfloat16 params were converted
        for name, param in converted_model.named_parameters():
            self.assertNotEqual(
                param.dtype, torch.bfloat16,
                f"Parameter {name} still has bfloat16 dtype"
            )
            if "embedding" in name:
                self.assertEqual(param.dtype, torch.float16)
    
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    def test_fsdp_with_hqq_quantization(self, mock_fsdp):
        """Test FSDP wrapper with HQQ quantization (MPS-compatible)."""
        from quantization_wrapper import QuantizationConfig
        
        # Create quantization config for MPS
        quant_config = QuantizationConfig(
            method=QuantizationMethod.HQQ_4BIT,  # HQQ works on MPS
            bits=4,
            compute_dtype=torch.float16,  # Not bfloat16
        )
        
        # Create wrapper with quantization awareness
        wrapper = MPSFSDPWrapper()
        
        # Create model
        model = SimpleLlamaModel(hidden_size=256, num_layers=4)
        
        # Mock FSDP
        mock_fsdp.return_value = MagicMock()
        
        # Wrap with correct dtype
        fsdp_model = wrapper.wrap_model(
            model,
            param_dtype=quant_config.compute_dtype,
        )
        
        # Verify correct dtype was used
        call_kwargs = mock_fsdp.call_args[1]
        if 'mixed_precision' in call_kwargs and call_kwargs['mixed_precision']:
            mp_config = call_kwargs['mixed_precision']
            # Should not use bfloat16
            self.assertNotEqual(getattr(mp_config, 'param_dtype', None), torch.bfloat16)


class TestMPSOperatorFallbacks(unittest.TestCase):
    """Test operator fallback handling."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MPSFSDPConfig(
            fallback_to_cpu_ops=[
                "aten::_fused_adam",
                "aten::_foreach_add",
                "aten::nll_loss_backward",
            ]
        )
        self.wrapper = MPSFSDPWrapper(self.config)
    
    def test_fallback_context_usage(self):
        """Test fallback context during operations."""
        # Test that environment variable is set correctly
        self.assertIsNone(os.environ.get("PYTORCH_MPS_FALLBACK"))
        
        with self.wrapper.operator_fallback.fallback_context():
            self.assertEqual(os.environ.get("PYTORCH_MPS_FALLBACK"), "1")
            
            # Simulate operation that might need fallback
            try:
                # This would normally trigger fallback for unsupported ops
                tensor = torch.randn(10, 10)
                # Some operations might fallback to CPU
                result = tensor.sum()
            except Exception as e:
                # In real scenario, PyTorch handles fallback internally
                pass
        
        # Environment should be cleaned up
        self.assertIsNone(os.environ.get("PYTORCH_MPS_FALLBACK"))
    
    @patch('mps_fsdp_wrapper.logger')
    def test_operator_warnings(self, mock_logger):
        """Test warnings for potentially problematic operators."""
        # Create wrapper with known problematic ops
        config = MPSFSDPConfig(
            fallback_to_cpu_ops=[
                "aten::_fused_adam",
                "aten::native_layer_norm_backward",
                "aten::embedding_dense_backward",
            ]
        )
        
        wrapper = MPSFSDPWrapper(config)
        
        # Should warn about fallback operators
        warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
        self.assertTrue(
            any("aten::_fused_adam" in warning for warning in warning_calls)
        )


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.mps_patcher = patch('torch.backends.mps.is_available', return_value=True)
        self.mps_patcher.start()
    
    def tearDown(self):
        """Clean up."""
        self.mps_patcher.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    def test_7b_model_configuration(self, mock_fsdp):
        """Test configuration for 7B parameter model."""
        # Create 7B-like model (scaled down for testing)
        model = SimpleLlamaModel(
            vocab_size=32000,
            hidden_size=4096,  # Real 7B size
            num_layers=2,  # Reduced for testing (real: 32)
        )
        
        # Estimate model size
        param_count = sum(p.numel() for p in model.parameters())
        model_size_gb = param_count * 2 / 1e9  # FP16
        
        # Create config for 7B model on M2 Max (64GB)
        available_memory_gb = 64
        
        wrapper = MPSFSDPWrapper()
        
        # Get optimal sharding strategy
        strategy = wrapper.memory_optimizer.optimize_sharding_strategy(
            int(model_size_gb * 1e9),
            int(available_memory_gb * 1e9),
        )
        
        # Update config
        wrapper.config.sharding_strategy = strategy
        
        # Mock FSDP
        mock_fsdp.return_value = MagicMock()
        
        # Wrap model
        fsdp_model = wrapper.wrap_transformer(model, LlamaBlock)
        
        # Verify configuration
        call_kwargs = mock_fsdp.call_args[1]
        self.assertEqual(call_kwargs['sharding_strategy'], strategy)
        
        # For 7B model with 64GB, should use NO_SHARD or SHARD_GRAD_OP
        self.assertIn(strategy, [ShardingStrategy.NO_SHARD, ShardingStrategy.SHARD_GRAD_OP])
    
    @patch('torch.mps.set_per_process_memory_fraction')
    @patch('torch.mps.empty_cache')
    def test_memory_pressure_handling(self, mock_empty_cache, mock_set_memory):
        """Test handling of memory pressure scenarios."""
        # Create config with aggressive memory optimization
        config = MPSFSDPConfig(
            unified_memory_pool_size=32 * 1024 * 1024 * 1024,  # 32GB
            aggressive_memory_optimization=True,
            cpu_offload=True,  # Enable CPU offload for memory pressure
        )
        
        wrapper = MPSFSDPWrapper(config)
        
        # Setup memory pool
        wrapper.memory_optimizer.setup_memory_pool()
        
        # Verify memory optimization was applied
        mock_set_memory.assert_called_with(0.8)  # 80% of memory
        mock_empty_cache.assert_called_once()  # Cache cleared
        
        # Test with large model that needs offloading
        model_size = 50 * 1024 * 1024 * 1024  # 50GB model
        available = 32 * 1024 * 1024 * 1024  # 32GB available
        
        strategy = wrapper.memory_optimizer.optimize_sharding_strategy(
            model_size, available
        )
        
        # Should use full sharding for large model
        self.assertEqual(strategy, ShardingStrategy.FULL_SHARD)
    
    def test_compatibility_matrix(self):
        """Test compatibility checking across PyTorch versions."""
        test_versions = [
            ("1.12.0", True),   # Should warn
            ("2.0.0", False),   # OK
            ("2.7.0", False),   # Best support
            ("2.10.0", False),  # Future version
        ]
        
        for version, should_warn in test_versions:
            with patch('torch.__version__', version):
                info = check_mps_fsdp_compatibility()
                
                has_warnings = len(info.get("warnings", [])) > 0
                self.assertEqual(
                    has_warnings, should_warn,
                    f"Version {version} warning mismatch"
                )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_no_mps_available(self, mock_mps):
        """Test behavior when MPS is not available."""
        with self.assertRaises(RuntimeError) as ctx:
            MPSFSDPWrapper()
        
        self.assertIn("MPS backend is not available", str(ctx.exception))
    
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_invalid_backend_override(self, mock_mps):
        """Test backend override to Gloo."""
        # Try to use NCCL (not supported on MPS)
        config = MPSFSDPConfig(backend="nccl", world_size=2)
        
        with warnings.catch_warnings(record=True) as w:
            wrapper = MPSFSDPWrapper(config)
            
            # Should warn and switch to Gloo
            self.assertTrue(len(w) > 0)
            self.assertEqual(wrapper.config.backend, "gloo")
    
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_empty_model_handling(self, mock_mps):
        """Test handling of empty or very small models."""
        wrapper = MPSFSDPWrapper()
        
        # Create tiny model
        model = nn.Linear(2, 2)
        
        # Should handle without errors
        with patch('torch.distributed.fsdp.FullyShardedDataParallel') as mock_fsdp:
            mock_fsdp.return_value = MagicMock()
            
            wrapped = wrapper.wrap_model(model)
            
            # Should still wrap even tiny models
            mock_fsdp.assert_called_once()


if __name__ == '__main__':
    unittest.main()