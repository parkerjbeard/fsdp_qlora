"""
Tests for MPS FSDP wrapper.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend, BackendManager
from src.backends.mps.mps_fsdp_wrapper import (
    MPSFSDPConfig,
    MPSFSDPWrapper,
    MPSOperatorFallback,
    UnifiedMemoryOptimizer,
    create_mps_fsdp_wrapper,
    wrap_model_for_mps,
    check_mps_fsdp_compatibility,
)


class TestMPSFSDPConfig(unittest.TestCase):
    """Test MPS FSDP configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MPSFSDPConfig()
        
        # Check defaults
        self.assertEqual(config.sharding_strategy, ShardingStrategy.FULL_SHARD)
        self.assertEqual(config.min_num_params, 1e6)
        self.assertTrue(config.use_mixed_precision)
        self.assertEqual(config.compute_dtype, torch.float16)  # Not bfloat16
        self.assertEqual(config.backend, "gloo")  # Not NCCL
        self.assertTrue(config.use_orig_params)
    
    def test_mps_specific_settings(self):
        """Test MPS-specific configuration."""
        config = MPSFSDPConfig()
        
        # MPS should use float16, not bfloat16
        self.assertEqual(config.compute_dtype, torch.float16)
        self.assertEqual(config.buffer_dtype, torch.float16)
        
        # Should use Gloo backend
        self.assertEqual(config.backend, "gloo")
        
        # Check fallback operators
        self.assertIn("aten::_fused_adam", config.fallback_to_cpu_ops)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MPSFSDPConfig(
            sharding_strategy=ShardingStrategy.NO_SHARD,
            min_num_params=1e7,
            use_mixed_precision=False,
            cpu_offload=True,
            aggressive_memory_optimization=True,
        )
        
        self.assertEqual(config.sharding_strategy, ShardingStrategy.NO_SHARD)
        self.assertEqual(config.min_num_params, 1e7)
        self.assertFalse(config.use_mixed_precision)
        self.assertTrue(config.cpu_offload)
        self.assertTrue(config.aggressive_memory_optimization)


class TestMPSOperatorFallback(unittest.TestCase):
    """Test MPS operator fallback handling."""
    
    def test_initialization(self):
        """Test operator fallback initialization."""
        fallback_ops = ["aten::op1", "aten::op2"]
        fallback = MPSOperatorFallback(fallback_ops)
        
        self.assertEqual(fallback.fallback_ops, {"aten::op1", "aten::op2"})
    
    def test_fallback_context(self):
        """Test fallback context manager."""
        fallback = MPSOperatorFallback(["aten::test_op"])
        
        # Check environment variable is set
        with fallback.fallback_context():
            self.assertEqual(os.environ.get("PYTORCH_MPS_FALLBACK"), "1")
        
        # Check it's cleaned up
        self.assertIsNone(os.environ.get("PYTORCH_MPS_FALLBACK"))
    
    @patch('mps_fsdp_wrapper.logger')
    def test_patch_unsupported_ops_warning(self, mock_logger):
        """Test warning for unsupported operators."""
        fallback = MPSOperatorFallback(["aten::op1", "aten::op2"])
        fallback.patch_unsupported_ops()
        
        # Should log warning about fallback operators
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("aten::op1", warning_msg)
        self.assertIn("aten::op2", warning_msg)


class TestUnifiedMemoryOptimizer(unittest.TestCase):
    """Test unified memory optimization."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = MPSFSDPConfig(
            unified_memory_pool_size=8 * 1024 * 1024 * 1024,  # 8GB
            aggressive_memory_optimization=True,
        )
        self.optimizer = UnifiedMemoryOptimizer(self.config)
    
    @patch('torch.mps.set_per_process_memory_fraction')
    @patch('torch.mps.empty_cache')
    def test_setup_memory_pool(self, mock_empty_cache, mock_set_memory):
        """Test memory pool setup."""
        self.optimizer.setup_memory_pool()
        
        # Should set memory fraction
        mock_set_memory.assert_called_once_with(0.8)
        
        # Should empty cache for aggressive optimization
        mock_empty_cache.assert_called_once()
    
    def test_optimize_sharding_strategy(self):
        """Test sharding strategy optimization."""
        # Large model relative to memory
        strategy = self.optimizer.optimize_sharding_strategy(
            model_size=9 * 1024 * 1024 * 1024,  # 9GB
            available_memory=10 * 1024 * 1024 * 1024,  # 10GB
        )
        self.assertEqual(strategy, ShardingStrategy.FULL_SHARD)
        
        # Medium model
        strategy = self.optimizer.optimize_sharding_strategy(
            model_size=6 * 1024 * 1024 * 1024,  # 6GB
            available_memory=10 * 1024 * 1024 * 1024,  # 10GB
        )
        self.assertEqual(strategy, ShardingStrategy.SHARD_GRAD_OP)
        
        # Small model
        strategy = self.optimizer.optimize_sharding_strategy(
            model_size=2 * 1024 * 1024 * 1024,  # 2GB
            available_memory=10 * 1024 * 1024 * 1024,  # 10GB
        )
        self.assertEqual(strategy, ShardingStrategy.NO_SHARD)
    
    def test_get_optimal_bucket_size(self):
        """Test optimal bucket size calculation."""
        # Large model
        bucket_size = UnifiedMemoryOptimizer.get_optimal_bucket_size(2e9)  # 2B params
        self.assertEqual(bucket_size, 200_000_000)
        
        # Small model
        bucket_size = UnifiedMemoryOptimizer.get_optimal_bucket_size(100e6)  # 100M params
        self.assertEqual(bucket_size, 50_000_000)


class TestMPSFSDPWrapper(unittest.TestCase):
    """Test MPS FSDP wrapper."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Mock MPS availability
        self.mps_patcher = patch('torch.backends.mps.is_available', return_value=True)
        self.mps_patcher.start()
        
        # Mock distributed
        self.dist_patcher = patch('torch.distributed.is_initialized', return_value=False)
        self.dist_patcher.start()
    
    def tearDown(self):
        """Clean up test environment."""
        self.mps_patcher.stop()
        self.dist_patcher.stop()
        
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = MPSFSDPWrapper()
        
        self.assertIsInstance(wrapper.config, MPSFSDPConfig)
        self.assertIsInstance(wrapper.operator_fallback, MPSOperatorFallback)
        self.assertIsInstance(wrapper.memory_optimizer, UnifiedMemoryOptimizer)
        self.assertFalse(wrapper._initialized)
    
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_initialization_no_mps(self, mock_mps):
        """Test initialization without MPS."""
        with self.assertRaises(RuntimeError) as ctx:
            MPSFSDPWrapper()
        
        self.assertIn("MPS backend is not available", str(ctx.exception))
    
    @patch('torch.distributed.init_process_group')
    def test_distributed_initialization(self, mock_init_pg):
        """Test distributed initialization."""
        config = MPSFSDPConfig(world_size=2, rank=0)
        wrapper = MPSFSDPWrapper(config)
        
        # Should initialize with Gloo
        mock_init_pg.assert_called_once()
        call_kwargs = mock_init_pg.call_args[1]
        self.assertEqual(call_kwargs['backend'], 'gloo')
        self.assertEqual(call_kwargs['world_size'], 2)
        self.assertEqual(call_kwargs['rank'], 0)
    
    def test_backend_override_warning(self):
        """Test warning when trying to use non-Gloo backend."""
        config = MPSFSDPConfig(backend="nccl", world_size=2)
        
        with warnings.catch_warnings(record=True) as w:
            wrapper = MPSFSDPWrapper(config)
            
            # Should warn about backend
            self.assertTrue(any("only supports Gloo" in str(warning.message) for warning in w))
            # Should switch to Gloo
            self.assertEqual(wrapper.config.backend, "gloo")
    
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    def test_wrap_model_basic(self, mock_fsdp):
        """Test basic model wrapping."""
        wrapper = MPSFSDPWrapper()
        
        # Create simple model
        model = nn.Linear(10, 10)
        
        # Mock FSDP return
        mock_wrapped = MagicMock()
        mock_fsdp.return_value = mock_wrapped
        
        # Wrap model
        wrapped_model = wrapper.wrap_model(model)
        
        # Check FSDP was called
        mock_fsdp.assert_called_once()
        call_kwargs = mock_fsdp.call_args[1]
        
        # Check key parameters
        self.assertEqual(call_kwargs['sharding_strategy'], ShardingStrategy.FULL_SHARD)
        self.assertEqual(call_kwargs['device_id'], torch.device('mps'))
        self.assertTrue(call_kwargs['use_orig_params'])
    
    def test_dtype_conversion(self):
        """Test bfloat16 to float16 conversion."""
        wrapper = MPSFSDPWrapper()
        
        # Create model with bfloat16 parameters
        model = nn.Linear(10, 10)
        model = model.to(torch.bfloat16)
        
        # Convert dtype
        converted_model = wrapper._convert_dtype(model, torch.float16)
        
        # Check all parameters are float16
        for param in converted_model.parameters():
            self.assertEqual(param.dtype, torch.float16)
    
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    def test_wrap_model_with_bfloat16_warning(self, mock_fsdp):
        """Test warning when using bfloat16."""
        wrapper = MPSFSDPWrapper()
        
        model = nn.Linear(10, 10)
        mock_fsdp.return_value = MagicMock()
        
        with warnings.catch_warnings(record=True) as w:
            wrapper.wrap_model(model, param_dtype=torch.bfloat16)
            
            # Should warn about bfloat16
            self.assertTrue(any("doesn't support bfloat16" in str(warning.message) for warning in w))
    
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    def test_wrap_transformer(self, mock_fsdp):
        """Test transformer model wrapping."""
        wrapper = MPSFSDPWrapper()
        
        # Create transformer model
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(512, 8)
                    for _ in range(4)
                ])
        
        model = SimpleTransformer()
        mock_fsdp.return_value = MagicMock()
        
        # Wrap with transformer policy
        wrapped = wrapper.wrap_transformer(model, nn.TransformerEncoderLayer)
        
        # Check auto wrap policy was used
        mock_fsdp.assert_called_once()
        call_kwargs = mock_fsdp.call_args[1]
        self.assertIsNotNone(call_kwargs['auto_wrap_policy'])
    
    @patch('torch.save')
    @patch('torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type')
    def test_save_checkpoint(self, mock_state_dict_type, mock_save):
        """Test checkpoint saving."""
        wrapper = MPSFSDPWrapper()
        wrapper.config.rank = 0  # Set as rank 0
        
        # Create mock FSDP model
        model = MagicMock()
        model.state_dict.return_value = {"param": torch.tensor([1.0])}
        
        # Save checkpoint
        wrapper.save_checkpoint(model, "test_checkpoint.pt", step=100)
        
        # Check save was called
        mock_save.assert_called_once()
        save_path, checkpoint = mock_save.call_args[0]
        self.assertEqual(save_path, "test_checkpoint.pt")
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("config", checkpoint)
        self.assertEqual(checkpoint["step"], 100)
    
    @patch('torch.load')
    @patch('torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type')
    def test_load_checkpoint(self, mock_state_dict_type, mock_load):
        """Test checkpoint loading."""
        wrapper = MPSFSDPWrapper()
        
        # Mock checkpoint
        mock_checkpoint = {
            "model_state_dict": {"param": torch.tensor([1.0])},
            "config": wrapper.config,
            "step": 100,
        }
        mock_load.return_value = mock_checkpoint
        
        # Create mock model
        model = MagicMock()
        
        # Load checkpoint
        checkpoint = wrapper.load_checkpoint(model, "test_checkpoint.pt")
        
        # Check load was called
        mock_load.assert_called_once_with("test_checkpoint.pt", map_location="cpu")
        model.load_state_dict.assert_called_once()
        self.assertEqual(checkpoint["step"], 100)
    
    def test_memory_stats(self):
        """Test memory statistics retrieval."""
        wrapper = MPSFSDPWrapper()
        
        with patch('torch.mps.current_allocated_memory', return_value=1e9):
            with patch('torch.mps.driver_allocated_memory', return_value=2e9):
                stats = wrapper.get_memory_stats()
                
                self.assertAlmostEqual(stats["allocated_gb"], 1.0, places=1)
                self.assertAlmostEqual(stats["reserved_gb"], 2.0, places=1)
    
    @patch('mps_fsdp_wrapper.logger')
    def test_profile_memory_context(self, mock_logger):
        """Test memory profiling context manager."""
        wrapper = MPSFSDPWrapper()
        wrapper.config.profile_memory = True
        
        with patch.object(wrapper, 'get_memory_stats') as mock_get_stats:
            mock_get_stats.side_effect = [
                {"allocated_gb": 1.0, "reserved_gb": 2.0},
                {"allocated_gb": 1.5, "reserved_gb": 2.5},
            ]
            
            with wrapper.profile_memory():
                pass
            
            # Should log memory delta
            mock_logger.info.assert_called()
            log_msg = mock_logger.info.call_args[0][0]
            self.assertIn("0.50GB", log_msg)  # Allocated delta


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_create_mps_fsdp_wrapper(self, mock_mps):
        """Test wrapper creation helper."""
        wrapper = create_mps_fsdp_wrapper(
            world_size=2,
            rank=1,
            sharding_strategy="SHARD_GRAD_OP",
            mixed_precision=False,
            cpu_offload=True,
        )
        
        self.assertIsInstance(wrapper, MPSFSDPWrapper)
        self.assertEqual(wrapper.config.world_size, 2)
        self.assertEqual(wrapper.config.rank, 1)
        self.assertEqual(wrapper.config.sharding_strategy, ShardingStrategy.SHARD_GRAD_OP)
        self.assertFalse(wrapper.config.use_mixed_precision)
        self.assertTrue(wrapper.config.cpu_offload)
    
    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    def test_wrap_model_for_mps(self, mock_fsdp, mock_mps):
        """Test quick model wrapping."""
        model = nn.Linear(10, 10)
        mock_fsdp.return_value = MagicMock()
        
        wrapped = wrap_model_for_mps(model, min_num_params=100)
        
        mock_fsdp.assert_called_once()
    
    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.backends.mps.is_built', return_value=True)
    @patch('torch.__version__', '2.7.0')
    def test_check_mps_fsdp_compatibility(self, mock_built, mock_available):
        """Test compatibility checking."""
        info = check_mps_fsdp_compatibility()
        
        self.assertTrue(info["mps_available"])
        self.assertTrue(info["mps_built"])
        self.assertEqual(info["pytorch_version"], "2.7.0")
        self.assertTrue(info["fsdp_available"])
        
        # Should not have warnings for 2.7
        self.assertEqual(len(info.get("warnings", [])), 0)
    
    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.__version__', '1.13.0')
    def test_check_compatibility_old_pytorch(self, mock_mps):
        """Test compatibility check with old PyTorch."""
        info = check_mps_fsdp_compatibility()
        
        # Should have warnings
        self.assertGreater(len(info["warnings"]), 0)
        self.assertIn("PyTorch 2.0+", info["warnings"][0])


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TransformerModel(nn.Module):
    """Transformer model for testing."""
    
    def __init__(self, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(1000, d_model)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, 1000)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        return self.output(x)


class TestIntegration(unittest.TestCase):
    """Integration tests for MPS FSDP wrapper."""
    
    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.distributed.fsdp.FullyShardedDataParallel')
    def test_end_to_end_wrapping(self, mock_fsdp, mock_mps):
        """Test end-to-end model wrapping and configuration."""
        # Create configuration
        config = MPSFSDPConfig(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_mixed_precision=True,
            min_num_params=100,
        )
        
        # Create wrapper
        wrapper = MPSFSDPWrapper(config)
        
        # Create model
        model = TransformerModel()
        
        # Mock FSDP
        mock_wrapped_model = MagicMock()
        mock_fsdp.return_value = mock_wrapped_model
        
        # Wrap model
        wrapped_model = wrapper.wrap_transformer(
            model,
            transformer_layer_cls=nn.TransformerEncoderLayer,
        )
        
        # Verify wrapping
        mock_fsdp.assert_called_once()
        call_args = mock_fsdp.call_args
        
        # Check model was passed
        self.assertIsInstance(call_args[0][0], TransformerModel)
        
        # Check configuration
        call_kwargs = call_args[1]
        self.assertEqual(call_kwargs['sharding_strategy'], ShardingStrategy.FULL_SHARD)
        self.assertIsNotNone(call_kwargs['mixed_precision'])
        self.assertEqual(call_kwargs['device_id'], torch.device('mps'))
    
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_memory_optimization_flow(self, mock_mps):
        """Test memory optimization workflow."""
        # Create config with memory optimization
        config = MPSFSDPConfig(
            unified_memory_pool_size=16 * 1024 * 1024 * 1024,  # 16GB
            aggressive_memory_optimization=True,
        )
        
        wrapper = MPSFSDPWrapper(config)
        
        # Test sharding strategy optimization
        model_size = 7 * 1024 * 1024 * 1024  # 7GB model
        available_memory = 16 * 1024 * 1024 * 1024  # 16GB available
        
        optimal_strategy = wrapper.memory_optimizer.optimize_sharding_strategy(
            model_size, available_memory
        )
        
        # Should recommend gradient/optimizer sharding for 7GB/16GB
        self.assertEqual(optimal_strategy, ShardingStrategy.SHARD_GRAD_OP)
        
        # Test bucket size
        bucket_size = wrapper.memory_optimizer.get_optimal_bucket_size(7e9)
        self.assertEqual(bucket_size, 200_000_000)  # Large model gets large buckets
    
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_dtype_handling_integration(self, mock_mps):
        """Test dtype handling in integration."""
        wrapper = MPSFSDPWrapper()
        
        # Create model with mixed dtypes
        model = SimpleModel()
        model.fc1 = model.fc1.to(torch.bfloat16)
        model.fc2 = model.fc2.to(torch.float32)
        
        # Convert dtypes
        converted = wrapper._convert_dtype(model, torch.float16)
        
        # Check conversions
        self.assertEqual(converted.fc1.weight.dtype, torch.float16)  # bfloat16 -> float16
        self.assertEqual(converted.fc2.weight.dtype, torch.float32)  # float32 unchanged


if __name__ == '__main__':
    unittest.main()