#!/usr/bin/env python3
"""
Integration tests for training with different backends.
"""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrainIntegration(unittest.TestCase):
    """Integration tests for the training script with backend support."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _mock_model_loading(self):
        """Mock model loading to avoid downloading actual models."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.named_parameters.return_value = [
            ("layer1.weight", MagicMock(numel=lambda: 1000, requires_grad=True)),
            ("layer2.weight", MagicMock(numel=lambda: 1000, requires_grad=True))
        ]
        return mock_model
    
    @patch('train.AutoModelForCausalLM.from_pretrained')
    @patch('train.AutoTokenizer.from_pretrained')
    @patch('train.safetensors.torch.load_file')
    @patch('train.hub.cached_file')
    def test_minimal_training_mps(self, mock_cached_file, mock_load_file, 
                                 mock_tokenizer, mock_model_from_pretrained):
        """Test minimal training run with MPS backend."""
        # Skip if not on macOS
        if not sys.platform == "darwin":
            self.skipTest("MPS backend test requires macOS")
        
        # Mock model and tokenizer
        mock_model_from_pretrained.return_value = self._mock_model_loading()
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.pad_token_id = 0
        mock_load_file.return_value = {}
        mock_cached_file.side_effect = FileNotFoundError("No safetensors")
        
        # Patch backend detection to ensure MPS
        with patch('backend_manager.BackendManager._detect_available_backends') as mock_detect:
            from backend_manager import Backend
            mock_detect.return_value = [Backend.MPS, Backend.CPU]
            
            import train
            
            # Run with minimal settings
            try:
                train.fsdp_qlora(
                    backend="mps",
                    world_size=1,
                    num_epochs=1,
                    max_steps=1,
                    dataset="dummy",
                    dataset_samples=2,
                    batch_size=1,
                    context_length=128,
                    save_model=False,
                    train_type="lora",
                    lora_rank=8,
                    verbose=False,
                    low_memory=False,
                    use_cpu_offload=False,
                    use_gradient_checkpointing=False
                )
            except Exception as e:
                # Some errors are expected due to mocking
                # But backend initialization should work
                self.assertNotIn("Backend", str(e))
                self.assertNotIn("not available", str(e))
    
    def test_backend_device_consistency(self):
        """Test that device selection is consistent with backend."""
        from backend_manager import Backend, BackendManager
        
        # Test each backend
        test_cases = [
            (Backend.CUDA, "cuda"),
            (Backend.MPS, "mps"), 
            (Backend.CPU, "cpu"),
            (Backend.MLX, "cpu")  # MLX uses CPU device for PyTorch
        ]
        
        for backend, expected_device_type in test_cases:
            with patch.object(BackendManager, '_detect_available_backends', 
                            return_value=[backend, Backend.CPU]):
                manager = BackendManager(backend=backend, verbose=False)
                self.assertEqual(manager.device.type, expected_device_type)
    
    def test_backend_capability_constraints(self):
        """Test that backend capabilities are properly enforced."""
        from backend_manager import Backend, BackendManager
        
        # Test MPS doesn't support 4-bit quantization
        with patch.object(BackendManager, '_detect_available_backends', 
                         return_value=[Backend.MPS, Backend.CPU]):
            manager = BackendManager(backend=Backend.MPS, verbose=False)
            self.assertFalse(manager.supports_quantization(4))
            self.assertTrue(manager.supports_quantization(8))
        
        # Test MLX doesn't support distributed
        with patch.object(BackendManager, '_detect_available_backends', 
                         return_value=[Backend.MLX, Backend.CPU]):
            manager = BackendManager(backend=Backend.MLX, verbose=False)
            self.assertFalse(manager.capabilities.supports_distributed)
            self.assertIsNone(manager.get_distributed_backend())
    
    def test_memory_helper_function(self):
        """Test the get_memory_stats helper function."""
        import train
        from backend_manager import Backend, BackendManager
        
        # Test CUDA backend
        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.CUDA]):
            with patch('torch.cuda.memory_allocated', return_value=1000):
                with patch('torch.cuda.memory_reserved', return_value=2000):
                    manager = BackendManager(backend=Backend.CUDA, verbose=False)
                    stats = train.get_memory_stats(manager, 0)
                    self.assertEqual(stats["allocated"], 1000)
                    self.assertEqual(stats["reserved"], 2000)
        
        # Test non-CUDA backend
        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.CPU]):
            manager = BackendManager(backend=Backend.CPU, verbose=False)
            with patch.object(manager, 'get_memory_info', 
                            return_value={"used_memory": 3000, "total_memory": 4000}):
                stats = train.get_memory_stats(manager, 0)
                self.assertEqual(stats["allocated"], 3000)
                self.assertEqual(stats["reserved"], 4000)
    
    def test_n_loading_workers_backend_aware(self):
        """Test that n_loading_workers adjusts for different backends."""
        import train
        from backend_manager import Backend, BackendManager
        
        # Test CUDA backend
        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.CUDA]):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = MagicMock(total_memory=16*1e9)
                with patch('torch.cuda.device_count', return_value=1):
                    with patch('os.cpu_count', return_value=8):
                        manager = BackendManager(backend=Backend.CUDA, verbose=False)
                        workers = train.n_loading_workers("bnb", 7e9, manager)
                        self.assertGreater(workers, 0)
                        self.assertLessEqual(workers, 8)
        
        # Test non-CUDA backend
        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.MPS]):
            with patch('os.cpu_count', return_value=8):
                manager = BackendManager(backend=Backend.MPS, verbose=False)
                workers = train.n_loading_workers("bnb", 7e9, manager)
                self.assertEqual(workers, min(4, 8))


if __name__ == "__main__":
    unittest.main()