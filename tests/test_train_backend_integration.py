#!/usr/bin/env python3
"""
Tests for backend integration in train.py
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import subprocess
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend


class TestTrainBackendCLI(unittest.TestCase):
    """Test CLI argument parsing for backend selection."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        # Save original sys.argv
        self.original_argv = sys.argv
        # Set minimal sys.argv to avoid pytest args interference
        sys.argv = ['train.py']
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        # Restore original sys.argv
        sys.argv = self.original_argv
    
    def test_help_shows_backend_option(self):
        """Test that --help shows the backend option."""
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            capture_output=True,
            text=True
        )
        self.assertIn("--backend", result.stdout)
        self.assertIn("Backend to use for training", result.stdout)
        self.assertIn("cuda", result.stdout)
        self.assertIn("mps", result.stdout)
        self.assertIn("mlx", result.stdout)
        self.assertIn("auto", result.stdout)
    
    @patch('train.mp.spawn')
    @patch('train.fsdp_main')
    @patch('train.BackendManager')
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.is_bf16_supported', return_value=True)
    def test_default_backend_cuda(self, mock_bf16, mock_device_count, 
                                 mock_backend_manager_class, mock_fsdp_main, mock_spawn):
        """Test that default backend is CUDA."""
        # Mock backend manager
        mock_backend_manager = MagicMock()
        mock_backend_manager.backend = Backend.CUDA
        mock_backend_manager.capabilities.supports_distributed = True
        mock_backend_manager_class.return_value = mock_backend_manager
        
        # Import here to trigger the decorators
        import train
        
        # Call main with minimal args
        train.main(
            num_epochs=1,
            max_steps=1,
            dataset="dummy",
            dataset_samples=16,
            save_model=False,
            verbose=False
        )
        
        # Verify backend manager was created with cuda as default
        mock_backend_manager_class.assert_called()
        call_args = mock_backend_manager_class.call_args
        self.assertEqual(call_args[1]['backend'], 'cuda')
    
    @patch('train.fsdp_main')
    @patch('train.BackendManager')
    def test_backend_auto_detection(self, mock_backend_manager_class, mock_fsdp_main):
        """Test auto backend detection."""
        # Mock backend manager
        mock_backend_manager = MagicMock()
        mock_backend_manager.backend = Backend.MPS
        mock_backend_manager.capabilities.supports_distributed = True
        mock_backend_manager_class.return_value = mock_backend_manager
        
        import train
        
        # Call fsdp_qlora directly to avoid @call_parse issues
        train.fsdp_qlora(
            backend="auto",
            num_epochs=1,
            max_steps=1,
            dataset="dummy",
            dataset_samples=16,
            save_model=False,
            verbose=False,
            world_size=1
        )
        
        # Verify backend manager was created with None (for auto-detection)
        mock_backend_manager_class.assert_called()
        call_args = mock_backend_manager_class.call_args
        self.assertIsNone(call_args[1]['backend'])
    
    @patch('train.fsdp_main')
    @patch('train.BackendManager')
    def test_backend_validation_error(self, mock_backend_manager_class, mock_fsdp_main):
        """Test that invalid backend raises error."""
        # Mock backend manager to simulate backend not available
        mock_backend_manager = MagicMock()
        mock_backend_manager.backend = Backend.CPU
        mock_backend_manager_class.return_value = mock_backend_manager
        
        import train
        
        # Try to use CUDA when it's not available
        with self.assertRaises(ValueError) as cm:
            train.main(
                backend="cuda",
                num_epochs=1,
                max_steps=1,
                dataset="dummy",
                dataset_samples=16,
                save_model=False,
                verbose=False
            )
        
        self.assertIn("not available", str(cm.exception))


class TestTrainBackendConfiguration(unittest.TestCase):
    """Test backend-specific configuration adjustments."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original sys.argv
        self.original_argv = sys.argv
        # Set minimal sys.argv to avoid pytest args interference
        sys.argv = ['train.py']
        
    def tearDown(self):
        """Clean up test environment."""
        # Restore original sys.argv
        sys.argv = self.original_argv
    
    @patch('train.fsdp_main')
    @patch('train.BackendManager')
    def test_mps_bfloat16_warning(self, mock_backend_manager_class, mock_fsdp_main):
        """Test that MPS backend warns about bfloat16 and switches to fp16."""
        # Mock backend manager for MPS
        mock_backend_manager = MagicMock()
        mock_backend_manager.backend = Backend.MPS
        mock_backend_manager.capabilities.supports_distributed = True
        mock_backend_manager.capabilities.supports_bfloat16 = False
        mock_backend_manager_class.return_value = mock_backend_manager
        
        import train
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            train.fsdp_qlora(
                backend="mps",
                precision="bf16",
                num_epochs=1,
                max_steps=1,
                dataset="dummy",
                dataset_samples=16,
                save_model=False,
                verbose=False,
                world_size=1
            )
            
            # Check warning was printed
            warning_calls = [call for call in mock_print.call_args_list 
                           if "MPS backend doesn't support bfloat16" in str(call)]
            self.assertTrue(len(warning_calls) > 0)
        
        # Verify precision was changed in args passed to fsdp_main
        fsdp_args = mock_fsdp_main.call_args[0][2]
        self.assertNotEqual(fsdp_args['precision'], 'bf16')
    
    @patch('train.fsdp_main')
    @patch('train.BackendManager')  
    def test_mlx_distributed_error(self, mock_backend_manager_class, mock_fsdp_main):
        """Test that MLX backend raises error with world_size > 1."""
        # Mock backend manager for MLX
        mock_backend_manager = MagicMock()
        mock_backend_manager.backend = Backend.MLX
        mock_backend_manager.capabilities.supports_distributed = False
        mock_backend_manager_class.return_value = mock_backend_manager
        
        import train
        
        # Try to use MLX with multiple GPUs
        with self.assertRaises(ValueError) as cm:
            train.fsdp_qlora(
                backend="mlx",
                world_size=2,
                num_epochs=1,
                max_steps=1,
                dataset="dummy",
                dataset_samples=16,
                save_model=False,
                verbose=False
            )
        
        self.assertIn("MLX backend doesn't support distributed training", str(cm.exception))
    
    @patch('train.fsdp_main')
    @patch('train.BackendManager')
    def test_cpu_backend_warning(self, mock_backend_manager_class, mock_fsdp_main):
        """Test that CPU backend shows warning."""
        # Mock backend manager for CPU
        mock_backend_manager = MagicMock()
        mock_backend_manager.backend = Backend.CPU
        mock_backend_manager.capabilities.supports_distributed = True
        mock_backend_manager_class.return_value = mock_backend_manager
        
        import train
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            train.fsdp_qlora(
                backend="cpu",
                num_epochs=1,
                max_steps=1,
                dataset="dummy",
                dataset_samples=16,
                save_model=False,
                verbose=False,
                world_size=1
            )
            
            # Check warning was printed
            warning_calls = [call for call in mock_print.call_args_list 
                           if "CPU backend selected" in str(call)]
            self.assertTrue(len(warning_calls) > 0)


class TestTrainBackendIntegration(unittest.TestCase):
    """Test integration of backend manager throughout train.py."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original sys.argv
        self.original_argv = sys.argv
        # Set minimal sys.argv to avoid pytest args interference
        sys.argv = ['train.py']
        
    def tearDown(self):
        """Clean up test environment."""
        # Restore original sys.argv
        sys.argv = self.original_argv
    
    @patch('train.mp.spawn')
    @patch('train.BackendManager')
    def test_cuda_uses_mp_spawn(self, mock_backend_manager_class, mock_spawn):
        """Test that CUDA backend uses mp.spawn."""
        # Mock backend manager for CUDA
        mock_backend_manager = MagicMock()
        mock_backend_manager.backend = Backend.CUDA
        mock_backend_manager.capabilities.supports_distributed = True
        mock_backend_manager_class.return_value = mock_backend_manager
        
        # Mock CUDA device count and bf16 support
        with patch('torch.cuda.device_count', return_value=2):
            with patch('torch.cuda.is_bf16_supported', return_value=True):
                import train
                
                train.fsdp_qlora(
                    backend="cuda",
                    num_epochs=1,
                    max_steps=1,
                    dataset="dummy",
                    dataset_samples=16,
                    save_model=False,
                    verbose=False,
                    world_size=-1  # Let it auto-detect from device count
                )
            
            # Verify mp.spawn was called
            mock_spawn.assert_called_once()
            self.assertEqual(mock_spawn.call_args[1]['nprocs'], 2)
    
    @patch('train.fsdp_main')
    @patch('train.BackendManager')
    def test_non_cuda_direct_call(self, mock_backend_manager_class, mock_fsdp_main):
        """Test that non-CUDA backends call fsdp_main directly."""
        # Mock backend manager for MPS
        mock_backend_manager = MagicMock()
        mock_backend_manager.backend = Backend.MPS
        mock_backend_manager.capabilities.supports_distributed = True
        mock_backend_manager_class.return_value = mock_backend_manager
        
        import train
        
        train.fsdp_qlora(
            backend="mps",
            num_epochs=1,
            max_steps=1,
            dataset="dummy",
            dataset_samples=16,
            save_model=False,
            verbose=False,
            world_size=1
        )
        
        # Verify fsdp_main was called directly (not through mp.spawn)
        mock_fsdp_main.assert_called_once()
        # First arg should be 0 (local_rank)
        self.assertEqual(mock_fsdp_main.call_args[0][0], 0)
    
    @patch('train.fsdp_main')
    @patch('train.BackendManager')
    def test_backend_manager_passed_to_fsdp_main(self, mock_backend_manager_class, mock_fsdp_main):
        """Test that backend manager is passed to fsdp_main in args."""
        # Mock backend manager
        mock_backend_manager = MagicMock()
        mock_backend_manager.backend = Backend.MPS
        mock_backend_manager.capabilities.supports_distributed = True
        mock_backend_manager_class.return_value = mock_backend_manager
        
        import train
        
        train.fsdp_qlora(
            backend="mps",
            num_epochs=1,
            max_steps=1,
            dataset="dummy",
            dataset_samples=16,
            save_model=False,
            verbose=False,
            world_size=1
        )
        
        # Verify backend_manager is in args
        fsdp_args = mock_fsdp_main.call_args[0][2]
        self.assertIn('backend_manager', fsdp_args)
        self.assertEqual(fsdp_args['backend_manager'], mock_backend_manager)


if __name__ == "__main__":
    unittest.main()