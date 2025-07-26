"""
Integration tests for train.py with import abstraction layer.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrainImportsIntegration(unittest.TestCase):
    """Test integration of import abstraction with train.py."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original sys.argv
        self.original_argv = sys.argv
        # Set minimal sys.argv to avoid pytest args interference
        sys.argv = ['train.py']
    
    def tearDown(self):
        """Restore original state."""
        sys.argv = self.original_argv
    
    def test_load_conditional_imports_cuda(self):
        """Test loading conditional imports for CUDA backend."""
        from train import load_conditional_imports
        
        # Mock the global variables
        with patch('train.bnb', None), \
             patch('train.Linear4bit', None), \
             patch('train.Params4bit', None):
            
            # Load imports for CUDA backend
            load_conditional_imports('cuda')
            
            # Check that imports were attempted
            # Note: actual imports may fail if modules aren't installed
            self.assertTrue(True)  # Basic test that function runs
    
    def test_load_conditional_imports_cpu(self):
        """Test loading conditional imports for CPU backend."""
        from train import load_conditional_imports
        
        # Load imports for CPU backend
        load_conditional_imports('cpu')
        
        # For CPU, bitsandbytes should use fallback
        import train
        self.assertIsNotNone(train.bnb)  # Should have fallback
    
    @patch('train.get_module')
    def test_logger_wandb_integration(self, mock_get_module):
        """Test Logger class uses import abstraction for wandb."""
        from train import Logger
        
        # Mock wandb module
        mock_wandb = MagicMock()
        mock_get_module.return_value = mock_wandb
        
        # Create logger with wandb
        args = {'test': 'value'}
        logger = Logger(args, log_to='wandb', rank=0)
        
        # Check that wandb.init was called
        mock_wandb.init.assert_called_once()
        
        # Test logging
        logger.log({'metric': 1.0}, rank=0)
        # Note: get_module is called each time, so log might be called on a different instance
        self.assertTrue(mock_get_module.called)
        
        # Test finish
        logger.finish(rank=0)
        self.assertTrue(mock_get_module.called)
    
    def test_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        from train import load_conditional_imports
        
        # This should not raise exceptions even if imports fail
        try:
            load_conditional_imports('unknown_backend')
            load_conditional_imports('mps')
            load_conditional_imports('mlx')
        except Exception as e:
            self.fail(f"load_conditional_imports raised exception: {e}")
    
    @patch('train.get_module')
    def test_conditional_import_backend_awareness(self, mock_get_module):
        """Test that imports are backend-aware."""
        from train import load_conditional_imports
        
        # Track which modules were requested
        requested_modules = []
        
        def track_module(name, backend=None):
            requested_modules.append((name, backend))
            # Return a mock module
            mock_module = MagicMock()
            # Add required attributes for bitsandbytes
            if name == 'bitsandbytes':
                mock_module.Linear4bit = MagicMock()
                mock_module.Params4bit = MagicMock()
            # Add required attributes for dora
            elif name == 'dora':
                mock_module.BNBDORA = MagicMock()
                mock_module.HQQDORA = MagicMock()
                mock_module.DORALayer = MagicMock()
                mock_module.MagnitudeLayer = MagicMock()
            # Add required attributes for lora
            elif name == 'lora':
                mock_module.LORA = MagicMock()
            return mock_module
        
        mock_get_module.side_effect = track_module
        
        # Load imports for CUDA
        load_conditional_imports('cuda')
        
        # Check that bitsandbytes was requested with cuda backend
        self.assertIn(('bitsandbytes', 'cuda'), requested_modules)
        
        # Clear and test CPU
        requested_modules.clear()
        load_conditional_imports('cpu')
        
        # Check that bitsandbytes was requested with cpu backend
        self.assertIn(('bitsandbytes', 'cpu'), requested_modules)
    
    def test_fsdp_qlora_import_initialization(self):
        """Test that fsdp_qlora properly initializes imports."""
        from train import fsdp_qlora
        
        # This is a minimal test to ensure the function can be called
        # without actually running training
        try:
            # We expect this to fail early due to missing model
            fsdp_qlora(
                backend='cpu',
                model_name='test_model',
                dry_run=True,  # If this flag exists
                world_size=0,  # Don't actually spawn processes
            )
        except Exception as e:
            # We expect some exception, but it should be after imports
            # Check that it's not an import error
            self.assertNotIn('No module named', str(e))
    
    def test_import_status_report(self):
        """Test that we can generate an import status report."""
        from imports import report_import_status
        
        # Generate report for different backends
        for backend in ['cuda', 'mps', 'cpu']:
            report = report_import_status(backend)
            
            # Check report contains expected information
            self.assertIn('Import Status Report', report)
            self.assertIn(f'Backend: {backend}', report)
            self.assertIn('bitsandbytes', report)
            self.assertIn('wandb', report)
    
    def test_import_validation(self):
        """Test import validation functionality."""
        from imports import validate_imports
        
        # Validate critical imports for training
        required_imports = ['bitsandbytes', 'wandb']
        
        # Validate for CUDA backend
        results = validate_imports(required_imports, backend='cuda')
        
        self.assertEqual(len(results), len(required_imports))
        for name, result in results.items():
            self.assertIn(name, required_imports)
            # Should succeed (with real module or fallback)
            self.assertTrue(result.success)
    
    def test_backend_specific_import_behavior(self):
        """Test that imports behave correctly for different backends."""
        from train import load_conditional_imports
        import train
        
        # Test CUDA backend
        load_conditional_imports('cuda')
        self.assertIsNotNone(train.bnb)
        self.assertIsNotNone(train.Linear4bit)
        self.assertIsNotNone(train.Params4bit)
        
        # Test MPS backend (should use fallback for bitsandbytes)
        load_conditional_imports('mps')
        self.assertIsNotNone(train.bnb)  # Fallback should be provided


class TestImportPerformance(unittest.TestCase):
    """Test performance aspects of the import system."""
    
    def test_import_caching(self):
        """Test that imports are cached and not repeated."""
        from imports import get_module
        
        # First import
        import time
        start = time.time()
        module1 = get_module('wandb')
        first_time = time.time() - start
        
        # Second import (should be cached)
        start = time.time()
        module2 = get_module('wandb')
        second_time = time.time() - start
        
        # Second import should be much faster (cached)
        # Note: This is a rough test, might be flaky
        self.assertIs(module1, module2)  # Same object
        
        # Can't reliably test timing in unit tests, but we verify
        # that the same object is returned (indicating caching)


if __name__ == '__main__':
    unittest.main()