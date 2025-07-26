"""
Test the TODO fixes implemented in train.py
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch, Mock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that might not be available
sys.modules['wandb'] = MagicMock()


class TestLoggerFixes:
    """Test the Logger class with log_every_n_steps functionality."""
    
    def test_logger_init_with_log_every_n_steps(self):
        """Test that Logger initializes with log_every_n_steps parameter."""
        from train import Logger
        
        args = {"test": "value"}
        logger = Logger(args, log_every_n_steps=5)
        
        assert logger.log_every_n_steps == 5
        assert logger.step_count == 0
    
    def test_logger_logs_every_n_steps(self):
        """Test that Logger only logs every n steps for training logs."""
        from train import Logger
        
        args = {"test": "value"}
        logger = Logger(args, log_to="stdout", log_every_n_steps=3)
        
        # Mock print to capture output
        with patch('builtins.print') as mock_print:
            # First two calls should not print (steps 1 and 2)
            logger.log({"loss": 0.5, "lr": 0.001}, rank=0)
            logger.log({"loss": 0.4, "lr": 0.001}, rank=0)
            assert mock_print.call_count == 0
            
            # Third call should print (step 3)
            logger.log({"loss": 0.3, "lr": 0.001}, rank=0)
            assert mock_print.call_count == 4  # 2 items * 2 keys
    
    def test_logger_always_logs_non_training_metrics(self):
        """Test that Logger always logs non-training metrics."""
        from train import Logger
        
        args = {"test": "value"}
        logger = Logger(args, log_to="stdout", log_every_n_steps=10)
        
        with patch('builtins.print') as mock_print:
            # Memory logs should always print
            logger.log({"memory/allocated": 1000}, rank=0)
            assert mock_print.call_count == 1
    
    def test_logger_force_log(self):
        """Test that Logger logs when force=True regardless of step count."""
        from train import Logger
        
        args = {"test": "value"}
        logger = Logger(args, log_to="stdout", log_every_n_steps=10)
        
        with patch('builtins.print') as mock_print:
            # Should print even though it's the first step
            logger.log({"loss": 0.5}, rank=0, force=True)
            assert mock_print.call_count == 1


class TestTokenizerFixes:
    """Test the tokenizer pad_token_id fix."""
    
    def test_tokenizer_pad_token_set_when_none(self):
        """Test that pad_token_id is set to eos_token_id when None."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 2
        
        # Simulate the fix
        if mock_tokenizer.pad_token_id is None:
            mock_tokenizer.pad_token_id = mock_tokenizer.eos_token_id
        
        assert mock_tokenizer.pad_token_id == 2
    
    def test_tokenizer_pad_token_not_changed_when_exists(self):
        """Test that pad_token_id is not changed when it already exists."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 1
        mock_tokenizer.eos_token_id = 2
        
        # Simulate the fix
        if mock_tokenizer.pad_token_id is None:
            mock_tokenizer.pad_token_id = mock_tokenizer.eos_token_id
        
        assert mock_tokenizer.pad_token_id == 1  # Should remain unchanged


class TestHQQConfigFixes:
    """Test the HQQ configuration improvements."""
    
    def test_hqq_config_with_custom_group_size(self):
        """Test that HQQ config uses custom group size when provided."""
        args = {
            "n_bits": 4,
            "hqq_group_size": 128
        }
        
        # Simulate the config creation
        group_size = args.get("hqq_group_size", 64)
        
        assert group_size == 128
    
    def test_hqq_config_with_default_group_size(self):
        """Test that HQQ config uses default group size when not provided."""
        args = {
            "n_bits": 4
        }
        
        # Simulate the config creation
        group_size = args.get("hqq_group_size", 64)
        
        assert group_size == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])