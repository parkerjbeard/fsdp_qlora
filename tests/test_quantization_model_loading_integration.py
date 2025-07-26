"""
Integration tests for quantization model loading functionality.

These tests verify end-to-end model loading with actual file operations
and integration between components.
"""

import json
import os
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.core.quantization_wrapper import (
    create_quantization_adapter,
    QuantizationConfig,
    QuantizationMethod,
    Backend
)
from src.core.backend_manager import BackendManager


class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size * 2, hidden_size)
        self.output = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.output(x)
        return x


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def model_config():
    """Create a model configuration."""
    return {
        'model_type': 'simple',
        'hidden_size': 32,
        'vocab_size': 1000,
        'num_classes': 10
    }


class TestBitsAndBytesIntegration:
    """Integration tests for BitsAndBytes model loading."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_and_load_bnb_4bit(self, temp_model_dir, simple_model, model_config):
        """Test saving and loading a 4-bit BNB quantized model."""
        # Create adapter
        config = QuantizationConfig(
            method=QuantizationMethod.BNB_NF4,
            bits=4,
            compute_dtype=torch.float16
        )
        adapter = create_quantization_adapter(Backend.CUDA, config)
        
        # Save model
        model_path = temp_model_dir / 'bnb_model'
        model_path.mkdir()
        
        # Save state dict
        torch.save(simple_model.state_dict(), model_path / 'pytorch_model.bin')
        
        # Save quantization config
        quant_config = {
            'bits': 4,
            'quant_type': 'nf4',
            'compute_dtype': 'float16'
        }
        with open(model_path / 'quantization_config.json', 'w') as f:
            json.dump(quant_config, f)
            
        # Save model config
        with open(model_path / 'config.json', 'w') as f:
            json.dump(model_config, f)
        
        # Test loading - this will fail without actual BNB but tests the flow
        try:
            loaded_model = adapter.load_quantized_model(str(model_path), model_config)
            assert loaded_model is not None
        except ImportError as e:
            # Expected if bitsandbytes not installed
            assert "bitsandbytes" in str(e) or "transformers" in str(e)
    
    def test_load_bnb_single_file(self, temp_model_dir, simple_model, model_config):
        """Test loading BNB model from single file."""
        config = QuantizationConfig(method=QuantizationMethod.BNB_INT8, bits=8)
        adapter = create_quantization_adapter(Backend.CUDA, config)
        
        # Save as single file
        model_file = temp_model_dir / 'model.pth'
        torch.save(simple_model.state_dict(), model_file)
        
        # Test loading with fallback implementation
        try:
            loaded_model = adapter.load_quantized_model(str(model_file), model_config)
            # Will use fallback implementation
            assert loaded_model is not None
        except ImportError:
            # Expected if dependencies not available
            pass


class TestHQQIntegration:
    """Integration tests for HQQ model loading."""
    
    def test_save_and_load_hqq(self, temp_model_dir, simple_model, model_config):
        """Test saving and loading an HQQ quantized model."""
        # Create adapter
        config = QuantizationConfig(
            method=QuantizationMethod.HQQ,
            bits=4,
            group_size=64,
            quant_zero=True
        )
        adapter = create_quantization_adapter(Backend.CUDA, config)
        
        # Save model directory structure
        model_path = temp_model_dir / 'hqq_model'
        model_path.mkdir()
        
        # Create HQQ-style state dict
        hqq_state_dict = {}
        for name, param in simple_model.state_dict().items():
            if 'weight' in name and len(param.shape) == 2:
                # Simulate HQQ quantized weights
                hqq_state_dict[f"{name.replace('.weight', '')}.W_q"] = param
                hqq_state_dict[f"{name.replace('.weight', '')}.meta"] = {
                    'scale': 0.1,
                    'zero': 0.0,
                    'group_size': 64
                }
            else:
                hqq_state_dict[name] = param
        
        # Save state dict
        torch.save(hqq_state_dict, model_path / 'pytorch_model.bin')
        
        # Save HQQ config
        hqq_config = {
            'weight_quant_params': {
                'nbits': 4,
                'group_size': 64,
                'quant_zero': True,
                'quant_scale': False
            }
        }
        with open(model_path / 'hqq_config.json', 'w') as f:
            json.dump(hqq_config, f)
        
        # Test loading
        try:
            loaded_model = adapter.load_quantized_model(str(model_path), model_config)
            assert loaded_model is not None
        except ImportError as e:
            # Expected if HQQ not installed
            assert "HQQ" in str(e) or "hqq" in str(e)


class TestMLXIntegration:
    """Integration tests for MLX model loading."""
    
    @pytest.mark.skipif(not (torch.backends.mps.is_available() or os.path.exists('/usr/bin/swift')), 
                       reason="Not on Apple Silicon")
    def test_save_and_load_mlx_npz(self, temp_model_dir, simple_model, model_config):
        """Test saving and loading MLX model in NPZ format."""
        # Create adapter
        config = QuantizationConfig(
            method=QuantizationMethod.MLX_INT4,
            bits=4,
            group_size=32
        )
        backend = Backend.MLX if os.path.exists('/usr/bin/swift') else Backend.MPS
        adapter = create_quantization_adapter(backend, config)
        
        # Simulate MLX weights (would normally use mx.save)
        import numpy as np
        mlx_weights = {}
        for name, param in simple_model.state_dict().items():
            mlx_weights[name] = param.detach().cpu().numpy()
        
        # Save as NPZ
        model_file = temp_model_dir / 'model.npz'
        np.savez(model_file, **mlx_weights)
        
        # Save config
        config_file = temp_model_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(model_config, f)
        
        # Test loading
        try:
            loaded_model = adapter.load_quantized_model(str(model_file), model_config)
            assert loaded_model is not None
            assert hasattr(loaded_model, 'mlx_model')
        except ImportError as e:
            # Expected if MLX not available
            assert "MLX" in str(e) or "mlx" in str(e)
    
    def test_convert_pytorch_to_mlx(self, temp_model_dir, simple_model, model_config):
        """Test converting PyTorch model to MLX format."""
        config = QuantizationConfig(method=QuantizationMethod.MLX_INT8, bits=8)
        adapter = create_quantization_adapter(Backend.MPS, config)
        
        # Save PyTorch model
        model_file = temp_model_dir / 'pytorch_model.bin'
        torch.save(simple_model.state_dict(), model_file)
        
        # Test conversion on load
        try:
            loaded_model = adapter.load_quantized_model(str(model_file), model_config)
            assert loaded_model is not None
        except (ImportError, ValueError):
            # Expected if MLX not available or on non-Apple platform
            pass


class TestQuantoIntegration:
    """Integration tests for Quanto model loading."""
    
    def test_save_and_load_quanto(self, temp_model_dir, simple_model, model_config):
        """Test saving and loading a Quanto quantized model."""
        # Create adapter
        config = QuantizationConfig(
            method=QuantizationMethod.QUANTO_INT4,
            bits=4,
            group_size=128
        )
        adapter = create_quantization_adapter(Backend.MPS, config)
        
        # Save model directory
        model_path = temp_model_dir / 'quanto_model'
        model_path.mkdir()
        
        # Save state dict
        torch.save(simple_model.state_dict(), model_path / 'pytorch_model.bin')
        
        # Save Quanto config
        quanto_config = {
            'quantization_config': {
                'quant_method': 'quanto',
                'bits': 4,
                'group_size': 128
            }
        }
        with open(model_path / 'config.json', 'w') as f:
            json.dump(quanto_config, f)
        
        # Test loading
        try:
            loaded_model = adapter.load_quantized_model(str(model_path), model_config)
            assert loaded_model is not None
        except ImportError as e:
            # Expected if Quanto not installed
            assert "quanto" in str(e).lower()
    
    def test_load_quanto_single_file_with_config(self, temp_model_dir, simple_model, model_config):
        """Test loading Quanto model from single file with embedded config."""
        config = QuantizationConfig(method=QuantizationMethod.QUANTO_INT2, bits=2)
        adapter = create_quantization_adapter(Backend.CPU, config)
        
        # Save with embedded config
        model_file = temp_model_dir / 'quanto_model.pth'
        checkpoint = {
            'state_dict': simple_model.state_dict(),
            'quanto_config': {
                'bits': 2,
                'group_size': 64
            }
        }
        torch.save(checkpoint, model_file)
        
        # Test loading
        try:
            loaded_model = adapter.load_quantized_model(str(model_file), model_config)
            assert loaded_model is not None
        except ImportError:
            # Expected if Quanto not installed
            pass


class TestCrossBackendLoading:
    """Test loading models across different backends."""
    
    def test_backend_compatibility_matrix(self, temp_model_dir, simple_model, model_config):
        """Test which quantization methods work with which backends."""
        # Save a simple model
        model_file = temp_model_dir / 'model.pth'
        torch.save(simple_model.state_dict(), model_file)
        
        # Test matrix of backends and methods
        test_cases = [
            (Backend.CUDA, QuantizationMethod.BNB_NF4, True),
            (Backend.CUDA, QuantizationMethod.HQQ, True),
            (Backend.MPS, QuantizationMethod.MLX_INT4, True),
            (Backend.MPS, QuantizationMethod.QUANTO_INT4, True),
            (Backend.CPU, QuantizationMethod.QUANTO_INT8, True),
            (Backend.CPU, QuantizationMethod.BNB_NF4, False),  # Should fallback
        ]
        
        for backend, method, should_work in test_cases:
            config = QuantizationConfig(method=method)
            adapter = create_quantization_adapter(backend, config)
            
            # Check adapter type
            if should_work:
                assert adapter.__class__.__name__ != 'FallbackAdapter'
            else:
                assert adapter.__class__.__name__ == 'FallbackAdapter'


class TestErrorHandling:
    """Test error handling in model loading."""
    
    def test_file_not_found_error(self):
        """Test handling of missing model files."""
        config = QuantizationConfig(method=QuantizationMethod.BNB_NF4)
        adapter = create_quantization_adapter(Backend.CUDA, config)
        
        with pytest.raises(FileNotFoundError):
            adapter.load_quantized_model('/nonexistent/path', {})
    
    def test_corrupted_state_dict(self, temp_model_dir):
        """Test handling of corrupted state dict."""
        config = QuantizationConfig(method=QuantizationMethod.HQQ)
        adapter = create_quantization_adapter(Backend.CUDA, config)
        
        # Save corrupted file
        model_file = temp_model_dir / 'corrupted.pth'
        with open(model_file, 'wb') as f:
            f.write(b'corrupted data')
        
        # Should raise an error when loading
        with pytest.raises(Exception):  # Could be various exceptions
            adapter.load_quantized_model(str(model_file), {})
    
    def test_mismatched_config(self, temp_model_dir, simple_model):
        """Test loading with mismatched model config."""
        config = QuantizationConfig(method=QuantizationMethod.QUANTO_INT4)
        adapter = create_quantization_adapter(Backend.MPS, config)
        
        # Save model
        model_file = temp_model_dir / 'model.pth'
        torch.save(simple_model.state_dict(), model_file)
        
        # Try to load with wrong config
        wrong_config = {
            'model_type': 'different',
            'hidden_size': 64,  # Wrong size
            'num_layers': 24
        }
        
        # Should handle gracefully (might warn or partially load)
        try:
            loaded_model = adapter.load_quantized_model(str(model_file), wrong_config)
            # If it loads, it should at least return something
            assert loaded_model is not None
        except (ImportError, RuntimeError, KeyError):
            # Various errors are acceptable depending on the adapter
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])