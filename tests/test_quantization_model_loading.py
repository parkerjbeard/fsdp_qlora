"""
Unit tests for quantization model loading functionality.

Tests all adapter implementations for loading quantized models:
- BitsAndBytes
- HQQ  
- MLX
- Quanto
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest
import torch
import torch.nn as nn

from src.core.quantization_wrapper import (
    BitsAndBytesAdapter,
    HQQAdapter,
    MLXAdapter,
    QuantoAdapter,
    QuantizationConfig,
    QuantizationMethod,
    Backend
)


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class TestBitsAndBytesModelLoading:
    """Test BitsAndBytes adapter model loading."""
    
    @pytest.fixture
    def adapter(self):
        """Create BitsAndBytes adapter."""
        config = QuantizationConfig(
            method=QuantizationMethod.BNB_NF4,
            bits=4,
            compute_dtype=torch.float16
        )
        return BitsAndBytesAdapter(Backend.CUDA, config)
    
    @patch('src.core.quantization_wrapper.get_module')
    @patch('torch.load')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_dir')
    def test_load_model_from_directory(self, mock_is_dir, mock_exists, mock_torch_load, mock_get_module, adapter):
        """Test loading model from directory."""
        # Setup mocks
        mock_is_dir.return_value = True
        mock_exists.side_effect = [True, True]  # pytorch_model.bin exists, quant config exists
        
        # Mock state dict
        state_dict = {
            'linear1.weight': torch.randn(20, 10),
            'linear2.weight': torch.randn(10, 20),
        }
        mock_torch_load.return_value = state_dict
        
        # Mock bitsandbytes module
        mock_bnb = MagicMock()
        mock_bnb.Linear4bit = MagicMock(return_value=nn.Linear(10, 10))
        mock_get_module.return_value = mock_bnb
        
        # Mock model config
        model_config = {
            'model_type': 'dummy',
            'hidden_size': 768
        }
        
        # Mock file reading for quant config
        with patch('builtins.open', mock_open(read_data='{"bits": 4, "quant_type": "nf4"}')):
            with patch('transformers.AutoModelForCausalLM') as mock_auto_model:
                with patch('accelerate.init_empty_weights'):
                    # Mock model creation
                    mock_model = DummyModel()
                    mock_auto_model.from_config.return_value = mock_model
                    
                    # Test loading
                    loaded_model = adapter.load_quantized_model('/path/to/model', model_config)
                    
                    # Verify
                    assert loaded_model is not None
                    mock_torch_load.assert_called_once()
                    mock_get_module.assert_called_with('bitsandbytes', 'cuda')
    
    @patch('src.core.quantization_wrapper.get_module')
    @patch('torch.load')
    @patch('pathlib.Path.is_dir')
    def test_load_model_from_file(self, mock_is_dir, mock_torch_load, mock_get_module, adapter):
        """Test loading model from single file."""
        # Setup mocks
        mock_is_dir.return_value = False
        
        # Mock state dict
        state_dict = {
            'linear1.weight': torch.randn(20, 10),
            'linear2.weight': torch.randn(10, 20),
        }
        mock_torch_load.return_value = state_dict
        
        # Mock bitsandbytes module
        mock_bnb = MagicMock()
        mock_bnb.Linear4bit = nn.Linear
        mock_get_module.return_value = mock_bnb
        
        # Test with fallback implementation (no transformers)
        with patch('transformers.AutoModelForCausalLM', side_effect=ImportError):
            loaded_model = adapter.load_quantized_model('/path/to/model.pth', {})
            
            # Verify fallback was used
            assert loaded_model is not None
            assert hasattr(loaded_model, 'linear1_weight')
    
    @patch('src.core.quantization_wrapper.get_module')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_dir')
    def test_load_model_file_not_found(self, mock_is_dir, mock_exists, mock_get_module, adapter):
        """Test loading model when file not found."""
        # Setup mocks
        mock_is_dir.return_value = True
        mock_exists.return_value = False  # No files exist
        
        mock_bnb = MagicMock()
        mock_get_module.return_value = mock_bnb
        
        # Test
        with pytest.raises(FileNotFoundError):
            adapter.load_quantized_model('/path/to/model', {})


class TestHQQModelLoading:
    """Test HQQ adapter model loading."""
    
    @pytest.fixture
    def adapter(self):
        """Create HQQ adapter."""
        config = QuantizationConfig(
            method=QuantizationMethod.HQQ,
            bits=4,
            group_size=64,
            quant_zero=True,
            quant_scale=False
        )
        return HQQAdapter(Backend.CUDA, config)
    
    @patch('src.core.quantization_wrapper.get_module')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_dir')
    def test_load_hqq_native_format(self, mock_is_dir, mock_exists, mock_get_module, adapter):
        """Test loading HQQ model in native format."""
        # Setup mocks
        mock_is_dir.return_value = False
        
        # Mock HQQ module
        mock_hqq = MagicMock()
        mock_get_module.return_value = mock_hqq
        
        # Mock HQQModel.load
        mock_model = MagicMock()
        mock_hqq_model = MagicMock()
        mock_hqq_model.load.return_value = mock_model
        
        # Test
        with patch('hqq.models.base.HQQModel', mock_hqq_model):
            loaded_model = adapter.load_quantized_model('/path/to/hqq_model.pt', {})
            
            # Verify
            assert loaded_model == mock_model
            mock_hqq_model.load.assert_called_once_with('/path/to/hqq_model.pt')
    
    @patch('src.core.quantization_wrapper.get_module')
    @patch('torch.load')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_dir')
    def test_load_hqq_from_state_dict(self, mock_is_dir, mock_exists, mock_torch_load, mock_get_module, adapter):
        """Test loading HQQ model from state dict."""
        # Setup mocks
        mock_is_dir.return_value = True
        mock_exists.side_effect = [True, True]  # hqq_config.json and checkpoint exist
        
        # Mock HQQ quantized state dict
        state_dict = {
            'model.layer1.W_q': torch.randn(100, 50),
            'model.layer1.meta': {'scale': 0.1, 'zero': 0.0},
            'model.layer1.bias': torch.randn(100),
            'model.layer2.W_q': torch.randn(50, 100),
            'model.layer2.meta': {'scale': 0.1, 'zero': 0.0},
        }
        mock_torch_load.return_value = state_dict
        
        # Mock HQQ module
        mock_hqq = MagicMock()
        mock_get_module.return_value = mock_hqq
        
        # Mock model config
        model_config = {'model_type': 'dummy'}
        
        # Mock file reading for HQQ config
        hqq_config = {
            'weight_quant_params': {
                'nbits': 4,
                'group_size': 64,
                'quant_zero': True,
                'quant_scale': False
            }
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(hqq_config))):
            with patch('transformers.AutoModelForCausalLM') as mock_auto_model:
                with patch('accelerate.init_empty_weights'):
                    with patch('hqq.core.quantize.Quantizer') as mock_quantizer:
                        # Mock model creation
                        mock_model = MagicMock()
                        mock_model.named_modules.return_value = [
                            ('layer1', MagicMock(W_q=None, meta=None, bias=None)),
                            ('layer2', MagicMock(W_q=None, meta=None))
                        ]
                        mock_auto_model.from_config.return_value = mock_model
                        
                        # Test loading
                        loaded_model = adapter.load_quantized_model('/path/to/model', model_config)
                        
                        # Verify
                        assert loaded_model is not None
                        mock_torch_load.assert_called_once()


class TestMLXModelLoading:
    """Test MLX adapter model loading."""
    
    @pytest.fixture
    def adapter(self):
        """Create MLX adapter."""
        config = QuantizationConfig(
            method=QuantizationMethod.MLX_INT4,
            bits=4,
            group_size=32
        )
        return MLXAdapter(Backend.MLX, config)
    
    @patch('mlx.core')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_dir')
    def test_load_mlx_npz_format(self, mock_is_dir, mock_exists, mock_mx, adapter):
        """Test loading MLX model in NPZ format."""
        # Setup mocks
        mock_is_dir.return_value = False
        
        # Mock MLX load
        mock_weights = {'layer1.weight': MagicMock(), 'layer2.weight': MagicMock()}
        mock_mx.load.return_value = mock_weights
        
        # Test
        with patch('mlx.nn'):
            with patch('mlx.utils.tree_flatten'):
                with patch('mlx.utils.tree_unflatten'):
                    with patch('src.core.quantization_wrapper.MLXModelWrapper') as mock_wrapper:
                        mock_wrapped = MagicMock()
                        mock_wrapper.return_value = mock_wrapped
                        
                        loaded_model = adapter.load_quantized_model('/path/to/model.npz', {})
                        
                        # Verify
                        assert loaded_model == mock_wrapped
                        mock_mx.load.assert_called_once_with('/path/to/model.npz')
                        assert hasattr(loaded_model, 'mlx_model')
    
    @patch('torch.load')
    @patch('mlx.core')
    @patch('pathlib.Path.suffix', new_callable=lambda: MagicMock(return_value='.bin'))
    @patch('pathlib.Path.is_dir')
    def test_load_mlx_from_pytorch(self, mock_is_dir, mock_suffix, mock_mx, mock_torch_load, adapter):
        """Test loading MLX model from PyTorch checkpoint."""
        # Setup mocks
        mock_is_dir.return_value = False
        
        # Mock PyTorch state dict
        state_dict = {
            'model.layer1.weight': torch.randn(20, 10),
            'model.layer2.weight': torch.randn(10, 20),
        }
        mock_torch_load.return_value = state_dict
        
        # Mock MLX array conversion
        mock_mx.array = MagicMock(side_effect=lambda x: f"mlx_{x.shape}")
        
        # Test
        with patch('mlx.nn'):
            with patch('mlx.utils.tree_flatten'):
                with patch('mlx.utils.tree_unflatten'):
                    with patch('transformers.AutoModelForCausalLM') as mock_auto_model:
                        with patch('accelerate.init_empty_weights'):
                            with patch('src.core.quantization_wrapper.MLXModelWrapper') as mock_wrapper:
                                # Mock model creation
                                mock_model = DummyModel()
                                mock_auto_model.from_config.return_value = mock_model
                                mock_wrapped = MagicMock()
                                mock_wrapper.return_value = mock_wrapped
                                
                                # Test loading
                                loaded_model = adapter.load_quantized_model('/path/to/model.bin', {})
                                
                                # Verify
                                assert loaded_model == mock_wrapped
                                assert hasattr(loaded_model, 'mlx_model')
                                mock_torch_load.assert_called_once()


class TestQuantoModelLoading:
    """Test Quanto adapter model loading."""
    
    @pytest.fixture
    def adapter(self):
        """Create Quanto adapter."""
        config = QuantizationConfig(
            method=QuantizationMethod.QUANTO_INT4,
            bits=4,
            group_size=128
        )
        return QuantoAdapter(Backend.MPS, config)
    
    @patch('optimum.quanto.quantize')
    @patch('optimum.quanto.freeze')
    @patch('optimum.quanto.qint4')
    @patch('torch.load')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_dir')
    def test_load_quanto_from_directory(self, mock_is_dir, mock_exists, mock_torch_load, 
                                       mock_qint4, mock_freeze, mock_quantize, adapter):
        """Test loading Quanto model from directory."""
        # Setup mocks
        mock_is_dir.return_value = True
        mock_exists.side_effect = [True, True]  # checkpoint and config exist
        
        # Mock state dict
        state_dict = {
            'model.layer1.weight': torch.randn(20, 10),
            'model.layer2.weight': torch.randn(10, 20),
        }
        mock_torch_load.return_value = state_dict
        
        # Mock Quanto config
        quanto_config = {
            'quantization_config': {
                'quant_method': 'quanto',
                'bits': 4
            }
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(quanto_config))):
            with patch('transformers.AutoModelForCausalLM') as mock_auto_model:
                with patch('accelerate.init_empty_weights'):
                    # Mock model creation
                    mock_model = DummyModel()
                    mock_auto_model.from_config.return_value = mock_model
                    
                    # Test loading
                    loaded_model = adapter.load_quantized_model('/path/to/model', {})
                    
                    # Verify
                    assert loaded_model is not None
                    mock_quantize.assert_called_once_with(mock_model, weights=mock_qint4)
                    mock_freeze.assert_called_once_with(mock_model)
                    mock_torch_load.assert_called_once()
    
    @patch('optimum.quanto.quantize')
    @patch('optimum.quanto.freeze')
    @patch('optimum.quanto.qint2')
    @patch('torch.load')
    @patch('pathlib.Path.is_dir')
    def test_load_quanto_single_file(self, mock_is_dir, mock_torch_load, 
                                    mock_qint2, mock_freeze, mock_quantize, adapter):
        """Test loading Quanto model from single file."""
        # Setup mocks
        mock_is_dir.return_value = False
        
        # Mock checkpoint with embedded config
        checkpoint = {
            'state_dict': {
                'model.layer1.weight': torch.randn(20, 10),
                'model.layer2.weight': torch.randn(10, 20),
            },
            'quanto_config': {
                'bits': 2
            }
        }
        mock_torch_load.return_value = checkpoint
        
        # Change adapter to 2-bit
        adapter.config.bits = 2
        adapter.config.method = QuantizationMethod.QUANTO_INT2
        
        with patch('transformers.AutoModelForCausalLM') as mock_auto_model:
            with patch('accelerate.init_empty_weights'):
                # Mock model creation
                mock_model = DummyModel()
                mock_auto_model.from_config.return_value = mock_model
                
                # Test loading
                loaded_model = adapter.load_quantized_model('/path/to/model.pth', {})
                
                # Verify
                assert loaded_model is not None
                mock_quantize.assert_called_once_with(mock_model, weights=mock_qint2)
                mock_freeze.assert_called_once()


class TestModelLoadingEdgeCases:
    """Test edge cases and error handling for model loading."""
    
    def test_unsupported_file_format_mlx(self):
        """Test MLX adapter with unsupported file format."""
        config = QuantizationConfig(method=QuantizationMethod.MLX_INT4)
        adapter = MLXAdapter(Backend.MLX, config)
        
        # We need to mock the MLX imports first to avoid ImportError
        with patch('mlx.core'):
            with patch('mlx.nn'):
                with patch('mlx.utils.tree_flatten'):
                    with patch('mlx.utils.tree_unflatten'):
                        with patch('pathlib.Path.is_dir', return_value=False):
                            with patch('pathlib.Path.suffix', new='.txt'):
                                with pytest.raises(ValueError, match="Unsupported checkpoint format"):
                                    adapter.load_quantized_model('/path/to/model.txt', {})
    
    def test_missing_required_module_hqq(self):
        """Test HQQ adapter when HQQ module not available."""
        config = QuantizationConfig(method=QuantizationMethod.HQQ)
        adapter = HQQAdapter(Backend.CUDA, config)
        
        with patch('src.core.quantization_wrapper.get_module', side_effect=ImportError):
            with pytest.raises(ImportError, match="HQQ not available"):
                adapter.load_quantized_model('/path/to/model', {})
    
    def test_missing_required_module_quanto(self):
        """Test Quanto adapter when Quanto module not available."""
        config = QuantizationConfig(method=QuantizationMethod.QUANTO_INT4)
        adapter = QuantoAdapter(Backend.MPS, config)
        
        with patch('optimum.quanto.quantize', side_effect=ImportError):
            with pytest.raises(ImportError, match="Quanto is required"):
                adapter.load_quantized_model('/path/to/model', {})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])