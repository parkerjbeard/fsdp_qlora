"""
Unit tests for MLX model implementations
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
import sys

# Mock MLX before importing our modules
mlx_mock = MagicMock()
mx_mock = MagicMock()
nn_mlx_mock = MagicMock()

# Set up array mock
class MockArray:
    def __init__(self, shape=None, dtype=None):
        self.shape = shape or (1, 10, 4096)
        self.dtype = dtype or 'float32'
        self._data = np.random.randn(*self.shape) if shape else None
        
    def transpose(self, *axes):
        return MockArray(shape=tuple(self.shape[i] for i in axes))
        
    def reshape(self, *shape):
        return MockArray(shape=shape)
        
    def __matmul__(self, other):
        return MockArray(shape=(self.shape[0], self.shape[1], other.shape[-1]))
        
    def __add__(self, other):
        return self
        
    def __mul__(self, other):
        return self
        
    def astype(self, dtype):
        self.dtype = dtype
        return self

mx_mock.array = MockArray
mx_mock.zeros = lambda shape: MockArray(shape=shape)
mx_mock.ones = lambda shape: MockArray(shape=shape)
mx_mock.concatenate = lambda arrays, axis: arrays[0]
mx_mock.softmax = lambda x, axis: x
mx_mock.float32 = 'float32'
mx_mock.inf = float('inf')
mx_mock.full = lambda shape, value: MockArray(shape=shape)
mx_mock.tile = lambda x, reps: x
mx_mock.repeat = lambda x, repeats, axis: x
mx_mock.split = lambda x, indices, axis: [x, x, x]
mx_mock.random.uniform = lambda low, high, shape: MockArray(shape=shape)
mx_mock.random.normal = lambda shape, scale: MockArray(shape=shape)
mx_mock.random.categorical = lambda x: MockArray(shape=(1,))
mx_mock.argmax = lambda x, axis: MockArray(shape=(1,))
mx_mock.cumsum = lambda x, axis: x
mx_mock.sum = lambda x, axis, keepdims: MockArray(shape=(1,))
mx_mock.log = lambda x: x
mx_mock.take_along_axis = lambda x, indices, axis: x
mx_mock.argsort = lambda x, axis: x
mx_mock.where = lambda condition, x, y: x
mx_mock.squeeze = lambda x, axis: x

# Mock nn module
nn_mlx_mock.Module = type('Module', (), {})
nn_mlx_mock.Linear = MagicMock
nn_mlx_mock.Embedding = MagicMock
nn_mlx_mock.LayerNorm = MagicMock
nn_mlx_mock.RoPE = MagicMock
nn_mlx_mock.Sequential = MagicMock
nn_mlx_mock.Dropout = MagicMock
nn_mlx_mock.GELU = MagicMock
nn_mlx_mock.MultiHeadAttention = MagicMock
nn_mlx_mock.MultiHeadAttention.create_additive_causal_mask = lambda T: MockArray(shape=(T, T))
nn_mlx_mock.silu = lambda x: x

# Fast module
mx_mock.fast = MagicMock()
mx_mock.fast.rms_norm = lambda x, weight, eps: x

# Load/save functions
mx_mock.load = lambda path: {}
mx_mock.save = lambda path, data: None

sys.modules['mlx'] = mlx_mock
sys.modules['mlx.core'] = mx_mock
sys.modules['mlx.nn'] = nn_mlx_mock
sys.modules['mlx.utils'] = MagicMock()
sys.modules['mlx.optimizers'] = MagicMock()

# Now import our modules
from src.backends.mlx.models import (
    LlamaModel, LlamaConfig,
    MistralModel, MistralConfig,
    PhiModel, PhiConfig,
    QwenModel, QwenConfig,
    MLXAttention, RMSNorm
)
from src.backends.mlx.mlx_model_wrapper import (
    MLXModelWrapper,
    create_mlx_model,
    convert_pytorch_to_mlx,
    PyTorchToMLXConverter
)


class TestMLXModels:
    """Test MLX model implementations."""
    
    def test_llama_config(self):
        """Test LLaMA configuration."""
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8
        )
        
        assert config.model_type == "llama"
        assert config.vocab_size == 32000
        assert config.hidden_size == 4096
        assert config.num_key_value_heads == 8
        
    def test_llama_config_from_huggingface(self):
        """Test creating LLaMA config from HuggingFace format."""
        hf_config = {
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rope_theta": 10000.0,
            "rope_scaling": {"type": "linear", "factor": 2.0}
        }
        
        config = LlamaConfig.from_huggingface(hf_config)
        
        assert config.vocab_size == 32000
        assert config.rope_scaling["factor"] == 2.0
        
    def test_mistral_config(self):
        """Test Mistral configuration."""
        config = MistralConfig(
            sliding_window=4096,
            num_key_value_heads=8
        )
        
        assert config.model_type == "mistral"
        assert config.sliding_window == 4096
        assert config.num_key_value_heads == 8
        
    def test_phi_config_variants(self):
        """Test Phi configuration for different variants."""
        # Phi-2 config
        hf_config_phi2 = {
            "model_type": "phi",
            "vocab_size": 51200,
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "partial_rotary_factor": 0.4
        }
        
        config_phi2 = PhiConfig.from_huggingface(hf_config_phi2)
        assert config_phi2.model_type == "phi2"
        assert config_phi2.vocab_size == 51200
        assert config_phi2.partial_rotary_factor == 0.4
        
        # Phi-3 config
        hf_config_phi3 = {
            "model_type": "phi3",
            "_name_or_path": "microsoft/Phi-3-mini",
            "vocab_size": 32064,
            "hidden_size": 3072,
            "num_hidden_layers": 32
        }
        
        config_phi3 = PhiConfig.from_huggingface(hf_config_phi3)
        assert config_phi3.model_type == "phi3"
        assert config_phi3.vocab_size == 32064
        
    def test_qwen_config(self):
        """Test Qwen configuration."""
        config = QwenConfig(
            vocab_size=151936,
            use_sliding_window=True,
            sliding_window_size=32768
        )
        
        assert config.model_type == "qwen"
        assert config.vocab_size == 151936
        assert config.use_sliding_window == True
        
    def test_qwen_config_rope_handling(self):
        """Test Qwen RoPE configuration handling."""
        # Qwen2 format with rope in separate key
        hf_config = {
            "model_type": "qwen",
            "rope": {
                "base": 1000000.0,
                "scaling_factor": 2.0
            }
        }
        
        config = QwenConfig.from_huggingface(hf_config)
        assert config.rope_theta == 1000000.0
        assert config.rope_scaling["factor"] == 2.0
        
    def test_rms_norm(self):
        """Test RMS normalization layer."""
        norm = RMSNorm(dims=4096, eps=1e-5)
        
        # Test forward pass
        x = MockArray(shape=(2, 10, 4096))
        output = norm(x)
        
        assert output.shape == x.shape
        
    def test_mlx_attention(self):
        """Test MLX attention module."""
        attention = MLXAttention(
            dims=4096,
            num_heads=32,
            num_kv_heads=8,
            rope_base=10000.0
        )
        
        # Test properties
        assert attention.num_heads == 32
        assert attention.num_kv_heads == 8
        assert attention.head_dim == 128  # 4096 / 32
        
        # Test forward pass
        x = MockArray(shape=(2, 10, 4096))
        output, cache = attention(x)
        
        assert output.shape == x.shape
        assert cache is None  # No cache provided
        
    def test_mlx_attention_with_cache(self):
        """Test MLX attention with key-value cache."""
        attention = MLXAttention(
            dims=4096,
            num_heads=32,
            num_kv_heads=8
        )
        
        # Create cache
        cache = (
            MockArray(shape=(2, 8, 5, 128)),  # key cache
            MockArray(shape=(2, 8, 5, 128))   # value cache
        )
        
        x = MockArray(shape=(2, 1, 4096))  # Single token
        output, new_cache = attention(x, cache=cache)
        
        assert output.shape == (2, 1, 4096)
        assert new_cache is not None
        
    def test_llama_model_creation(self):
        """Test LLaMA model creation."""
        config = LlamaConfig(
            vocab_size=32000,
            hidden_size=128,  # Small for testing
            num_hidden_layers=2,
            num_attention_heads=4
        )
        
        model = LlamaModel(config)
        
        assert hasattr(model, 'tok_embeddings')
        assert hasattr(model, 'layers')
        assert len(model.layers) == 2
        assert hasattr(model, 'norm')
        assert hasattr(model, 'lm_head')
        
    def test_mistral_model_creation(self):
        """Test Mistral model creation."""
        config = MistralConfig(
            vocab_size=32000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            sliding_window=128
        )
        
        model = MistralModel(config)
        
        assert hasattr(model, 'tok_embeddings')
        assert len(model.layers) == 2
        
    def test_phi_model_creation(self):
        """Test Phi model creation."""
        config = PhiConfig(
            vocab_size=51200,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            partial_rotary_factor=0.4
        )
        
        model = PhiModel(config)
        
        assert hasattr(model, 'tok_embeddings')
        assert len(model.layers) == 2
        
    def test_qwen_model_creation(self):
        """Test Qwen model creation."""
        config = QwenConfig(
            vocab_size=151936,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4
        )
        
        model = QwenModel(config)
        
        assert hasattr(model, 'tok_embeddings')
        assert len(model.layers) == 2


class TestPyTorchMLXConversion:
    """Test PyTorch to MLX conversion."""
    
    def test_tensor_converter(self):
        """Test tensor conversion."""
        import torch
        
        converter = PyTorchToMLXConverter()
        
        # Create PyTorch tensor
        torch_tensor = torch.randn(2, 10, 4096)
        
        # Convert to MLX
        mlx_array = converter.convert_tensor(torch_tensor)
        
        # Should return our mock array
        assert hasattr(mlx_array, 'shape')
        
    def test_linear_layer_conversion(self):
        """Test linear layer conversion."""
        import torch.nn as torch_nn
        
        converter = PyTorchToMLXConverter()
        
        # Create PyTorch linear layer
        torch_linear = torch_nn.Linear(4096, 4096, bias=True)
        
        # Mock the conversion
        with patch.object(converter, 'convert_tensor', return_value=MockArray()):
            mlx_linear = converter.convert_linear_layer(torch_linear, quantize=False)
            
            # Check it was called to create linear layer
            assert mlx_linear is not None
            
    def test_embedding_conversion(self):
        """Test embedding layer conversion."""
        import torch.nn as torch_nn
        
        converter = PyTorchToMLXConverter()
        
        # Create PyTorch embedding
        torch_embedding = torch_nn.Embedding(32000, 4096)
        
        # Mock the conversion
        with patch.object(converter, 'convert_tensor', return_value=MockArray()):
            mlx_embedding = converter.convert_embedding(torch_embedding)
            
            assert mlx_embedding is not None


class TestMLXModelWrapper:
    """Test MLX model wrapper functionality."""
    
    def test_model_wrapper_creation(self):
        """Test creating model wrapper."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.config = LlamaConfig()
        
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        
        # Create wrapper
        wrapper = MLXModelWrapper(mock_model, mock_tokenizer)
        
        assert wrapper.mlx_model == mock_model
        assert wrapper.tokenizer == mock_tokenizer
        
    def test_wrapper_forward_pass(self):
        """Test wrapper forward pass."""
        import torch
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.return_value = MockArray(shape=(1, 10, 32000))
        mock_model.config = LlamaConfig(vocab_size=32000)
        
        # Create wrapper
        wrapper = MLXModelWrapper(mock_model, None)
        
        # Create input
        input_ids = torch.randint(0, 32000, (1, 10))
        
        # Forward pass
        with patch.object(wrapper, '_torch_to_mlx', return_value=MockArray()):
            with patch.object(wrapper, '_mlx_to_torch', return_value=torch.randn(1, 10, 32000)):
                output = wrapper.forward(input_ids)
                
                assert 'logits' in output
                assert output['logits'].shape == (1, 10, 32000)
                
    def test_wrapper_with_labels(self):
        """Test wrapper forward pass with labels."""
        import torch
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.return_value = MockArray(shape=(1, 10, 32000))
        mock_model.config = LlamaConfig(vocab_size=32000)
        
        # Create wrapper
        wrapper = MLXModelWrapper(mock_model, None)
        
        # Create input and labels
        input_ids = torch.randint(0, 32000, (1, 10))
        labels = torch.randint(0, 32000, (1, 10))
        
        # Forward pass
        with patch.object(wrapper, '_torch_to_mlx', return_value=MockArray()):
            with patch.object(wrapper, '_mlx_to_torch', return_value=torch.randn(1, 10, 32000)):
                output = wrapper.forward(input_ids, labels=labels)
                
                assert 'loss' in output
                assert 'logits' in output
                
    def test_wrapper_train_eval_modes(self):
        """Test wrapper train/eval mode switching."""
        mock_model = MagicMock()
        wrapper = MLXModelWrapper(mock_model, None)
        
        # Test train mode
        wrapper.train()
        assert wrapper.training == True
        
        # Test eval mode
        wrapper.eval()
        assert wrapper.training == False


class TestMLXModelFactory:
    """Test MLX model factory functions."""
    
    @patch('src.backends.mlx.mlx_model_wrapper.AutoConfig')
    @patch('src.backends.mlx.mlx_model_wrapper.AutoTokenizer')
    def test_create_mlx_model_llama(self, mock_tokenizer, mock_config):
        """Test creating LLaMA model with factory."""
        # Mock HuggingFace config
        mock_config.from_pretrained.return_value.to_dict.return_value = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }
        
        # Mock tokenizer
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        # Create model
        with patch('src.backends.mlx.mlx_model_wrapper.mx.load', return_value={}):
            wrapper = create_mlx_model("meta-llama/Llama-2-7b-hf")
            
            assert wrapper is not None
            assert hasattr(wrapper, 'mlx_model')
            assert hasattr(wrapper, 'tokenizer')
            
    @patch('src.backends.mlx.mlx_model_wrapper.Path')
    def test_create_mlx_model_local_path(self, mock_path):
        """Test creating model from local path."""
        # Mock path exists
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__truediv__.return_value.exists.return_value = True
        
        # Mock config file
        mock_config = {
            "model_type": "mistral",
            "architectures": ["MistralForCausalLM"],
            "vocab_size": 32000,
            "hidden_size": 4096
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = str(mock_config)
            
            with patch('json.load', return_value=mock_config):
                with patch('src.backends.mlx.mlx_model_wrapper.AutoTokenizer'):
                    with patch('src.backends.mlx.mlx_model_wrapper.mx.load', return_value={}):
                        wrapper = create_mlx_model("/path/to/model")
                        
                        assert wrapper is not None
                        
    def test_create_mlx_model_unsupported(self):
        """Test error on unsupported model type."""
        mock_config = {
            "model_type": "unsupported_model",
            "architectures": ["UnsupportedModel"]
        }
        
        with patch('src.backends.mlx.mlx_model_wrapper.AutoConfig') as mock_auto_config:
            mock_auto_config.from_pretrained.return_value.to_dict.return_value = mock_config
            
            with pytest.raises(ValueError, match="Unsupported model type"):
                create_mlx_model("unsupported/model")
                
    def test_convert_pytorch_to_mlx(self):
        """Test PyTorch to MLX model conversion."""
        import torch.nn as torch_nn
        
        # Create mock PyTorch model
        class MockPyTorchModel(torch_nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch_nn.Embedding(32000, 4096)
                self.layers = torch_nn.ModuleList([
                    torch_nn.Linear(4096, 4096) for _ in range(2)
                ])
                self.norm = torch_nn.LayerNorm(4096)
                self.lm_head = torch_nn.Linear(4096, 32000)
                
        pytorch_model = MockPyTorchModel()
        
        # Create MLX config
        from src.backends.mlx.mlx_model_wrapper import MLXConfig
        config = MLXConfig(
            model_type="llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=2
        )
        
        # Test conversion
        with patch('src.backends.mlx.mlx_model_wrapper.PyTorchToMLXConverter.convert_tensor', 
                   return_value=MockArray()):
            mlx_model = convert_pytorch_to_mlx(pytorch_model, config, quantize=False)
            
            assert mlx_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])