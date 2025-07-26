"""
Integration tests for MLX functionality
"""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
import json
import numpy as np

# Mock MLX for testing
from tests.test_mlx_models import mlx_mock, mx_mock, nn_mlx_mock, MockArray

# Import after mocking
from src.backends.mlx.mlx_model_wrapper import (
    create_mlx_model,
    MLXModelWrapper,
    UnifiedMemoryOptimizer
)
from src.backends.mlx.pytorch_mlx_bridge import (
    convert_huggingface_to_mlx,
    convert_checkpoint,
    ModelConverter
)
from src.backends.mlx.models import (
    LlamaModel, LlamaConfig,
    MistralModel, MistralConfig,
    PhiModel, PhiConfig,
    QwenModel, QwenConfig
)
from src.core.quantization_wrapper import QuantizationConfig, QuantizationMethod


class TestMLXModelIntegration:
    """Integration tests for MLX model functionality."""
    
    def test_end_to_end_model_creation(self):
        """Test complete model creation pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock model files
            config_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "vocab_size": 1000,
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
                
            # Create mock weights
            weights_path = os.path.join(tmpdir, "weights.npz")
            with patch('src.backends.mlx.mlx_model_wrapper.mx.save'):
                with patch('src.backends.mlx.mlx_model_wrapper.mx.load', return_value={}):
                    # Mock tokenizer
                    with patch('src.backends.mlx.mlx_model_wrapper.AutoTokenizer') as mock_tokenizer:
                        mock_tokenizer.from_pretrained.return_value = MagicMock()
                        
                        # Create model
                        wrapper = create_mlx_model(tmpdir)
                        
                        assert wrapper is not None
                        assert isinstance(wrapper, MLXModelWrapper)
                        
    def test_model_with_quantization(self):
        """Test model creation with quantization."""
        # Create quantization config
        quant_config = QuantizationConfig(
            method=QuantizationMethod.MLX_INT4,
            bits=4,
            group_size=64
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock model
            config_data = {
                "model_type": "mistral",
                "architectures": ["MistralForCausalLM"],
                "vocab_size": 1000,
                "hidden_size": 128,
                "num_hidden_layers": 2
            }
            
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
                
            # Mock MLX quantizer
            with patch('src.backends.mlx.mlx_model_wrapper.MLXQuantizer') as mock_quantizer:
                mock_quantizer.return_value.quantize_model.return_value = MagicMock()
                
                with patch('src.backends.mlx.mlx_model_wrapper.AutoTokenizer'):
                    with patch('src.backends.mlx.mlx_model_wrapper.mx.load', return_value={}):
                        wrapper = create_mlx_model(
                            tmpdir,
                            quantization_config=quant_config
                        )
                        
                        # Verify quantizer was called
                        mock_quantizer.assert_called_once()
                        mock_quantizer.return_value.quantize_model.assert_called_once()
                        
    def test_model_with_lora(self):
        """Test model creation with LoRA."""
        lora_config = {
            "rank": 16,
            "alpha": 32.0,
            "dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock model
            config_data = {
                "model_type": "phi",
                "architectures": ["PhiForCausalLM"],
                "vocab_size": 1000,
                "hidden_size": 128,
                "num_hidden_layers": 2
            }
            
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
                
            with patch('src.backends.mlx.mlx_model_wrapper.AutoTokenizer'):
                with patch('src.backends.mlx.mlx_model_wrapper.mx.load', return_value={}):
                    # Create mock model with apply_lora method
                    mock_apply_lora = MagicMock()
                    
                    with patch('src.backends.mlx.models.phi.PhiModel') as mock_phi:
                        mock_phi.return_value.apply_lora = mock_apply_lora
                        mock_phi.return_value.config = PhiConfig()
                        
                        wrapper = create_mlx_model(
                            tmpdir,
                            lora_config=lora_config
                        )
                        
                        # Verify LoRA was applied
                        mock_apply_lora.assert_called_once_with(
                            ["q_proj", "v_proj", "k_proj", "o_proj"]
                        )
                        
    def test_qwen_model_special_handling(self):
        """Test Qwen model with trust_remote_code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_data = {
                "model_type": "qwen",
                "architectures": ["QwenForCausalLM"],
                "vocab_size": 151936,
                "hidden_size": 4096,
                "num_hidden_layers": 32
            }
            
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
                
            # Test that trust_remote_code is set for Qwen
            with patch('src.backends.mlx.mlx_model_wrapper.AutoTokenizer') as mock_tokenizer:
                with patch('src.backends.mlx.mlx_model_wrapper.mx.load', return_value={}):
                    wrapper = create_mlx_model(tmpdir)
                    
                    # Verify trust_remote_code was passed
                    mock_tokenizer.from_pretrained.assert_called_with(
                        tmpdir,
                        trust_remote_code=True
                    )


class TestMLXTrainingIntegration:
    """Test MLX training integration."""
    
    def test_training_loop_integration(self):
        """Test integration with training loop."""
        import torch
        
        # Create mock model
        mock_mlx_model = MagicMock()
        mock_mlx_model.config = LlamaConfig(vocab_size=1000, hidden_size=128)
        mock_mlx_model.return_value = MockArray(shape=(2, 10, 1000))
        
        # Create wrapper
        wrapper = MLXModelWrapper(mock_mlx_model, None)
        
        # Simulate training step
        input_ids = torch.randint(0, 1000, (2, 10))
        labels = torch.randint(0, 1000, (2, 10))
        
        with patch.object(wrapper, '_torch_to_mlx', return_value=MockArray()):
            with patch.object(wrapper, '_mlx_to_torch') as mock_convert:
                # Return proper tensor for logits
                mock_convert.return_value = torch.randn(2, 10, 1000)
                
                # Forward pass
                outputs = wrapper.forward(input_ids, labels=labels)
                
                assert 'loss' in outputs
                assert 'logits' in outputs
                
                # Check loss is a scalar tensor
                assert outputs['loss'] is not None
                
    def test_gradient_accumulation(self):
        """Test gradient accumulation workflow."""
        import torch
        
        # Create mock model with LoRA
        mock_mlx_model = MagicMock()
        mock_mlx_model.config = MistralConfig()
        
        # Create mock LoRA layer
        mock_lora_layer = MagicMock()
        mock_lora_layer.lora_a = MockArray(shape=(16, 128))
        mock_lora_layer.lora_b = MockArray(shape=(128, 16))
        
        # Setup model structure
        mock_mlx_model.layers = [MagicMock()]
        mock_mlx_model.layers[0].attention = mock_lora_layer
        
        wrapper = MLXModelWrapper(mock_mlx_model, None)
        
        # Test parameter collection
        with patch.object(wrapper, '_mlx_to_torch', return_value=torch.randn(16, 128)):
            params = wrapper.parameters()
            
            # Should have collected LoRA parameters
            assert len(params) > 0
            
    def test_memory_optimization(self):
        """Test unified memory optimization."""
        mock_model = MagicMock()
        wrapper = MLXModelWrapper(mock_model, None)
        
        # Create optimizer
        optimizer = UnifiedMemoryOptimizer(wrapper)
        
        # Test memory optimization
        optimizer.optimize_memory_layout()
        
        # Check caches were cleared
        assert len(wrapper._input_cache) == 0
        assert len(wrapper._output_cache) == 0
        
        # Test memory profiling
        with patch('psutil.Process') as mock_process:
            mock_memory = MagicMock()
            mock_memory.rss = 4 * 1e9  # 4GB
            mock_memory.vms = 8 * 1e9  # 8GB
            mock_process.return_value.memory_info.return_value = mock_memory
            mock_process.return_value.memory_percent.return_value = 50.0
            
            with patch('psutil.virtual_memory') as mock_virtual:
                mock_virtual.return_value.available = 16 * 1e9  # 16GB
                
                profile = optimizer.profile_memory_usage()
                
                assert profile['rss_gb'] == 4.0
                assert profile['vms_gb'] == 8.0
                assert profile['available_gb'] == 16.0
                assert profile['percent'] == 50.0


class TestPyTorchMLXBridge:
    """Test PyTorch-MLX bridge functionality."""
    
    def test_model_converter(self):
        """Test model converter."""
        converter = ModelConverter()
        
        # Test it has the expected converters
        assert hasattr(converter, 'layer_converters')
        assert hasattr(converter, 'convert_module')
        
    @patch('src.backends.mlx.pytorch_mlx_bridge.AutoModel')
    @patch('src.backends.mlx.pytorch_mlx_bridge.convert_model')
    @patch('src.backends.mlx.pytorch_mlx_bridge.save_model')
    def test_convert_huggingface_to_mlx(self, mock_save, mock_convert, mock_auto):
        """Test HuggingFace to MLX conversion."""
        # Mock HuggingFace model
        mock_torch_model = MagicMock()
        mock_auto.from_pretrained.return_value = mock_torch_model
        
        # Mock conversion
        mock_mlx_model = MagicMock()
        mock_convert.return_value = mock_mlx_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test conversion
            model, tokenizer = convert_huggingface_to_mlx(
                "bert-base-uncased",
                tmpdir,
                quantize=False
            )
            
            # Verify calls
            mock_auto.from_pretrained.assert_called_once()
            mock_convert.assert_called_once()
            mock_save.assert_called_once()
            
    def test_checkpoint_conversion(self):
        """Test checkpoint conversion between frameworks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock PyTorch checkpoint
            import torch
            checkpoint_data = {
                'model.embed_tokens.weight': torch.randn(1000, 128),
                'model.layers.0.self_attn.q_proj.weight': torch.randn(128, 128)
            }
            
            input_path = os.path.join(tmpdir, "input")
            output_path = os.path.join(tmpdir, "output")
            os.makedirs(input_path)
            
            torch.save(checkpoint_data, os.path.join(input_path, "model.pt"))
            
            # Test conversion
            with patch('src.backends.mlx.pytorch_mlx_bridge.mx.save'):
                result_path = convert_checkpoint(
                    os.path.join(input_path, "model.pt"),
                    output_path,
                    from_framework="pytorch",
                    to_framework="mlx"
                )
                
                assert result_path == output_path
                assert os.path.exists(os.path.join(output_path, "conversion_info.json"))


class TestMLXModelSaving:
    """Test model saving and loading."""
    
    def test_save_pretrained(self):
        """Test saving model in MLX format."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.config = LlamaConfig()
        mock_model.parameters.return_value = {'weight': MockArray()}
        
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        
        wrapper = MLXModelWrapper(mock_model, mock_tokenizer)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            with patch('src.backends.mlx.mlx_model_wrapper.mx.save'):
                with patch('src.backends.mlx.mlx_model_wrapper.tree_flatten', 
                          return_value={'weight': MockArray()}):
                    wrapper.save_pretrained(tmpdir)
                    
                    # Check files were created
                    assert os.path.exists(os.path.join(tmpdir, "config.json"))
                    
                    # Verify tokenizer save was called
                    mock_tokenizer.save_pretrained.assert_called_with(tmpdir)
                    
    def test_load_pretrained(self):
        """Test loading model from MLX format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_data = {
                "model_type": "llama",
                "vocab_size": 1000,
                "hidden_size": 128,
                "num_hidden_layers": 2
            }
            
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
                
            # Mock weights loading
            with patch('src.backends.mlx.mlx_model_wrapper.mx.load', return_value={}):
                with patch('src.backends.mlx.mlx_model_wrapper.AutoTokenizer'):
                    # Need to mock the model creation
                    with patch('src.backends.mlx.mlx_model_wrapper.MLXModel') as mock_model_class:
                        mock_model = MagicMock()
                        mock_model.load_weights = MagicMock()
                        mock_model_class.return_value = mock_model
                        
                        wrapper = MLXModelWrapper.from_pretrained(tmpdir)
                        
                        assert wrapper is not None
                        assert isinstance(wrapper, MLXModelWrapper)


class TestMLXPerformance:
    """Test MLX performance optimizations."""
    
    def test_caching_behavior(self):
        """Test input/output caching."""
        import torch
        
        mock_model = MagicMock()
        mock_model.return_value = MockArray()
        
        wrapper = MLXModelWrapper(mock_model, None)
        
        # Create tensors
        tensor1 = torch.randn(1, 10)
        tensor2 = torch.randn(1, 10)
        
        # Convert same tensor multiple times
        mlx1 = wrapper._torch_to_mlx(tensor1)
        mlx2 = wrapper._torch_to_mlx(tensor1)  # Should use cache
        
        # Check cache
        assert len(wrapper._input_cache) > 0
        
        # Convert different tensor
        mlx3 = wrapper._torch_to_mlx(tensor2)
        
        # Cache should have both
        assert len(wrapper._input_cache) >= 2
        
    def test_model_sanitization(self):
        """Test model sanitization for optimization."""
        config = LlamaConfig()
        model = LlamaModel(config)
        
        # Test sanitize method exists
        assert hasattr(model, 'sanitize')
        
        # Run sanitization
        model.sanitize()  # Should not raise errors


class TestMLXErrorHandling:
    """Test error handling in MLX components."""
    
    def test_missing_mlx_import(self):
        """Test handling when MLX is not available."""
        # This is already handled by our mocking, but test the concept
        with patch('src.backends.mlx.mlx_model_wrapper.MLX_AVAILABLE', False):
            with pytest.raises(ImportError, match="MLX is not available"):
                create_mlx_model("test-model")
                
    def test_unsupported_model_type(self):
        """Test error on unsupported model."""
        config_data = {
            "model_type": "unknown_model",
            "architectures": ["UnknownModel"]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
                
            with patch('src.backends.mlx.mlx_model_wrapper.AutoTokenizer'):
                with pytest.raises(ValueError, match="Unsupported model type"):
                    create_mlx_model(tmpdir)
                    
    def test_missing_weights(self):
        """Test handling missing weight files."""
        config = LlamaConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="Model weights not found"):
                LlamaModel.from_pretrained(tmpdir, config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])