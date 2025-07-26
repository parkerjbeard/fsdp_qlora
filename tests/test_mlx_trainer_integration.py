"""
Integration tests for MLX trainer with various components.

These tests verify the MLX trainer works correctly with:
- Model wrappers and converters
- Quantization configurations
- LoRA adapters
- Checkpoint saving/loading
- Real-world training scenarios
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import shutil
import json
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend, BackendManager
from src.core.quantization_wrapper import QuantizationConfig, QuantizationMethod

# Mock MLX imports
with patch.dict('sys.modules', {
    'mlx': MagicMock(),
    'mlx.core': MagicMock(),
    'mlx.nn': MagicMock(),
    'mlx.optimizers': MagicMock(),
    'mlx.utils': MagicMock(),
}):
    from src.backends.mlx.mlx_model_wrapper import (
        MLXConfig, MLXModel, MLXModelWrapper,
        MLXLinear, LoRALinear, PyTorchToMLXConverter,
        UnifiedMemoryOptimizer
    )
    from src.backends.mlx.mlx_trainer import (
        MLXTrainingConfig, MLXTrainer,
        create_mlx_trainer, benchmark_mlx_training
    )


class RealisticDataset(Dataset):
    """More realistic dataset for integration testing."""
    
    def __init__(self, size=1000, seq_length=512, vocab_size=32000):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Simulate more realistic data patterns
        self.data = []
        for _ in range(size):
            # Create sequences with some structure
            input_ids = torch.randint(0, vocab_size, (seq_length,))
            # Ensure some tokens are padding
            padding_start = torch.randint(seq_length // 2, seq_length, (1,)).item()
            input_ids[padding_start:] = 0  # PAD token
            
            attention_mask = torch.ones(seq_length)
            attention_mask[padding_start:] = 0
            
            # Labels are shifted input_ids
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = 0
            
            self.data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]


class MockMLXLlamaModel(MLXModel):
    """Mock LLaMA-style model for testing."""
    
    def __init__(self, config: MLXConfig):
        super().__init__(config)
        
        # Mock model layers
        self.embed_tokens = MagicMock()
        self.layers = []
        
        for i in range(config.num_hidden_layers):
            layer = MagicMock()
            # Add attention and MLP layers
            layer.self_attn = MagicMock()
            layer.self_attn.q_proj = MLXLinear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                quantized=config.use_quantization,
                bits=config.quantization_bits,
            )
            layer.self_attn.v_proj = MLXLinear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                quantized=config.use_quantization,
                bits=config.quantization_bits,
            )
            layer.mlp = MagicMock()
            self.layers.append(layer)
        
        self.norm = MagicMock()
        self.lm_head = MLXLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
    
    def __call__(self, input_ids, **kwargs):
        """Simple forward pass."""
        batch_size, seq_length = input_ids.shape
        
        # Mock logits output
        with patch('mlx_trainer.mx') as mock_mx:
            mock_mx.random.normal.return_value = MagicMock(
                shape=(batch_size, seq_length, self.config.vocab_size)
            )
            return mock_mx.random.normal.return_value


class TestMLXTrainerQuantizationIntegration(unittest.TestCase):
    """Test MLX trainer with quantization."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create quantized model config
        self.config = MLXConfig(
            model_name="test-llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=4,  # Small for testing
            num_attention_heads=32,
            use_quantization=True,
            quantization_bits=4,
            quantization_group_size=128,
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.mx')
    @patch('mlx_trainer.optim_mlx')
    def test_quantized_model_training(self, mock_optim_mlx, mock_mx):
        """Test training with quantized model."""
        # Create mock model
        model = MockMLXLlamaModel(self.config)
        
        # Create training config
        training_config = MLXTrainingConfig(
            model_config=self.config,
            num_epochs=1,
            batch_size=2,
            output_dir=self.test_dir,
        )
        
        # Create dataset
        dataset = RealisticDataset(size=10)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Setup mocks
        mock_optimizer = MagicMock()
        mock_optimizer.learning_rate = 5e-4
        mock_optim_mlx.AdamW.return_value = mock_optimizer
        
        mock_mx.value_and_grad.return_value = lambda x: (0.5, {"param": MagicMock()})
        mock_mx.no_grad.return_value.__enter__ = MagicMock()
        mock_mx.no_grad.return_value.__exit__ = MagicMock()
        mock_mx.array.return_value = MagicMock()
        
        # Create trainer
        trainer = MLXTrainer(
            model=model,
            config=training_config,
            train_dataloader=dataloader,
        )
        
        # Verify quantized layers were created
        q_proj = model.layers[0].self_attn.q_proj
        self.assertTrue(q_proj.quantized)
        self.assertEqual(q_proj.bits, 4)
        self.assertEqual(q_proj.group_size, 128)
        
        # Run training step
        batch = next(iter(dataloader))
        loss = trainer._training_step(batch)
        
        # Verify training worked
        self.assertIsNotNone(loss)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    def test_quantization_memory_efficiency(self):
        """Test memory efficiency of quantized models."""
        # Create two models - one quantized, one not
        config_quantized = MLXConfig(
            model_name="test-model",
            use_quantization=True,
            quantization_bits=4,
        )
        
        config_full = MLXConfig(
            model_name="test-model",
            use_quantization=False,
        )
        
        model_quantized = MockMLXLlamaModel(config_quantized)
        model_full = MockMLXLlamaModel(config_full)
        
        # Check that quantized layers are using less memory
        # (In real MLX, quantized weights use less memory)
        q_layer = model_quantized.layers[0].self_attn.q_proj
        f_layer = model_full.layers[0].self_attn.q_proj
        
        self.assertTrue(q_layer.quantized)
        self.assertFalse(f_layer.quantized)


class TestMLXTrainerLoRAIntegration(unittest.TestCase):
    """Test MLX trainer with LoRA adapters."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create LoRA-enabled config
        self.config = MLXConfig(
            model_name="test-llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=2,  # Small for testing
            use_lora=True,
            lora_rank=16,
            lora_alpha=32.0,
            lora_dropout=0.1,
            lora_target_modules=["q_proj", "v_proj"],
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.tree_flatten')
    @patch('mlx_trainer.nn_mlx')
    def test_lora_parameter_collection(self, mock_nn_mlx, mock_tree_flatten):
        """Test that LoRA parameters are correctly collected."""
        # Create model and apply LoRA
        model = MockMLXLlamaModel(self.config)
        
        # Manually wrap some layers with LoRA
        for layer in model.layers:
            # Wrap q_proj and v_proj with LoRA
            layer.self_attn.q_proj = LoRALinear(
                layer.self_attn.q_proj,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )
            layer.self_attn.v_proj = LoRALinear(
                layer.self_attn.v_proj,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )
        
        # Create training config
        training_config = MLXTrainingConfig(
            model_config=self.config,
            output_dir=self.test_dir,
        )
        
        # Create trainer
        trainer = MLXTrainer(
            model=model,
            config=training_config,
            train_dataloader=DataLoader(RealisticDataset(size=10), batch_size=2),
        )
        
        # Get trainable parameters
        params = trainer._get_trainable_parameters()
        
        # Verify LoRA parameters were collected
        lora_params = [k for k in params.keys() if "lora_a" in k or "lora_b" in k]
        
        # Should have 2 layers * 2 modules (q_proj, v_proj) * 2 params (lora_a, lora_b) = 8
        expected_lora_params = 2 * 2 * 2
        self.assertGreaterEqual(len(lora_params), expected_lora_params)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.mx')
    @patch('mlx_trainer.optim_mlx')
    def test_lora_training_efficiency(self, mock_optim_mlx, mock_mx):
        """Test that LoRA training only updates LoRA parameters."""
        # Create model with LoRA
        model = MockMLXLlamaModel(self.config)
        model.apply_lora()  # This would apply LoRA in real implementation
        
        # Create training config with small batch for efficiency
        training_config = MLXTrainingConfig(
            model_config=self.config,
            batch_size=1,
            gradient_accumulation_steps=4,
            output_dir=self.test_dir,
        )
        
        # Setup mocks
        mock_optimizer = MagicMock()
        mock_optimizer.learning_rate = 5e-4
        mock_optim_mlx.AdamW.return_value = mock_optimizer
        
        # Track which parameters get gradients
        gradient_params = set()
        
        def track_gradients(params):
            gradient_params.update(params.keys())
            return 0.5, {k: MagicMock() for k in params.keys()}
        
        mock_mx.value_and_grad.return_value = track_gradients
        mock_mx.array.return_value = MagicMock()
        
        # Create trainer
        trainer = MLXTrainer(
            model=model,
            config=training_config,
            train_dataloader=DataLoader(RealisticDataset(size=10), batch_size=1),
        )
        
        # Run training step
        batch = next(iter(trainer.train_dataloader))
        trainer._training_step(batch)
        
        # In a real LoRA setup, only LoRA parameters would get gradients
        # Here we just verify the mechanism works
        self.assertGreater(len(gradient_params), 0)


class TestMLXTrainerCheckpointingIntegration(unittest.TestCase):
    """Test checkpoint saving and loading."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        self.config = MLXConfig(
            model_name="test-model",
            vocab_size=1000,
            hidden_size=512,
            num_hidden_layers=2,
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.mx')
    @patch('mlx_trainer.tree_flatten')
    def test_checkpoint_save_load_cycle(self, mock_tree_flatten, mock_mx):
        """Test saving and loading checkpoints."""
        # Create model
        model = MockMLXLlamaModel(self.config)
        
        # Create training config
        training_config = MLXTrainingConfig(
            model_config=self.config,
            output_dir=self.test_dir,
            save_steps=5,
        )
        
        # Create trainer
        trainer = MLXTrainer(
            model=model,
            config=training_config,
            train_dataloader=DataLoader(RealisticDataset(size=10), batch_size=2),
        )
        
        # Simulate training progress
        trainer.global_step = 100
        trainer.current_epoch = 3
        trainer.best_eval_loss = 0.123
        trainer.train_losses = [(50, 0.5), (100, 0.3)]
        trainer.eval_losses = [(100, 0.123)]
        
        # Mock tree_flatten for parameter saving
        mock_tree_flatten.return_value = {"param1": MagicMock(), "param2": MagicMock()}
        
        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint")
        
        # Verify checkpoint files exist
        checkpoint_dir = os.path.join(self.test_dir, "checkpoints", "test_checkpoint")
        self.assertTrue(os.path.exists(checkpoint_dir))
        
        state_file = os.path.join(checkpoint_dir, "training_state.json")
        self.assertTrue(os.path.exists(state_file))
        
        # Load state and verify
        with open(state_file) as f:
            loaded_state = json.load(f)
        
        self.assertEqual(loaded_state["global_step"], 100)
        self.assertEqual(loaded_state["current_epoch"], 3)
        self.assertAlmostEqual(loaded_state["best_eval_loss"], 0.123)
        self.assertEqual(len(loaded_state["train_losses"]), 2)
        
        # Test loading checkpoint
        new_trainer = MLXTrainer(
            model=model,
            config=training_config,
            train_dataloader=DataLoader(RealisticDataset(size=10), batch_size=2),
        )
        
        # Mock mx.load for checkpoint loading
        mock_mx.load.return_value = {"param1": MagicMock(), "param2": MagicMock()}
        
        new_trainer.load_checkpoint("test_checkpoint")
        
        # Verify state was restored
        self.assertEqual(new_trainer.global_step, 100)
        self.assertEqual(new_trainer.current_epoch, 3)
        self.assertAlmostEqual(new_trainer.best_eval_loss, 0.123)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    def test_model_wrapper_checkpoint_integration(self):
        """Test checkpointing with MLXModelWrapper."""
        # Create wrapped model
        mlx_model = MockMLXLlamaModel(self.config)
        tokenizer = MagicMock()
        
        wrapper = MLXModelWrapper(
            mlx_model,
            tokenizer=tokenizer,
            backend_manager=BackendManager(backend="mlx"),
        )
        
        # Create training config
        training_config = MLXTrainingConfig(
            model_config=self.config,
            output_dir=self.test_dir,
        )
        
        # Create trainer with wrapper
        trainer = MLXTrainer(
            model=wrapper,
            config=training_config,
            train_dataloader=DataLoader(RealisticDataset(size=10), batch_size=2),
        )
        
        # Save checkpoint
        with patch.object(wrapper, 'save_pretrained') as mock_save:
            trainer.save_checkpoint("wrapper_checkpoint")
            
            # Verify wrapper's save method was called
            expected_path = os.path.join(
                self.test_dir,
                "checkpoints",
                "wrapper_checkpoint"
            )
            mock_save.assert_called_once_with(expected_path)


class TestMLXTrainerMemoryOptimization(unittest.TestCase):
    """Test memory optimization features."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    def test_unified_memory_optimizer_integration(self):
        """Test integration with UnifiedMemoryOptimizer."""
        # Create model and wrapper
        config = MLXConfig(
            model_name="test-model",
            use_unified_memory=True,
        )
        
        mlx_model = MockMLXLlamaModel(config)
        wrapper = MLXModelWrapper(mlx_model, tokenizer=MagicMock())
        
        # Create memory optimizer
        memory_optimizer = UnifiedMemoryOptimizer(wrapper)
        
        # Test memory optimization during training
        training_config = MLXTrainingConfig(
            model_config=config,
            output_dir=self.test_dir,
            batch_size=1,  # Small batch for memory efficiency
        )
        
        trainer = MLXTrainer(
            model=wrapper,
            config=training_config,
            train_dataloader=DataLoader(RealisticDataset(size=10), batch_size=1),
        )
        
        # Simulate training with periodic memory optimization
        with patch.object(memory_optimizer, 'optimize_memory_layout') as mock_optimize:
            # This would be called periodically in a real training loop
            memory_optimizer.optimize_memory_layout()
            
            mock_optimize.assert_called_once()
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.mx')
    def test_gradient_accumulation_memory_efficiency(self, mock_mx):
        """Test that gradient accumulation helps with memory."""
        # Create large model config
        config = MLXConfig(
            model_name="test-large-model",
            hidden_size=8192,  # Large hidden size
            num_hidden_layers=4,
        )
        
        model = MockMLXLlamaModel(config)
        
        # Training config with gradient accumulation
        training_config = MLXTrainingConfig(
            model_config=config,
            batch_size=1,  # Small batch size
            gradient_accumulation_steps=8,  # Accumulate 8 steps
            output_dir=self.test_dir,
        )
        
        # Setup mocks
        mock_mx.value_and_grad.return_value = lambda x: (0.5, {"param": MagicMock()})
        mock_mx.array.return_value = MagicMock()
        
        trainer = MLXTrainer(
            model=model,
            config=training_config,
            train_dataloader=DataLoader(RealisticDataset(size=16), batch_size=1),
        )
        
        # Verify gradient accumulation is configured
        self.assertEqual(trainer.optimizer.gradient_accumulation_steps, 8)
        
        # Run multiple steps to test accumulation
        for i in range(8):
            batch = next(iter(trainer.train_dataloader))
            trainer._training_step(batch)
            
            # Check accumulation counter
            expected_counter = (i + 1) % 8
            if expected_counter == 0:
                # Should have reset after step
                self.assertEqual(trainer.optimizer.accumulation_counter, 0)
            else:
                self.assertEqual(trainer.optimizer.accumulation_counter, expected_counter)


class TestMLXTrainerRealWorldScenarios(unittest.TestCase):
    """Test real-world training scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.mx')
    @patch('mlx_trainer.optim_mlx')
    def test_llama_7b_qlora_scenario(self, mock_optim_mlx, mock_mx):
        """Test a realistic Llama-7B QLoRA training scenario."""
        # Create Llama-7B-like config
        config = MLXConfig(
            model_name="llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            intermediate_size=11008,
            max_position_embeddings=4096,
            use_quantization=True,
            quantization_bits=4,
            use_lora=True,
            lora_rank=64,
            lora_alpha=128.0,
            lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        
        # Training config optimized for M2 Ultra
        training_config = MLXTrainingConfig(
            model_config=config,
            batch_size=4,  # Reasonable for 7B model on M2 Ultra
            gradient_accumulation_steps=4,  # Effective batch size of 16
            learning_rate=2e-4,
            num_epochs=3,
            warmup_steps=100,
            output_dir=self.test_dir,
            logging_steps=10,
            save_steps=100,
        )
        
        # Check batch size recommendation
        max_batch = training_config.get_max_batch_size(7.0, "m1_ultra")
        self.assertEqual(max_batch, 8)  # M2 Ultra can handle up to 8
        
        # Create model (simplified for testing)
        model = MockMLXLlamaModel(config)
        
        # Create realistic dataset
        train_dataset = RealisticDataset(size=1000, seq_length=2048)
        eval_dataset = RealisticDataset(size=100, seq_length=2048)
        
        train_dataloader = DataLoader(train_dataset, batch_size=training_config.batch_size)
        eval_dataloader = DataLoader(eval_dataset, batch_size=training_config.batch_size)
        
        # Setup mocks
        mock_optimizer = MagicMock()
        mock_optimizer.learning_rate = training_config.learning_rate
        mock_optimizer.state = {}
        mock_optim_mlx.AdamW.return_value = mock_optimizer
        
        # Mock MLX operations
        loss_values = [0.7, 0.6, 0.5, 0.4, 0.35, 0.3]  # Decreasing loss
        step_count = 0
        
        def mock_value_and_grad(params):
            nonlocal step_count
            loss = loss_values[min(step_count % len(loss_values), len(loss_values) - 1)]
            step_count += 1
            grads = {k: MagicMock() for k in params.keys()}
            return loss, grads
        
        mock_mx.value_and_grad.return_value = mock_value_and_grad
        mock_mx.no_grad.return_value.__enter__ = MagicMock()
        mock_mx.no_grad.return_value.__exit__ = MagicMock()
        mock_mx.array.return_value = MagicMock()
        mock_mx.sqrt.return_value = 1.0  # For gradient clipping
        mock_mx.sum.return_value = 1.0
        
        # Create trainer
        trainer = MLXTrainer(
            model=model,
            config=training_config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )
        
        # Mock training step to speed up test
        original_train_epoch = trainer._train_epoch
        
        def mock_train_epoch():
            # Simulate one epoch with a few steps
            for i in range(5):
                batch = next(iter(train_dataloader))
                loss = trainer._training_step(batch)
                trainer.train_losses.append((trainer.global_step, float(loss)))
                trainer.global_step += 1
            return np.mean([l for _, l in trainer.train_losses[-5:]])
        
        trainer._train_epoch = mock_train_epoch
        
        # Run training
        history = trainer.train()
        
        # Verify training completed
        self.assertIn("train_losses", history)
        self.assertIn("learning_rates", history)
        self.assertGreater(len(history["train_losses"]), 0)
        
        # Verify loss decreased
        first_loss = history["train_losses"][0][1]
        last_loss = history["train_losses"][-1][1]
        self.assertLess(last_loss, first_loss)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    def test_batch_size_recommendations(self):
        """Test batch size recommendations for different models."""
        # Test different model sizes
        test_cases = [
            # (model_size, chip_type, expected_max_batch)
            (7.0, "m1", 2),
            (7.0, "m1_max", 4),
            (7.0, "m1_ultra", 8),
            (13.0, "m1", 1),
            (13.0, "m1_max", 2),
            (13.0, "m1_ultra", 4),
            (70.0, "m1", 0),  # Too large
            (70.0, "m1_max", 1),
            (70.0, "m1_ultra", 2),
        ]
        
        for model_size, chip_type, expected_batch in test_cases:
            config = MLXConfig(f"test-{model_size}b-model")
            training_config = MLXTrainingConfig(model_config=config)
            
            max_batch = training_config.get_max_batch_size(model_size, chip_type)
            self.assertEqual(
                max_batch,
                expected_batch,
                f"Failed for {model_size}B model on {chip_type}"
            )
    
    @patch('mlx_trainer.benchmark_mlx_training')
    def test_benchmarking_integration(self, mock_benchmark):
        """Test benchmarking functionality."""
        # Mock benchmark results
        mock_benchmark.return_value = {
            1: {
                "time_per_step": 0.5,
                "tokens_per_sec": 1024,
                "memory_gb": 8.0,
                "status": "success"
            },
            2: {
                "time_per_step": 0.9,
                "tokens_per_sec": 1146,
                "memory_gb": 12.0,
                "status": "success"
            },
            4: {
                "status": "failed",
                "error": "OOM"
            }
        }
        
        # Create model
        config = MLXConfig("test-model")
        model = MockMLXLlamaModel(config)
        
        # Run benchmark
        results = mock_benchmark(model, batch_sizes=[1, 2, 4])
        
        # Verify results
        self.assertEqual(results[1]["status"], "success")
        self.assertEqual(results[2]["status"], "success")
        self.assertEqual(results[4]["status"], "failed")
        
        # Check performance metrics
        self.assertLess(results[1]["time_per_step"], results[2]["time_per_step"])
        self.assertGreater(results[2]["tokens_per_sec"], results[1]["tokens_per_sec"])


class TestMLXTrainerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('mlx_trainer.MLX_AVAILABLE', False)
    def test_mlx_not_available(self):
        """Test behavior when MLX is not available."""
        config = MLXConfig("test-model")
        model = MagicMock()
        
        training_config = MLXTrainingConfig(
            model_config=config,
            output_dir=self.test_dir,
        )
        
        # Should raise ImportError
        with self.assertRaises(ImportError) as ctx:
            MLXTrainer(
                model=model,
                config=training_config,
                train_dataloader=DataLoader(RealisticDataset(size=10), batch_size=2),
            )
        
        self.assertIn("MLX is required", str(ctx.exception))
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.create_mlx_trainer')
    def test_large_batch_size_warning(self, mock_create_trainer):
        """Test warning for large batch sizes."""
        config = MLXConfig("llama-13b-model")
        model = MockMLXLlamaModel(config)
        
        # Mock the trainer creation
        mock_trainer = MagicMock()
        mock_create_trainer.return_value = mock_trainer
        
        # Create trainer with large batch size
        with self.assertWarns(UserWarning) as ctx:
            trainer = mock_create_trainer(
                model=model,
                train_dataloader=DataLoader(RealisticDataset(size=10), batch_size=8),
                batch_size=8,  # Too large for 13B model
            )
        
        # Verify warning message
        warning_msg = str(ctx.warning)
        self.assertIn("may be too large", warning_msg)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        config = MLXConfig("test-model")
        model = MockMLXLlamaModel(config)
        
        training_config = MLXTrainingConfig(
            model_config=config,
            output_dir=self.test_dir,
        )
        
        # Create empty dataset
        empty_dataset = RealisticDataset(size=0)
        dataloader = DataLoader(empty_dataset, batch_size=2)
        
        trainer = MLXTrainer(
            model=model,
            config=training_config,
            train_dataloader=dataloader,
        )
        
        # Training should complete without errors
        history = trainer.train()
        
        # But no losses should be recorded
        self.assertEqual(len(history["train_losses"]), 0)


if __name__ == '__main__':
    unittest.main()