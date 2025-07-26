"""
Tests for the MLX trainer.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call
import tempfile
import shutil
import json
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend, BackendManager

# Mock MLX imports
with patch.dict('sys.modules', {
    'mlx': MagicMock(),
    'mlx.core': MagicMock(),
    'mlx.nn': MagicMock(),
    'mlx.optimizers': MagicMock(),
    'mlx.utils': MagicMock(),
}):
    from src.backends.mlx.mlx_model_wrapper import MLXConfig, MLXModel, MLXModelWrapper
    from src.backends.mlx.mlx_trainer import (
        MLXTrainingConfig,
        MLXOptimizer,
        MLXLossComputer,
        MLXTrainer,
        create_mlx_trainer,
        benchmark_mlx_training,
    )


class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, size=100, seq_length=128, vocab_size=1000):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_length,)),
            "attention_mask": torch.ones(self.seq_length),
            "labels": torch.randint(0, self.vocab_size, (self.seq_length,)),
        }


class TestMLXTrainingConfig(unittest.TestCase):
    """Test MLXTrainingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        model_config = MLXConfig("test-model")
        config = MLXTrainingConfig(model_config=model_config)
        
        self.assertEqual(config.learning_rate, 5e-4)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.gradient_accumulation_steps, 1)
        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.max_grad_norm, 1.0)
    
    def test_batch_size_limits(self):
        """Test batch size recommendations."""
        model_config = MLXConfig("test-model")
        config = MLXTrainingConfig(model_config=model_config)
        
        # Test M1 Ultra limits
        self.assertEqual(config.get_max_batch_size(7.0, "m1_ultra"), 8)
        self.assertEqual(config.get_max_batch_size(13.0, "m1_ultra"), 4)
        self.assertEqual(config.get_max_batch_size(70.0, "m1_ultra"), 2)
        
        # Test M1 Max limits
        self.assertEqual(config.get_max_batch_size(7.0, "m1_max"), 4)
        self.assertEqual(config.get_max_batch_size(13.0, "m1_max"), 2)
        self.assertEqual(config.get_max_batch_size(70.0, "m1_max"), 1)


class TestMLXOptimizer(unittest.TestCase):
    """Test MLXOptimizer with gradient accumulation."""
    
    def setUp(self):
        """Set up test optimizer."""
        # Mock MLX optimizer
        self.mock_mlx_optimizer = MagicMock()
        self.mock_mlx_optimizer.learning_rate = 5e-4
        self.mock_mlx_optimizer.state = {}
        
        self.optimizer = MLXOptimizer(
            self.mock_mlx_optimizer,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0
        )
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation logic."""
        # Mock gradients
        grads1 = {"param1": MagicMock(), "param2": MagicMock()}
        grads2 = {"param1": MagicMock(), "param2": MagicMock()}
        
        # First accumulation
        self.optimizer.accumulate_gradients(grads1)
        self.assertEqual(self.optimizer.accumulation_counter, 1)
        self.assertFalse(self.optimizer.step())
        
        # Second accumulation
        self.optimizer.accumulate_gradients(grads2)
        self.assertEqual(self.optimizer.accumulation_counter, 2)
        self.assertFalse(self.optimizer.step())
        
        # Third accumulation
        self.optimizer.accumulate_gradients(grads1)
        self.assertEqual(self.optimizer.accumulation_counter, 3)
        self.assertFalse(self.optimizer.step())
        
        # Fourth accumulation - should trigger step
        with patch('mlx_trainer.tree_map') as mock_tree_map:
            mock_tree_map.return_value = grads1
            self.optimizer.accumulate_gradients(grads2)
            self.assertEqual(self.optimizer.accumulation_counter, 4)
            self.assertTrue(self.optimizer.step())
            
            # Check optimizer update was called
            self.mock_mlx_optimizer.update.assert_called_once()
            
            # Check counter was reset
            self.assertEqual(self.optimizer.accumulation_counter, 0)
    
    @patch('mlx_trainer.mx')
    def test_gradient_clipping(self, mock_mx):
        """Test gradient clipping."""
        # Mock gradient norm computation
        mock_mx.sqrt.return_value = 2.0  # Norm > max_grad_norm
        mock_mx.sum.return_value = 4.0
        mock_mx.array.return_value = MagicMock()
        
        # Mock gradients
        grads = {"param1": MagicMock(), "param2": MagicMock()}
        
        # Clip gradients
        with patch('mlx_trainer.tree_flatten', return_value=[MagicMock(), MagicMock()]):
            with patch('mlx_trainer.tree_map') as mock_tree_map:
                clipped = self.optimizer._clip_gradients(grads)
                
                # Check that scaling was applied
                mock_tree_map.assert_called()


class TestMLXLossComputer(unittest.TestCase):
    """Test MLXLossComputer."""
    
    @patch('mlx_trainer.mx')
    def test_cross_entropy_loss(self, mock_mx):
        """Test cross-entropy loss computation."""
        # Mock MLX functions
        mock_mx.log_softmax.return_value = MagicMock(shape=(10, 1000))
        mock_mx.arange.return_value = MagicMock()
        mock_mx.sum.return_value = MagicMock()
        mock_mx.maximum.return_value = MagicMock()
        
        # Create mock logits and labels
        logits = MagicMock(shape=(2, 5, 1000))
        labels = MagicMock(shape=(2, 5))
        
        # Compute loss
        loss_computer = MLXLossComputer()
        loss = loss_computer.cross_entropy_loss(logits, labels)
        
        # Check that log_softmax was called
        mock_mx.log_softmax.assert_called_once()
    
    @patch('mlx_trainer.mx')
    def test_perplexity_computation(self, mock_mx):
        """Test perplexity computation."""
        mock_mx.exp.return_value = 10.0
        
        loss_computer = MLXLossComputer()
        loss = MagicMock()
        
        perplexity = loss_computer.compute_perplexity(loss)
        
        mock_mx.exp.assert_called_once_with(loss)
        self.assertEqual(perplexity, 10.0)


class TestMLXTrainer(unittest.TestCase):
    """Test MLXTrainer."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock model
        self.mock_model = MagicMock(spec=MLXModel)
        self.mock_model.config = MLXConfig("test-model")
        self.mock_model.parameters.return_value = {}
        
        # Create dummy dataset
        self.train_dataset = DummyDataset(size=20)
        self.eval_dataset = DummyDataset(size=10)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=2)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=2)
        
        # Create config
        self.config = MLXTrainingConfig(
            model_config=self.mock_model.config,
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=2,
            output_dir=self.test_dir,
            logging_steps=5,
            save_steps=10,
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.optim_mlx')
    def test_trainer_initialization(self, mock_optim_mlx):
        """Test trainer initialization."""
        # Mock optimizer
        mock_optimizer = MagicMock()
        mock_optim_mlx.AdamW.return_value = mock_optimizer
        
        trainer = MLXTrainer(
            model=self.mock_model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
        )
        
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.loss_computer)
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.current_epoch, 0)
        
        # Check directories were created
        self.assertTrue(os.path.exists(self.config.output_dir))
    
    @patch('mlx_trainer.MLX_AVAILABLE', False)
    def test_trainer_without_mlx(self):
        """Test trainer initialization without MLX."""
        with self.assertRaises(ImportError):
            MLXTrainer(
                model=self.mock_model,
                config=self.config,
                train_dataloader=self.train_dataloader,
            )
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.tree_flatten')
    def test_get_trainable_parameters(self, mock_tree_flatten):
        """Test getting trainable parameters."""
        # Mock LoRA parameters
        self.mock_model.layer1 = MagicMock()
        self.mock_model.layer1.lora_a = MagicMock()
        self.mock_model.layer1.lora_b = MagicMock()
        
        trainer = MLXTrainer(
            model=self.mock_model,
            config=self.config,
            train_dataloader=self.train_dataloader,
        )
        
        # Mock parameter collection
        with patch('mlx_trainer.mx.array'):
            params = trainer._get_trainable_parameters()
            
            # Should have collected LoRA parameters
            self.assertIn("lora_a", str(params.keys()) or "lora_b" in str(params.keys()))
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.mx')
    def test_prepare_batch(self, mock_mx):
        """Test batch preparation."""
        trainer = MLXTrainer(
            model=self.mock_model,
            config=self.config,
            train_dataloader=self.train_dataloader,
        )
        
        # Create PyTorch batch
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "labels": torch.randint(0, 1000, (2, 128)),
        }
        
        # Convert to MLX
        mock_mx.array.return_value = MagicMock()
        mlx_batch = trainer._prepare_batch(batch)
        
        # Check conversion
        self.assertEqual(len(mlx_batch), 2)
        self.assertIn("input_ids", mlx_batch)
        self.assertIn("labels", mlx_batch)
        
        # Check mx.array was called
        self.assertEqual(mock_mx.array.call_count, 2)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.mx')
    def test_training_step(self, mock_mx):
        """Test single training step."""
        # Mock MLX functions
        mock_mx.value_and_grad.return_value = lambda x: (MagicMock(), {"param": MagicMock()})
        
        trainer = MLXTrainer(
            model=self.mock_model,
            config=self.config,
            train_dataloader=self.train_dataloader,
        )
        
        # Create batch
        batch = next(iter(self.train_dataloader))
        
        # Perform training step
        loss = trainer._training_step(batch)
        
        # Check that gradients were computed
        mock_mx.value_and_grad.assert_called_once()
        
        # Check that optimizer accumulation was called
        self.assertEqual(trainer.optimizer.accumulation_counter, 1)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        trainer = MLXTrainer(
            model=self.mock_model,
            config=self.config,
            train_dataloader=self.train_dataloader,
        )
        
        # Update training state
        trainer.global_step = 100
        trainer.current_epoch = 2
        trainer.train_losses = [(50, 0.5), (100, 0.4)]
        
        # Save checkpoint
        with patch('mlx_trainer.mx.save') as mock_save:
            with patch('mlx_trainer.tree_flatten', return_value={}):
                trainer.save_checkpoint("test")
                
                # Check save was called
                mock_save.assert_called_once()
        
        # Check training state was saved
        state_path = os.path.join(
            self.config.output_dir,
            "checkpoints",
            "test",
            "training_state.json"
        )
        self.assertTrue(os.path.exists(state_path))
        
        # Load and verify state
        with open(state_path) as f:
            state = json.load(f)
        
        self.assertEqual(state["global_step"], 100)
        self.assertEqual(state["current_epoch"], 2)
        self.assertEqual(len(state["train_losses"]), 2)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    def test_learning_rate_schedule(self):
        """Test learning rate scheduling."""
        trainer = MLXTrainer(
            model=self.mock_model,
            config=self.config,
            train_dataloader=self.train_dataloader,
        )
        
        # Test warmup
        trainer.global_step = 50
        trainer.config.warmup_steps = 100
        trainer._update_learning_rate()
        
        # LR should be half of base LR
        expected_lr = self.config.learning_rate * 0.5
        self.assertAlmostEqual(
            trainer.optimizer.optimizer.learning_rate,
            expected_lr,
            places=6
        )
        
        # Test after warmup
        trainer.global_step = 150
        trainer._update_learning_rate()
        
        # LR should be decreasing (cosine decay)
        self.assertLess(
            trainer.optimizer.optimizer.learning_rate,
            self.config.learning_rate
        )


class TestMLXTrainerIntegration(unittest.TestCase):
    """Integration tests for MLX trainer."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    @patch('mlx_trainer.mx')
    @patch('mlx_trainer.optim_mlx')
    def test_full_training_loop(self, mock_optim_mlx, mock_mx):
        """Test complete training loop."""
        # Mock MLX components
        mock_mx.value_and_grad.return_value = lambda x: (0.5, {"param": MagicMock()})
        mock_mx.no_grad.return_value.__enter__ = MagicMock()
        mock_mx.no_grad.return_value.__exit__ = MagicMock()
        
        mock_optimizer = MagicMock()
        mock_optimizer.learning_rate = 5e-4
        mock_optim_mlx.AdamW.return_value = mock_optimizer
        
        # Create small dataset
        train_dataset = DummyDataset(size=10)
        train_dataloader = DataLoader(train_dataset, batch_size=2)
        
        # Create model and config
        model = MagicMock()
        model.config = MLXConfig("test-model")
        
        config = MLXTrainingConfig(
            model_config=model.config,
            num_epochs=1,
            batch_size=2,
            output_dir=self.test_dir,
            logging_steps=2,
            save_steps=5,
        )
        
        # Create trainer
        trainer = MLXTrainer(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
        )
        
        # Mock training step
        trainer._training_step = MagicMock(return_value=0.5)
        
        # Run training
        history = trainer.train()
        
        # Check training was called correct number of times
        expected_steps = len(train_dataloader)
        self.assertEqual(trainer._training_step.call_count, expected_steps)
        
        # Check history
        self.assertIn("train_losses", history)
        self.assertIn("learning_rates", history)
        
        # Check final checkpoint was saved
        final_checkpoint = os.path.join(
            self.test_dir,
            "checkpoints",
            "final"
        )
        self.assertTrue(os.path.exists(final_checkpoint))
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    def test_model_wrapper_integration(self):
        """Test integration with MLXModelWrapper."""
        # Create wrapper
        mock_mlx_model = MagicMock()
        mock_mlx_model.config = MLXConfig("test-model")
        
        wrapper = MLXModelWrapper(
            mock_mlx_model,
            tokenizer=MagicMock(),
            backend_manager=BackendManager(backend="mlx")
        )
        
        # Create trainer config
        config = MLXTrainingConfig(
            model_config=mock_mlx_model.config,
            output_dir=self.test_dir,
        )
        
        # Create dummy data
        train_dataloader = DataLoader(DummyDataset(size=10), batch_size=2)
        
        # Create trainer with wrapper
        trainer = MLXTrainer(
            model=wrapper,
            config=config,
            train_dataloader=train_dataloader,
        )
        
        # Check that wrapper methods are used
        self.assertIsInstance(trainer.model, MLXModelWrapper)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    @patch('mlx_trainer.MLX_AVAILABLE', True)
    def test_create_mlx_trainer(self):
        """Test trainer creation helper."""
        model = MagicMock()
        model.config = MLXConfig("test-model-13b")
        
        train_dataloader = DataLoader(DummyDataset(size=10), batch_size=8)
        
        # Create trainer with warning for large batch size
        with self.assertWarns(UserWarning):
            trainer = create_mlx_trainer(
                model=model,
                train_dataloader=train_dataloader,
                batch_size=8,  # Too large for 13B model
            )
        
        self.assertIsInstance(trainer, MLXTrainer)
    
    @patch('mlx_trainer.mx')
    def test_benchmark_mlx_training(self, mock_mx):
        """Test benchmarking function."""
        # Mock MLX operations
        mock_mx.random.randint.return_value = MagicMock()
        mock_mx.mean.return_value = MagicMock()
        mock_mx.grad.return_value = MagicMock()
        
        # Mock model
        model = MagicMock()
        model.return_value = MagicMock()
        
        # Run benchmark
        results = benchmark_mlx_training(
            model=model,
            batch_sizes=[1, 2],
            num_steps=2
        )
        
        # Check results
        self.assertIn(1, results)
        self.assertIn(2, results)
        
        for batch_size, metrics in results.items():
            if metrics["status"] == "success":
                self.assertIn("time_per_step", metrics)
                self.assertIn("tokens_per_sec", metrics)


if __name__ == '__main__':
    unittest.main()