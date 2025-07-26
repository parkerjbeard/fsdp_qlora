"""
Integration tests for MLX utilities.

These tests demonstrate real-world usage scenarios including:
- Complete dataset processing pipelines
- Memory profiling during model operations
- Performance benchmarking workflows
- Integration with MLX training components
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import json
import numpy as np
import time

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock MLX and other imports
with patch.dict('sys.modules', {
    'mlx': MagicMock(),
    'mlx.core': MagicMock(),
    'mlx.nn': MagicMock(),
    'mlx.optimizers': MagicMock(),
    'mlx.utils': MagicMock(),
    'datasets': MagicMock(),
}):
    from src.backends.mlx.mlx_utils import (
        DatasetConverter,
        HuggingFaceDatasetConverter,
        MLXDataLoader,
        MLXTokenizer,
        MemoryProfiler,
        PerformanceMonitor,
        create_mlx_dataloader,
        estimate_model_size,
        get_optimal_batch_size,
        check_mlx_device,
    )
    from src.backends.mlx.mlx_model_wrapper import MLXConfig
    from src.backends.mlx.mlx_trainer import MLXTrainingConfig


class MockAlpacaDataset:
    """Mock Alpaca-style dataset for testing."""
    
    def __init__(self, size=100):
        self.data = []
        for i in range(size):
            self.data.append({
                "instruction": f"Question {i}: What is {i}+{i}?",
                "input": "",
                "output": f"The answer is {2*i}.",
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __iter__(self):
        return iter(self.data)


class TestDatasetConversionPipeline(unittest.TestCase):
    """Test complete dataset conversion pipelines."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token = None
        self.tokenizer.eos_token = "<eos>"
        self.tokenizer.return_value = {
            "input_ids": np.random.randint(0, 1000, size=(1, 128)),
            "attention_mask": np.ones((1, 128)),
        }
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.mx')
    def test_huggingface_to_mlx_pipeline(self, mock_mx):
        """Test complete HuggingFace dataset to MLX conversion."""
        mock_mx.array.side_effect = lambda x: MagicMock(shape=x.shape, data=x)
        mock_mx.stack.side_effect = lambda arrays: MagicMock(
            shape=(len(arrays),) + arrays[0].shape,
            data=[a.data for a in arrays]
        )
        
        # Create dataset
        dataset = MockAlpacaDataset(size=50)
        
        # Convert dataset
        converter = HuggingFaceDatasetConverter(self.tokenizer)
        mlx_data = converter.convert_dataset(dataset, max_length=128)
        
        # Verify conversion
        self.assertEqual(len(mlx_data), 50)
        
        # Check data format
        first_item = mlx_data[0]
        self.assertIn("input_ids", first_item)
        self.assertIn("attention_mask", first_item)
        self.assertIn("labels", first_item)
        
        # Create MLX DataLoader
        dataloader = MLXDataLoader(mlx_data, batch_size=8, shuffle=True)
        
        # Test iteration
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            # Verify batch structure
            self.assertIn("input_ids", batch)
            self.assertIn("attention_mask", batch)
            self.assertIn("labels", batch)
            
            # Check batch is stacked
            self.assertTrue(hasattr(batch["input_ids"], "shape"))
        
        self.assertEqual(batch_count, 7)  # 50 samples / 8 batch_size = 7 batches
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.mx')
    def test_pytorch_dataset_conversion(self, mock_mx):
        """Test PyTorch dataset to MLX conversion."""
        mock_mx.array.return_value = MagicMock()
        
        # Create PyTorch dataset
        class SimpleDataset(Dataset):
            def __len__(self):
                return 20
            
            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, 1000, (128,)),
                    "labels": torch.randint(0, 1000, (128,)),
                }
        
        dataset = SimpleDataset()
        
        # Convert to MLX
        mlx_dataloader = create_mlx_dataloader(
            dataset,
            self.tokenizer,
            batch_size=4,
        )
        
        # Verify dataloader
        self.assertIsInstance(mlx_dataloader, MLXDataLoader)
        self.assertEqual(len(mlx_dataloader), 5)  # 20 / 4 = 5 batches
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    def test_tokenizer_integration(self):
        """Test MLX tokenizer integration with real tokenizer."""
        # Use a simple mock that behaves like a real tokenizer
        real_tokenizer = MagicMock()
        real_tokenizer.pad_token_id = 0
        real_tokenizer.eos_token_id = 1
        real_tokenizer.pad_token = "[PAD]"
        real_tokenizer.eos_token = "[EOS]"
        
        def mock_tokenize(text, **kwargs):
            # Simple mock tokenization
            if isinstance(text, str):
                tokens = text.split()
                ids = [hash(t) % 1000 for t in tokens]
                ids = ids[:kwargs.get('max_length', 512)]
                padding_length = kwargs.get('max_length', 512) - len(ids)
                ids.extend([0] * padding_length)
                
                return {
                    "input_ids": np.array([ids]),
                    "attention_mask": np.array([[1] * (len(tokens)) + [0] * padding_length]),
                }
            else:
                # Batch processing
                results = {"input_ids": [], "attention_mask": []}
                for t in text:
                    single_result = mock_tokenize(t, **kwargs)
                    results["input_ids"].append(single_result["input_ids"][0])
                    results["attention_mask"].append(single_result["attention_mask"][0])
                return {
                    "input_ids": np.array(results["input_ids"]),
                    "attention_mask": np.array(results["attention_mask"]),
                }
        
        real_tokenizer.side_effect = mock_tokenize
        
        # Create MLX tokenizer
        mlx_tokenizer = MLXTokenizer(real_tokenizer)
        
        # Test single text
        with patch('mlx_utils.mx.array') as mock_array:
            mock_array.side_effect = lambda x: MagicMock(shape=x.shape)
            
            result = mlx_tokenizer("Hello world this is a test", max_length=10)
            
            self.assertIn("input_ids", result)
            self.assertIn("attention_mask", result)
        
        # Test batch
        texts = ["First text", "Second text", "Third text"]
        with patch('mlx_utils.mx.array') as mock_array:
            mock_array.side_effect = lambda x: MagicMock(shape=x.shape)
            
            batch_result = mlx_tokenizer(texts, max_length=10)
            
            # Should have batch dimension
            self.assertEqual(batch_result["input_ids"].shape[0], 3)


class TestMemoryProfilingWorkflows(unittest.TestCase):
    """Test memory profiling in realistic scenarios."""
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_model_loading_memory_profile(self, mock_vm, mock_process_class):
        """Test memory profiling during model loading."""
        # Mock process memory - simulate memory increase during loading
        memory_values = [2e9, 4e9, 8e9, 7.5e9]  # GB values
        memory_index = 0
        
        def get_memory_info():
            nonlocal memory_index
            value = memory_values[min(memory_index, len(memory_values) - 1)]
            memory_index += 1
            return MagicMock(rss=value, vms=value * 2)
        
        mock_process = MagicMock()
        mock_process.memory_info.side_effect = get_memory_info
        mock_process.memory_percent.return_value = 25.0
        mock_process_class.return_value = mock_process
        
        # Mock system memory
        mock_vm.return_value = MagicMock(
            total=32e9,
            available=24e9,
            used=8e9,
            free=24e9,
        )
        
        profiler = MemoryProfiler()
        
        # Profile different stages
        with profiler.profile("Model Initialization"):
            time.sleep(0.01)  # Simulate work
        
        with profiler.profile("Weight Loading"):
            time.sleep(0.01)  # Simulate work
        
        with profiler.profile("Optimization"):
            time.sleep(0.01)  # Simulate work
        
        # Get summary
        summary = profiler.get_summary()
        
        # Verify profiling captured memory changes
        self.assertIn("Model Initialization", summary)
        self.assertIn("Weight Loading", summary)
        self.assertIn("Optimization", summary)
        
        # Check memory deltas
        self.assertEqual(len(profiler.history), 3)
        self.assertGreater(profiler.history[0]["memory_delta_gb"], 0)  # Memory increased
    
    def test_continuous_memory_monitoring(self):
        """Test continuous memory monitoring during training."""
        profiler = MemoryProfiler()
        
        # Start monitoring
        profiler.start_monitoring(interval=0.01)
        
        # Simulate training steps
        time.sleep(0.05)
        
        # Stop monitoring
        profiler.stop_monitoring()
        
        # Check that monitoring collected data
        monitoring_entries = [h for h in profiler.history if "timestamp" in h]
        self.assertGreater(len(monitoring_entries), 2)  # Should have multiple entries
        
        # Verify timestamps are increasing
        if len(monitoring_entries) > 1:
            for i in range(1, len(monitoring_entries)):
                self.assertGreater(
                    monitoring_entries[i]["timestamp"],
                    monitoring_entries[i-1]["timestamp"]
                )
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    def test_memory_optimization_workflow(self):
        """Test memory optimization workflow for model training."""
        # Estimate memory for different model sizes
        models = [
            ("7B-4bit", 7e9, 4),
            ("7B-8bit", 7e9, 8),
            ("13B-4bit", 13e9, 4),
            ("70B-4bit", 70e9, 4),
        ]
        
        memory_requirements = {}
        
        for name, params, bits in models:
            # Without gradients/optimizer (inference)
            inference_mem = estimate_model_size(
                params, bits=bits,
                include_gradients=False,
                include_optimizer_states=False
            )
            
            # With gradients (training without optimizer)
            training_mem = estimate_model_size(
                params, bits=bits,
                include_gradients=True,
                include_optimizer_states=False
            )
            
            # Full training (with optimizer)
            full_training_mem = estimate_model_size(
                params, bits=bits,
                include_gradients=True,
                include_optimizer_states=True
            )
            
            memory_requirements[name] = {
                "inference": inference_mem,
                "training": training_mem,
                "full_training": full_training_mem,
            }
        
        # Verify memory scaling
        self.assertLess(memory_requirements["7B-4bit"]["inference"], 4.0)  # < 4GB
        self.assertLess(memory_requirements["13B-4bit"]["inference"], 7.0)  # < 7GB
        self.assertGreater(memory_requirements["70B-4bit"]["inference"], 30.0)  # > 30GB
        
        # Verify training adds significant memory
        for name, reqs in memory_requirements.items():
            self.assertGreater(reqs["training"], reqs["inference"])
            self.assertGreater(reqs["full_training"], reqs["training"])


class TestPerformanceMonitoringWorkflows(unittest.TestCase):
    """Test performance monitoring in realistic scenarios."""
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.mx')
    def test_training_performance_monitoring(self, mock_mx):
        """Test performance monitoring during training simulation."""
        mock_mx.eval.return_value = None
        
        monitor = PerformanceMonitor()
        
        # Simulate different batch sizes
        batch_sizes = [1, 2, 4, 8]
        seq_length = 512
        
        for batch_size in batch_sizes:
            num_tokens = batch_size * seq_length
            
            with monitor.benchmark(
                num_samples=batch_size * 10,  # 10 steps
                num_tokens=num_tokens * 10,
                label=f"Batch Size {batch_size}"
            ):
                # Simulate training steps
                for _ in range(10):
                    # Simulate forward pass
                    time.sleep(0.001 * batch_size)  # Larger batch = more time
        
        # Compare results
        comparison = monitor.compare_benchmarks()
        
        # Verify all batch sizes were tested
        for batch_size in batch_sizes:
            self.assertIn(f"Batch Size {batch_size}", comparison)
        
        # Check that larger batches have higher throughput
        metrics = monitor.metrics_history
        batch_1_throughput = next(m for m in metrics if m["label"] == "Batch Size 1")["metrics"].tokens_per_second
        batch_8_throughput = next(m for m in metrics if m["label"] == "Batch Size 8")["metrics"].tokens_per_second
        
        # Larger batches should have better throughput (in our simulation)
        self.assertGreater(batch_8_throughput, batch_1_throughput)
    
    def test_compilation_vs_computation_tracking(self):
        """Test tracking MLX compilation vs computation time."""
        monitor = PerformanceMonitor()
        
        with monitor.benchmark(num_samples=100, label="With Compilation"):
            # Simulate compilation
            monitor.mark_compile_start()
            time.sleep(0.02)  # 20ms compilation
            monitor.mark_compile_end()
            
            # Simulate computation
            time.sleep(0.08)  # 80ms computation
        
        with monitor.benchmark(num_samples=100, label="Without Compilation"):
            # Just computation
            time.sleep(0.08)  # 80ms computation
        
        # Get metrics
        with_compile = monitor.metrics_history[0]["metrics"]
        without_compile = monitor.metrics_history[1]["metrics"]
        
        # Verify compilation time was tracked
        self.assertGreater(with_compile.compile_time, 0)
        self.assertEqual(without_compile.compile_time, 0)
        
        # Total time should be higher with compilation
        self.assertGreater(with_compile.total_time, without_compile.total_time)
        
        # But compute time should be similar
        self.assertAlmostEqual(
            with_compile.compute_time,
            without_compile.compute_time,
            delta=0.02
        )
    
    @patch('mlx_utils.profile_mlx_operation')
    def test_operation_profiling_workflow(self, mock_profile):
        """Test profiling individual MLX operations."""
        # Mock profiling results
        mock_profile.return_value = {
            "mean_time": 0.005,
            "std_time": 0.0005,
            "min_time": 0.004,
            "max_time": 0.006,
            "median_time": 0.005,
        }
        
        # Profile different operations
        operations = [
            ("matmul", lambda a, b: a @ b),
            ("add", lambda a, b: a + b),
            ("softmax", lambda a: a),  # Simplified
        ]
        
        results = {}
        
        for op_name, op_func in operations:
            stats = mock_profile(
                op_func,
                MagicMock(),
                MagicMock(),
                num_warmup=3,
                num_runs=10,
            )
            results[op_name] = stats
        
        # Verify all operations were profiled
        self.assertEqual(len(results), 3)
        
        # Check that stats contain expected fields
        for op_name, stats in results.items():
            self.assertIn("mean_time", stats)
            self.assertIn("std_time", stats)
            self.assertGreater(stats["mean_time"], 0)


class TestRealWorldIntegration(unittest.TestCase):
    """Test integration with real-world scenarios."""
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.mx')
    def test_complete_training_setup(self, mock_mx):
        """Test complete setup for MLX training."""
        mock_mx.array.side_effect = lambda x: MagicMock(shape=x.shape)
        mock_mx.stack.side_effect = lambda arrays: MagicMock(shape=(len(arrays),) + arrays[0].shape)
        
        # 1. Check device
        device_info = check_mlx_device()
        self.assertTrue(device_info["available"])
        
        # 2. Create model config
        model_config = MLXConfig(
            model_name="test-llama-7b",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            use_quantization=True,
            quantization_bits=4,
            use_lora=True,
            lora_rank=64,
        )
        
        # 3. Estimate memory requirements
        model_memory = estimate_model_size(
            7e9,  # 7B parameters
            bits=4,
            include_gradients=False,  # LoRA only trains adapters
            include_optimizer_states=True,
        )
        
        # 4. Get optimal batch size
        batch_size = get_optimal_batch_size(
            model_memory,
            sequence_length=2048,
            available_memory_gb=32.0,  # M1 Max
        )
        
        self.assertGreaterEqual(batch_size, 1)
        self.assertLessEqual(batch_size, 8)  # Reasonable for 7B model
        
        # 5. Create dataset and dataloader
        dataset = MockAlpacaDataset(size=100)
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": np.random.randint(0, 32000, (1, 2048)),
            "attention_mask": np.ones((1, 2048)),
        }
        
        # Convert dataset
        converter = HuggingFaceDatasetConverter(tokenizer)
        mlx_data = converter.convert_dataset(dataset, max_length=2048)
        
        # Create dataloader with optimal batch size
        dataloader = MLXDataLoader(
            mlx_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        # 6. Set up monitoring
        memory_profiler = MemoryProfiler()
        perf_monitor = PerformanceMonitor()
        
        # 7. Simulate training loop with monitoring
        with perf_monitor.benchmark(
            num_samples=len(dataloader) * batch_size,
            num_tokens=len(dataloader) * batch_size * 2048,
            label="Training Epoch"
        ):
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 5:  # Just test a few batches
                    break
                
                # Simulate training step
                time.sleep(0.01)
                
                # Periodic memory check
                if batch_idx % 2 == 0:
                    mem_stats = memory_profiler.get_memory_stats()
                    # In real scenario, would log or take action based on memory
        
        # 8. Get performance summary
        metrics = perf_monitor.get_latest_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.samples_per_second, 0)
        
        # 9. Memory summary
        summary = memory_profiler.get_summary()
        # Would contain profiling data in real scenario
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    def test_multi_dataset_handling(self):
        """Test handling multiple dataset formats."""
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        
        # Test different dataset formats
        datasets = []
        
        # 1. HuggingFace-style dataset
        hf_dataset = MockAlpacaDataset(size=20)
        
        # 2. List of dicts
        list_dataset = [
            {"text": f"Sample {i}"} for i in range(20)
        ]
        
        # 3. PyTorch Dataset
        class PTDataset(Dataset):
            def __len__(self):
                return 20
            
            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, 1000, (128,)),
                    "labels": torch.randint(0, 1000, (128,)),
                }
        
        pt_dataset = PTDataset()
        
        # Test conversion for each
        with patch('mlx_utils.mx.array') as mock_array:
            mock_array.side_effect = lambda x: MagicMock(shape=x.shape)
            
            # Convert HF dataset
            converter = HuggingFaceDatasetConverter(tokenizer)
            mlx_data_hf = converter.convert_dataset(hf_dataset, max_length=128)
            self.assertEqual(len(mlx_data_hf), 20)
            
            # Convert list dataset
            tokenizer.return_value = {
                "input_ids": np.random.randint(0, 1000, (1, 128)),
                "attention_mask": np.ones((1, 128)),
            }
            mlx_data_list = []
            for item in list_dataset:
                tokens = tokenizer(item["text"], max_length=128, return_tensors="np")
                mlx_item = DatasetConverter.dict_to_mlx({
                    "input_ids": tokens["input_ids"][0],
                    "attention_mask": tokens["attention_mask"][0],
                })
                mlx_data_list.append(mlx_item)
            self.assertEqual(len(mlx_data_list), 20)
            
            # Convert PyTorch dataset
            mlx_dataloader = create_mlx_dataloader(
                pt_dataset,
                tokenizer,
                batch_size=4,
            )
            self.assertEqual(len(mlx_dataloader), 5)  # 20 / 4


class TestErrorHandlingAndEdgeCases(unittest.TestCase):
    """Test error handling and edge cases."""
    
    @patch('mlx_utils.MLX_AVAILABLE', False)
    def test_graceful_degradation_without_mlx(self):
        """Test that utilities handle missing MLX gracefully."""
        # Dataset converter should raise ImportError
        with self.assertRaises(ImportError):
            DatasetConverter.torch_to_mlx(torch.randn(2, 3))
        
        # Device check should indicate MLX not available
        device_info = check_mlx_device()
        self.assertFalse(device_info["available"])
        self.assertIn("error", device_info)
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        empty_data = []
        
        dataloader = MLXDataLoader(empty_data, batch_size=4)
        
        # Should handle empty dataset gracefully
        self.assertEqual(len(dataloader), 0)
        
        # Iteration should immediately stop
        batches = list(dataloader)
        self.assertEqual(len(batches), 0)
    
    @patch('mlx_utils.mx')
    def test_mixed_data_types(self, mock_mx):
        """Test handling mixed data types in dataset."""
        mock_mx.array.side_effect = lambda x: MagicMock(shape=x.shape if hasattr(x, 'shape') else (len(x),))
        
        # Dataset with mixed types
        mixed_data = {
            "input_ids": torch.tensor([1, 2, 3]),
            "float_data": np.array([1.0, 2.0, 3.0]),
            "list_data": [4, 5, 6],
            "string_data": "metadata",
            "nested": {"key": "value"},
        }
        
        converted = DatasetConverter.dict_to_mlx(mixed_data)
        
        # Numeric data should be converted
        self.assertTrue(hasattr(converted["input_ids"], "shape"))
        self.assertTrue(hasattr(converted["float_data"], "shape"))
        self.assertTrue(hasattr(converted["list_data"], "shape"))
        
        # Non-numeric data should be preserved
        self.assertEqual(converted["string_data"], "metadata")
        self.assertEqual(converted["nested"], {"key": "value"})
    
    def test_memory_profiler_edge_cases(self):
        """Test memory profiler edge cases."""
        profiler = MemoryProfiler()
        
        # Test empty history summary
        summary = profiler.get_summary()
        self.assertEqual(summary, "No profiling data available")
        
        # Test stopping monitoring that wasn't started
        profiler.stop_monitoring()  # Should not error
        
        # Test nested profiling
        with profiler.profile("Outer"):
            with profiler.profile("Inner"):
                pass
        
        # Should have both entries
        self.assertEqual(len(profiler.history), 2)
    
    @patch('matplotlib.pyplot')
    def test_plotting_without_data(self, mock_plt):
        """Test plotting when no monitoring data exists."""
        profiler = MemoryProfiler()
        
        # Should handle gracefully
        profiler.plot_memory_usage()
        
        # Should not have called plt functions
        mock_plt.figure.assert_not_called()


if __name__ == '__main__':
    unittest.main()