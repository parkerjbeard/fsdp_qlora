"""
Tests for MLX utilities module.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile
import numpy as np
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock MLX imports before importing mlx_utils
with patch.dict('sys.modules', {
    'mlx': MagicMock(),
    'mlx.core': MagicMock(),
    'mlx.nn': MagicMock(),
    'mlx.optimizers': MagicMock(),
    'mlx.utils': MagicMock(),
}):
    from src.backends.mlx.mlx_utils import (
        DatasetConverter,
        HuggingFaceDatasetConverter,
        MLXDataLoader,
        MLXTokenizer,
        MemoryStats,
        MemoryProfiler,
        PerformanceMetrics,
        PerformanceMonitor,
        estimate_model_size,
        get_optimal_batch_size,
        format_memory_size,
        check_mlx_device,
        create_mlx_dataloader,
        profile_mlx_operation,
    )


class TestDatasetConverter(unittest.TestCase):
    """Test dataset conversion utilities."""
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.mx')
    def test_torch_to_mlx(self, mock_mx):
        """Test PyTorch tensor to MLX array conversion."""
        # Create test tensor
        tensor = torch.randn(2, 3, 4)
        
        # Mock MLX array creation
        mock_array = MagicMock()
        mock_mx.array.return_value = mock_array
        
        # Convert
        result = DatasetConverter.torch_to_mlx(tensor)
        
        # Verify
        mock_mx.array.assert_called_once()
        call_args = mock_mx.array.call_args[0][0]
        np.testing.assert_array_equal(call_args, tensor.numpy())
        self.assertEqual(result, mock_array)
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.mx')
    def test_torch_to_mlx_different_dtypes(self, mock_mx):
        """Test conversion with different dtypes."""
        # Test float16
        tensor_f16 = torch.randn(2, 3).half()
        DatasetConverter.torch_to_mlx(tensor_f16)
        
        # Test bfloat16 (should convert to float32)
        tensor_bf16 = torch.randn(2, 3).bfloat16()
        DatasetConverter.torch_to_mlx(tensor_bf16)
        
        # Verify conversions
        self.assertEqual(mock_mx.array.call_count, 2)
    
    @patch('mlx_utils.MLX_AVAILABLE', False)
    def test_torch_to_mlx_no_mlx(self):
        """Test error when MLX is not available."""
        tensor = torch.randn(2, 3)
        
        with self.assertRaises(ImportError):
            DatasetConverter.torch_to_mlx(tensor)
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.mx')
    def test_numpy_to_mlx(self, mock_mx):
        """Test numpy array to MLX array conversion."""
        array = np.random.randn(2, 3, 4)
        
        mock_array = MagicMock()
        mock_mx.array.return_value = mock_array
        
        result = DatasetConverter.numpy_to_mlx(array)
        
        mock_mx.array.assert_called_once_with(array)
        self.assertEqual(result, mock_array)
    
    def test_mlx_to_torch(self):
        """Test MLX array to PyTorch tensor conversion."""
        # Mock MLX array
        mock_array = MagicMock()
        np_array = np.random.randn(2, 3)
        
        # Mock np.array() to return our numpy array
        with patch('numpy.array', return_value=np_array):
            result = DatasetConverter.mlx_to_torch(mock_array)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2, 3))
    
    def test_mlx_to_torch_with_device(self):
        """Test MLX to torch conversion with device specification."""
        mock_array = MagicMock()
        np_array = np.random.randn(2, 3)
        
        with patch('numpy.array', return_value=np_array):
            result = DatasetConverter.mlx_to_torch(mock_array, device="cpu")
        
        self.assertEqual(result.device.type, "cpu")
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.mx')
    def test_dict_to_mlx(self, mock_mx):
        """Test dictionary conversion to MLX format."""
        mock_mx.array.side_effect = lambda x: f"mlx_{x.tolist()}"
        
        data = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": np.array([1, 1, 1]),
            "labels": [2, 3, 4],
            "text": "hello world",
        }
        
        result = DatasetConverter.dict_to_mlx(data)
        
        # Check conversions
        self.assertTrue(result["input_ids"].startswith("mlx_"))
        self.assertTrue(result["attention_mask"].startswith("mlx_"))
        self.assertTrue(result["labels"].startswith("mlx_"))
        self.assertEqual(result["text"], "hello world")  # String unchanged


class TestHuggingFaceDatasetConverter(unittest.TestCase):
    """Test HuggingFace dataset converter."""
    
    def setUp(self):
        """Set up test tokenizer."""
        self.tokenizer = MagicMock()
        self.tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 0]]),
            "attention_mask": np.array([[1, 1, 1, 1, 0]]),
        }
        self.converter = HuggingFaceDatasetConverter(self.tokenizer)
    
    @patch('mlx_utils.mx')
    def test_convert_dataset(self, mock_mx):
        """Test dataset conversion."""
        mock_mx.array.side_effect = lambda x: f"mlx_array_{x.shape}"
        
        # Mock dataset
        dataset = [
            {"text": "Hello world"},
            {"text": "Fine-tuning with MLX"},
        ]
        
        result = self.converter.convert_dataset(dataset, max_length=5)
        
        # Check results
        self.assertEqual(len(result), 2)
        self.assertIn("input_ids", result[0])
        self.assertIn("attention_mask", result[0])
        self.assertIn("labels", result[0])
        
        # Verify tokenizer was called
        self.assertEqual(self.tokenizer.call_count, 2)
    
    def test_format_alpaca_prompt(self):
        """Test Alpaca prompt formatting."""
        # Test with input
        item = {
            "instruction": "Translate to French",
            "input": "Hello world",
            "output": "Bonjour le monde",
        }
        
        prompt = self.converter._format_alpaca_prompt(item)
        self.assertIn("### Instruction:", prompt)
        self.assertIn("### Input:", prompt)
        self.assertIn("### Response:", prompt)
        
        # Test without input
        item_no_input = {
            "instruction": "Tell me a joke",
            "output": "Why did the chicken cross the road?",
        }
        
        prompt_no_input = self.converter._format_alpaca_prompt(item_no_input)
        self.assertIn("### Instruction:", prompt_no_input)
        self.assertNotIn("### Input:", prompt_no_input)
    
    @patch('mlx_utils.mx')
    def test_convert_dataset_alpaca_format(self, mock_mx):
        """Test conversion with Alpaca-style dataset."""
        mock_mx.array.return_value = MagicMock()
        
        dataset = [
            {
                "instruction": "What is 2+2?",
                "input": "",
                "output": "4",
            }
        ]
        
        # Converter should detect non-text field and use Alpaca formatting
        with patch.object(self.converter, '_format_alpaca_prompt', return_value="formatted") as mock_format:
            result = self.converter.convert_dataset(dataset, text_field="text")
            mock_format.assert_called_once()


class TestMLXDataLoader(unittest.TestCase):
    """Test MLX DataLoader."""
    
    def setUp(self):
        """Create test data."""
        # Mock MLX arrays
        self.data = []
        for i in range(10):
            self.data.append({
                "input_ids": MagicMock(shape=(5,)),
                "labels": MagicMock(shape=(5,)),
            })
    
    @patch('mlx_utils.mx')
    def test_basic_iteration(self, mock_mx):
        """Test basic dataloader iteration."""
        mock_mx.stack.side_effect = lambda x: f"stacked_{len(x)}"
        
        dataloader = MLXDataLoader(self.data, batch_size=3, shuffle=False)
        
        batches = list(dataloader)
        
        # Check number of batches
        self.assertEqual(len(batches), 4)  # 10 samples / 3 batch_size = 4 batches
        
        # Check batch content
        first_batch = batches[0]
        self.assertIn("input_ids", first_batch)
        self.assertIn("labels", first_batch)
        self.assertEqual(first_batch["input_ids"], "stacked_3")
    
    def test_shuffle(self):
        """Test shuffling functionality."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        dataloader1 = MLXDataLoader(self.data, batch_size=2, shuffle=True)
        indices1 = dataloader1._indices.copy()
        
        dataloader2 = MLXDataLoader(self.data, batch_size=2, shuffle=True)
        indices2 = dataloader2._indices.copy()
        
        # Indices should be different (with high probability)
        self.assertFalse(np.array_equal(indices1, indices2))
    
    def test_drop_last(self):
        """Test drop_last functionality."""
        # With drop_last=True
        dataloader_drop = MLXDataLoader(self.data, batch_size=3, drop_last=True)
        self.assertEqual(len(dataloader_drop), 3)  # 10 / 3 = 3 full batches
        
        # With drop_last=False
        dataloader_keep = MLXDataLoader(self.data, batch_size=3, drop_last=False)
        self.assertEqual(len(dataloader_keep), 4)  # 10 / 3 = 3 full + 1 partial


class TestMLXTokenizer(unittest.TestCase):
    """Test MLX tokenizer wrapper."""
    
    def setUp(self):
        """Set up mock tokenizer."""
        self.hf_tokenizer = MagicMock()
        self.hf_tokenizer.pad_token = None
        self.hf_tokenizer.eos_token = "<eos>"
        
        self.mlx_tokenizer = MLXTokenizer(self.hf_tokenizer)
    
    def test_initialization(self):
        """Test tokenizer initialization."""
        # Pad token should be set to eos token
        self.assertEqual(self.hf_tokenizer.pad_token, "<eos>")
    
    @patch('mlx_utils.mx')
    def test_call_single_text(self, mock_mx):
        """Test tokenizing single text."""
        mock_mx.array.side_effect = lambda x: f"mlx_{x.shape}"
        
        self.hf_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3, 4]]),
            "attention_mask": np.array([[1, 1, 1, 1]]),
        }
        
        result = self.mlx_tokenizer("Hello world", max_length=128)
        
        # Check tokenizer was called correctly
        self.hf_tokenizer.assert_called_once_with(
            "Hello world",
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        
        # Check conversion to MLX
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)
        self.assertTrue(result["input_ids"].startswith("mlx_"))
    
    def test_decode(self):
        """Test decoding functionality."""
        # Test with MLX array
        mock_array = MagicMock()
        mock_array.tolist.return_value = [1, 2, 3]
        
        self.hf_tokenizer.decode.return_value = "Hello world"
        
        result = self.mlx_tokenizer.decode(mock_array)
        
        self.hf_tokenizer.decode.assert_called_once_with([1, 2, 3])
        self.assertEqual(result, "Hello world")
        
        # Test with list
        result2 = self.mlx_tokenizer.decode([4, 5, 6])
        self.hf_tokenizer.decode.assert_called_with([4, 5, 6])
    
    def test_batch_decode(self):
        """Test batch decoding."""
        mock_array = MagicMock()
        
        with patch('numpy.array') as mock_np_array:
            mock_np_array.return_value.tolist.return_value = [[1, 2], [3, 4]]
            
            self.hf_tokenizer.batch_decode.return_value = ["text1", "text2"]
            
            result = self.mlx_tokenizer.batch_decode(mock_array)
            
            self.assertEqual(result, ["text1", "text2"])


class TestMemoryProfiler(unittest.TestCase):
    """Test memory profiling utilities."""
    
    def test_memory_stats_creation(self):
        """Test MemoryStats dataclass."""
        stats = MemoryStats(
            process_rss_gb=4.5,
            process_vms_gb=8.0,
            process_percent=25.0,
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            used_memory_gb=8.0,
            free_memory_gb=8.0,
        )
        
        self.assertEqual(stats.process_rss_gb, 4.5)
        self.assertEqual(stats.process_percent, 25.0)
        
        # Test string representation
        str_repr = str(stats)
        self.assertIn("4.50 GB", str_repr)
        self.assertIn("25.0%", str_repr)
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_get_memory_stats(self, mock_vm, mock_process_class):
        """Test getting memory statistics."""
        # Mock process memory
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=4.5e9, vms=8e9)
        mock_process.memory_percent.return_value = 25.0
        mock_process_class.return_value = mock_process
        
        # Mock system memory
        mock_vm.return_value = MagicMock(
            total=16e9,
            available=8e9,
            used=8e9,
            free=8e9,
            wired=2e9,
            compressed=1e9,
        )
        
        profiler = MemoryProfiler()
        stats = profiler.get_memory_stats()
        
        self.assertAlmostEqual(stats.process_rss_gb, 4.5, places=1)
        self.assertAlmostEqual(stats.total_memory_gb, 16.0, places=1)
        self.assertAlmostEqual(stats.wired_memory_gb, 2.0, places=1)
    
    @patch('psutil.Process')
    def test_profile_context_manager(self, mock_process_class):
        """Test memory profiling context manager."""
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=4e9, vms=8e9)
        mock_process.memory_percent.return_value = 25.0
        mock_process_class.return_value = mock_process
        
        profiler = MemoryProfiler()
        
        with profiler.profile("Test Operation"):
            # Simulate memory increase
            mock_process.memory_info.return_value = MagicMock(rss=5e9, vms=9e9)
        
        # Check history
        self.assertEqual(len(profiler.history), 1)
        entry = profiler.history[0]
        self.assertEqual(entry["label"], "Test Operation")
        self.assertAlmostEqual(entry["memory_delta_gb"], 1.0, places=1)
    
    def test_get_summary(self):
        """Test profiling summary generation."""
        profiler = MemoryProfiler()
        
        # Add mock history
        profiler.history = [
            {
                "label": "Model Loading",
                "duration": 2.5,
                "start_memory_gb": 2.0,
                "end_memory_gb": 6.0,
                "memory_delta_gb": 4.0,
            }
        ]
        
        summary = profiler.get_summary()
        self.assertIn("Model Loading", summary)
        self.assertIn("2.50s", summary)
        self.assertIn("+4.00 GB", summary)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring tools."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics dataclass."""
        metrics = PerformanceMetrics(
            total_time=10.0,
            samples_per_second=100.0,
            tokens_per_second=51200.0,
            peak_memory_gb=8.0,
            compile_time=1.0,
            compute_time=9.0,
        )
        
        str_repr = str(metrics)
        self.assertIn("10.00s", str_repr)
        self.assertIn("100.0 samples/s", str_repr)
        self.assertIn("51200.0 tokens/s", str_repr)
        self.assertIn("10.0%", str_repr)  # Compile time percentage
    
    @patch('mlx_utils.MemoryProfiler')
    def test_benchmark_context_manager(self, mock_profiler_class):
        """Test performance benchmarking."""
        # Mock memory profiler
        mock_profiler = MagicMock()
        mock_profiler.history = [
            {"memory_gb": 4.0},
            {"memory_gb": 5.0},
            {"memory_gb": 4.5},
        ]
        mock_profiler_class.return_value = mock_profiler
        
        monitor = PerformanceMonitor()
        
        with monitor.benchmark(num_samples=100, num_tokens=51200, label="Training"):
            time.sleep(0.01)  # Simulate work
        
        # Check metrics
        metrics = monitor.get_latest_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.total_time, 0)
        self.assertGreater(metrics.samples_per_second, 0)
        self.assertEqual(metrics.peak_memory_gb, 5.0)
    
    def test_compile_time_tracking(self):
        """Test MLX compilation time tracking."""
        monitor = PerformanceMonitor()
        
        with monitor.benchmark(num_samples=10, label="Test"):
            monitor.mark_compile_start()
            time.sleep(0.01)
            monitor.mark_compile_end()
            time.sleep(0.01)
        
        metrics = monitor.get_latest_metrics()
        self.assertGreater(metrics.compile_time, 0)
        self.assertLess(metrics.compile_time, metrics.total_time)
    
    def test_compare_benchmarks(self):
        """Test benchmark comparison."""
        monitor = PerformanceMonitor()
        
        # Add mock benchmarks
        monitor.metrics_history = [
            {
                "label": "Method A",
                "metrics": PerformanceMetrics(
                    total_time=10.0,
                    samples_per_second=100.0,
                    tokens_per_second=51200.0,
                    peak_memory_gb=8.0,
                ),
            },
            {
                "label": "Method B",
                "metrics": PerformanceMetrics(
                    total_time=5.0,
                    samples_per_second=200.0,
                    tokens_per_second=102400.0,
                    peak_memory_gb=10.0,
                ),
            },
        ]
        
        comparison = monitor.compare_benchmarks()
        self.assertIn("Method A", comparison)
        self.assertIn("Method B", comparison)
        self.assertIn("100.0", comparison)  # Samples/s for Method A
        self.assertIn("200.0", comparison)  # Samples/s for Method B


class TestHelperUtilities(unittest.TestCase):
    """Test helper utility functions."""
    
    def test_estimate_model_size(self):
        """Test model size estimation."""
        # 7B model, 4-bit quantization
        size_gb = estimate_model_size(7e9, bits=4, include_gradients=False)
        self.assertAlmostEqual(size_gb, 3.5, places=1)  # 7B * 0.5 bytes / 1e9
        
        # With gradients (fp32)
        size_with_grad = estimate_model_size(7e9, bits=4, include_gradients=True)
        self.assertGreater(size_with_grad, size_gb)
        
        # With optimizer states
        size_with_opt = estimate_model_size(
            7e9, bits=4, include_gradients=True, include_optimizer_states=True
        )
        self.assertGreater(size_with_opt, size_with_grad)
    
    @patch('psutil.virtual_memory')
    def test_get_optimal_batch_size(self, mock_vm):
        """Test optimal batch size calculation."""
        mock_vm.return_value = MagicMock(available=32e9)  # 32GB available
        
        # 7B model
        batch_size = get_optimal_batch_size(7.0, sequence_length=2048)
        self.assertGreaterEqual(batch_size, 1)
        self.assertLessEqual(batch_size, 32)
        
        # With custom available memory
        batch_size_custom = get_optimal_batch_size(
            7.0, sequence_length=2048, available_memory_gb=16.0
        )
        self.assertLess(batch_size_custom, batch_size)
    
    def test_format_memory_size(self):
        """Test memory size formatting."""
        self.assertEqual(format_memory_size(512), "512.00 B")
        self.assertEqual(format_memory_size(1024), "1.00 KB")
        self.assertEqual(format_memory_size(1024 * 1024), "1.00 MB")
        self.assertEqual(format_memory_size(1024 * 1024 * 1024), "1.00 GB")
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('platform.platform')
    @patch('platform.processor')
    @patch('psutil.virtual_memory')
    def test_check_mlx_device(self, mock_vm, mock_processor, mock_platform):
        """Test MLX device checking."""
        mock_platform.return_value = "macOS-13.0"
        mock_processor.return_value = "Apple M2"
        mock_vm.return_value = MagicMock(total=32e9)
        
        info = check_mlx_device()
        
        self.assertTrue(info["available"])
        self.assertEqual(info["device"], "Apple Silicon")
        self.assertTrue(info["unified_memory"])
        self.assertEqual(info["chip_series"], "M2")
        self.assertAlmostEqual(info["total_memory_gb"], 32.0, places=1)
    
    @patch('mlx_utils.MLX_AVAILABLE', False)
    def test_check_mlx_device_not_available(self):
        """Test device check when MLX not available."""
        info = check_mlx_device()
        self.assertFalse(info["available"])
        self.assertIn("error", info)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.HuggingFaceDatasetConverter')
    def test_create_mlx_dataloader_hf_dataset(self, mock_converter_class):
        """Test creating dataloader from HuggingFace dataset."""
        # Mock dataset
        from datasets import Dataset as HFDataset
        dataset = MagicMock(spec=HFDataset)
        
        # Mock converter
        mock_converter = MagicMock()
        mock_converter.convert_dataset.return_value = [
            {"input_ids": MagicMock(), "labels": MagicMock()}
        ]
        mock_converter_class.return_value = mock_converter
        
        # Mock tokenizer
        tokenizer = MagicMock()
        
        # Create dataloader
        dataloader = create_mlx_dataloader(
            dataset,
            tokenizer,
            batch_size=4,
            max_length=512,
        )
        
        # Verify
        self.assertIsInstance(dataloader, MLXDataLoader)
        self.assertEqual(dataloader.batch_size, 4)
        mock_converter.convert_dataset.assert_called_once()
    
    def test_create_mlx_dataloader_list(self):
        """Test creating dataloader from list of dicts."""
        dataset = [
            {"input_ids": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([4, 5, 6])},
        ]
        
        tokenizer = MagicMock()
        
        with patch('mlx_utils.DatasetConverter.dict_to_mlx') as mock_convert:
            mock_convert.return_value = {"input_ids": MagicMock()}
            
            dataloader = create_mlx_dataloader(
                dataset,
                tokenizer,
                batch_size=2,
            )
            
            self.assertEqual(mock_convert.call_count, 2)
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.mx')
    def test_profile_mlx_operation(self, mock_mx):
        """Test MLX operation profiling."""
        mock_mx.eval.return_value = None
        
        def dummy_op(x, y):
            return x + y
        
        stats = profile_mlx_operation(
            dummy_op,
            1,
            2,
            num_warmup=2,
            num_runs=5,
        )
        
        # Check stats
        self.assertIn("mean_time", stats)
        self.assertIn("std_time", stats)
        self.assertIn("min_time", stats)
        self.assertIn("max_time", stats)
        self.assertIn("median_time", stats)
        
        # Verify warmup and runs
        self.assertEqual(mock_mx.eval.call_count, 5)  # Only timed runs call eval
    
    @patch('mlx_utils.MLX_AVAILABLE', False)
    def test_profile_mlx_operation_no_mlx(self):
        """Test profiling error when MLX not available."""
        def dummy_op():
            pass
        
        with self.assertRaises(ImportError):
            profile_mlx_operation(dummy_op)


class TestIntegration(unittest.TestCase):
    """Integration tests for MLX utilities."""
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    @patch('mlx_utils.mx')
    def test_end_to_end_dataset_conversion(self, mock_mx):
        """Test end-to-end dataset conversion and loading."""
        mock_mx.array.side_effect = lambda x: MagicMock(shape=x.shape)
        mock_mx.stack.side_effect = lambda x: MagicMock(shape=(len(x),) + x[0].shape)
        
        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5]]),
            "attention_mask": np.array([[1, 1, 1, 1, 0]]),
        }
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        
        # Create MLX tokenizer
        mlx_tokenizer = MLXTokenizer(tokenizer)
        
        # Create dataset
        texts = ["Hello world", "MLX is great", "Fine-tuning on Apple Silicon"]
        mlx_data = []
        
        for text in texts:
            tokens = mlx_tokenizer(text, max_length=5)
            # Add labels
            tokens["labels"] = tokens["input_ids"]
            mlx_data.append(tokens)
        
        # Create dataloader
        dataloader = MLXDataLoader(mlx_data, batch_size=2, shuffle=False)
        
        # Iterate through batches
        batches = list(dataloader)
        
        self.assertEqual(len(batches), 2)  # 3 samples / 2 batch_size
        
        # Check first batch
        first_batch = batches[0]
        self.assertIn("input_ids", first_batch)
        self.assertIn("attention_mask", first_batch)
        self.assertIn("labels", first_batch)
    
    @patch('mlx_utils.MLX_AVAILABLE', True)
    def test_memory_and_performance_monitoring(self):
        """Test combined memory and performance monitoring."""
        monitor = PerformanceMonitor()
        
        # Simulate a training-like workload
        with monitor.benchmark(num_samples=100, num_tokens=5000, label="Training Step"):
            # Simulate some work
            data = np.random.randn(100, 50)
            result = np.mean(data, axis=1)
            time.sleep(0.01)
        
        # Get results
        metrics = monitor.get_latest_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.samples_per_second, 0)
        
        # Check memory tracking
        self.assertGreater(metrics.peak_memory_gb, 0)
        
        # Get summary
        summary = monitor.get_summary()
        self.assertIn("Training Step", summary)


if __name__ == '__main__':
    unittest.main()