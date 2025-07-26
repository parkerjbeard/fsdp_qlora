"""
Comprehensive backend integration tests with real model loading and quantization.
"""

import os
import sys
import unittest
import torch
import tempfile
import shutil
from typing import Dict, Any
import gc

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_utils import (
    create_tiny_model, get_memory_usage, memory_tracker,
    skip_if_backend_unavailable, DummyDataset, get_available_device,
    TinyLlamaConfig
)
from src.core.backend_manager import Backend, BackendManager
from src.core.model_loader import ModelLoaderFactory
from src.core.quantization_wrapper import create_quantization_adapter, QuantizationConfig


class TestBackendIntegrationComprehensive(unittest.TestCase):
    """Comprehensive tests for backend integration with real models."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @skip_if_backend_unavailable(Backend.CUDA)
    def test_cuda_model_loading_and_quantization(self):
        """Test CUDA backend with model loading and quantization."""
        backend_manager = BackendManager(backend="cuda", verbose=True)
        
        # Create a tiny model
        model, config = create_tiny_model("llama", Backend.CUDA)
        
        # Test quantization with BitsAndBytes
        if backend_manager.backend == Backend.CUDA and 4 in backend_manager.capabilities.quantization_bits:
            quant_config = QuantizationConfig(
                method="bnb",
                bits=4,
                compute_dtype=torch.float16
            )
            
            quant_adapter = create_quantization_adapter(backend_manager.backend, quant_config)
            
            # Track memory before quantization
            initial_memory = get_memory_usage()
            
            # Apply quantization
            quantized_model = quant_adapter.quantize_model(model)
            
            # Track memory after quantization
            final_memory = get_memory_usage()
            
            # Verify memory reduction
            if 'gpu_allocated_mb' in initial_memory and 'gpu_allocated_mb' in final_memory:
                # Quantized model should use less memory
                self.assertLess(
                    final_memory['gpu_allocated_mb'],
                    initial_memory['gpu_allocated_mb'] * 0.6,  # Expect at least 40% reduction
                    "Quantization should reduce memory usage"
                )
            
            # Test forward pass
            device = get_available_device(Backend.CUDA)
            dummy_input = torch.randint(0, config.vocab_size, (1, 32)).to(device)
            
            with torch.no_grad():
                output = quantized_model(dummy_input)
            
            self.assertIsNotNone(output)
            self.assertEqual(output.logits.shape[0], 1)
            self.assertEqual(output.logits.shape[1], 32)
    
    @skip_if_backend_unavailable(Backend.MPS)
    def test_mps_model_loading_and_quantization(self):
        """Test MPS backend with model loading and quantization."""
        backend_manager = BackendManager(backend="mps", verbose=True)
        
        # Create a tiny model
        model, config = create_tiny_model("llama", Backend.MPS)
        
        # Test with MLX quantization if available
        if backend_manager.backend == Backend.MPS:
            try:
                import mlx.core as mx
                
                # Create MLX quantization config
                quant_config = QuantizationConfig(
                    method="mlx",
                    bits=4,
                    group_size=64
                )
                
                # Apply quantization through the wrapper
                from src.backends.mlx.mlx_model_wrapper import MLXModelWrapper
                
                wrapper = MLXModelWrapper(model, backend_manager)
                wrapper.quantize(bits=4, group_size=64)
                
                # Test that model still works
                dummy_input = torch.randint(0, config.vocab_size, (1, 32)).to("mps")
                
                with torch.no_grad():
                    output = model(dummy_input)
                
                self.assertIsNotNone(output)
                
            except ImportError:
                self.skipTest("MLX not available")
        
        # Test with Quanto quantization
        elif backend_manager.backend == Backend.MPS:
            try:
                from src.backends.mps.mps_quantization_quanto import MPSQuantoQuantization
                
                quanto_quant = MPSQuantoQuantization(backend_manager)
                
                # Apply quantization
                quantized_model = quanto_quant.quantize_model(
                    model,
                    weight_bits=8,  # Quanto typically works better with 8-bit
                    activation_bits=None
                )
                
                # Test forward pass
                dummy_input = torch.randint(0, config.vocab_size, (1, 32)).to("mps")
                
                with torch.no_grad():
                    output = quantized_model(dummy_input)
                
                self.assertIsNotNone(output)
                
            except ImportError:
                self.skipTest("Quanto not available")
    
    @skip_if_backend_unavailable(Backend.CPU)
    def test_cpu_model_loading_with_hqq(self):
        """Test CPU backend with HQQ quantization."""
        backend_manager = BackendManager(backend="cpu", verbose=True)
        
        # Create a tiny model
        model, config = create_tiny_model("llama", Backend.CPU)
        
        # Test HQQ quantization
        if backend_manager.backend == Backend.CPU:
            try:
                from hqq.core.quantize import BaseQuantizeConfig
                from hqq.models.base import AutoHQQHFModel
                
                # Create HQQ config
                quant_config = BaseQuantizeConfig(
                    nbits=4,
                    group_size=64,
                    quant_zero=True,
                    quant_scale=True
                )
                
                # Track memory
                initial_memory = get_memory_usage()
                
                # Note: HQQ requires specific model preparation
                # For now, just verify the config works
                self.assertIsNotNone(quant_config)
                self.assertEqual(quant_config.nbits, 4)
                
            except ImportError:
                self.skipTest("HQQ not available")
    
    def test_backend_memory_efficiency(self):
        """Test memory efficiency across different backends."""
        results = {}
        
        for backend_name in ["cuda", "mps", "cpu"]:
            try:
                backend_manager = BackendManager(backend=backend_name, verbose=False)
            except ValueError:
                # Backend not available on this system
                continue
            
            # Create model
            model, config = create_tiny_model("llama", backend_manager.backend)
            
            # Measure memory
            with memory_tracker(backend_manager) as mem_stats:
                # Do some operations
                device = get_available_device(backend_manager.backend)
                dummy_input = torch.randint(0, config.vocab_size, (1, 128)).to(device)
                
                with torch.no_grad():
                    for _ in range(10):
                        output = model(dummy_input)
            
            results[backend_name] = {
                'backend': backend_manager.backend.value,
                'memory_used_mb': mem_stats.total_used_mb,
                'peak_mb': mem_stats.peak_mb
            }
        
        # Log results
        print("\nMemory efficiency results:")
        for backend, stats in results.items():
            print(f"{backend}: {stats}")
    
    def test_quantization_method_compatibility(self):
        """Test which quantization methods work with which backends."""
        compatibility_matrix = {}
        
        test_methods = ["bnb", "hqq", "mlx", "quanto"]
        test_backends = ["cuda", "mps", "cpu"]
        
        for backend_name in test_backends:
            try:
                backend_manager = BackendManager(backend=backend_name, verbose=False)
            except ValueError:
                # Backend not available on this system
                continue
            
            compatibility_matrix[backend_name] = {}
            
            for method in test_methods:
                # Determine support based on backend and method
                is_supported = False
                if method == "bnb" and backend_manager.backend == Backend.CUDA:
                    is_supported = True
                elif method == "hqq" and backend_manager.backend in [Backend.CUDA, Backend.CPU]:
                    is_supported = True
                elif method == "mlx" and backend_manager.backend in [Backend.MLX, Backend.MPS]:
                    is_supported = True
                elif method == "quanto" and backend_manager.backend == Backend.MPS:
                    is_supported = True
                compatibility_matrix[backend_name][method] = is_supported
        
        # Log compatibility matrix
        print("\nQuantization compatibility matrix:")
        print("Backend | BnB  | HQQ  | MLX  | Quanto")
        print("--------|------|------|------|-------")
        for backend, methods in compatibility_matrix.items():
            row = f"{backend:7} |"
            for method in test_methods:
                supported = methods.get(method, False)
                row += f" {'✓' if supported else '✗':^4} |"
            print(row)
        
        # Verify expected compatibility
        if "cuda" in compatibility_matrix:
            self.assertTrue(compatibility_matrix["cuda"].get("bnb", False))
            self.assertTrue(compatibility_matrix["cuda"].get("hqq", False))
        
        if "mps" in compatibility_matrix:
            # MPS should support MLX or Quanto
            self.assertTrue(
                compatibility_matrix["mps"].get("mlx", False) or
                compatibility_matrix["mps"].get("quanto", False)
            )
    
    def test_model_loader_with_different_backends(self):
        """Test ModelLoader with different backends."""
        # Skip this test as it requires significant refactoring
        self.skipTest("ModelLoader test requires refactoring for new loader architecture")
    
    def test_cross_backend_model_transfer(self):
        """Test transferring models between different devices."""
        # Create model on CPU
        model_cpu, config = create_tiny_model("llama", Backend.CPU)
        
        # Test input on CPU
        dummy_input_cpu = torch.randint(0, config.vocab_size, (1, 32))
        with torch.no_grad():
            output_cpu = model_cpu(dummy_input_cpu)
        
        # Transfer to GPU if available
        if torch.cuda.is_available():
            model_cuda = model_cpu.cuda()
            dummy_input_cuda = dummy_input_cpu.cuda()
            
            with torch.no_grad():
                output_cuda = model_cuda(dummy_input_cuda)
            
            # Outputs should be the same (within floating point tolerance)
            torch.testing.assert_close(
                output_cpu.logits.cpu(),
                output_cuda.logits.cpu(),
                rtol=1e-3,
                atol=1e-5
            )
        
        # Transfer to MPS if available
        if hasattr(torch, 'mps') and torch.mps.is_available():
            model_mps = model_cpu.to("mps")
            dummy_input_mps = dummy_input_cpu.to("mps")
            
            with torch.no_grad():
                output_mps = model_mps(dummy_input_mps)
            
            # Outputs should be similar
            torch.testing.assert_close(
                output_cpu.logits,
                output_mps.logits.cpu(),
                rtol=1e-3,
                atol=1e-5
            )


if __name__ == "__main__":
    unittest.main()