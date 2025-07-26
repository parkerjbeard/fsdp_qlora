"""
Integration tests for the backend_manager module.

These tests verify real-world usage scenarios and interactions with PyTorch.
"""

import os
import sys
import unittest
import warnings
from unittest.mock import patch

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend, BackendManager


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=768, hidden_size=1024, output_size=768):
        """Initialize the dummy model with specified dimensions."""
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the model."""
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestBackendManagerIntegration(unittest.TestCase):
    """Integration tests for BackendManager."""

    def setUp(self):
        """Set up test environment."""
        # Suppress warnings during tests
        warnings.filterwarnings("ignore")

    def tearDown(self):
        """Clean up after tests."""
        # Reset warnings
        warnings.resetwarnings()

    def test_real_backend_detection(self):
        """Test backend detection on the actual system."""
        manager = BackendManager(verbose=False)

        # Verify we got a valid backend
        self.assertIsInstance(manager.backend, Backend)

        # Verify device is valid
        self.assertIsInstance(manager.device, torch.device)

        # Print actual detection results for debugging
        print(f"\nDetected backend: {manager.backend}")
        print(f"Available backends: {manager._available_backends}")

    def test_model_to_device(self):
        """Test moving a model to the detected device."""
        manager = BackendManager(verbose=False)
        model = DummyModel()

        # Skip if MLX backend (doesn't use torch devices)
        if manager.backend != Backend.MLX:
            # Move model to device
            model = model.to(manager.device)

            # Verify model is on correct device
            for param in model.parameters():
                self.assertEqual(param.device.type, manager.device.type)

    def test_tensor_operations(self):
        """Test basic tensor operations on detected backend."""
        manager = BackendManager(verbose=False)

        # Skip if MLX backend
        if manager.backend != Backend.MLX:
            # Create tensors on device
            x = torch.randn(32, 768, device=manager.device)
            y = torch.randn(32, 768, device=manager.device)

            # Perform operations
            z = x + y
            result = torch.matmul(x, y.T)

            # Verify results are on correct device
            self.assertEqual(z.device.type, manager.device.type)
            self.assertEqual(result.device.type, manager.device.type)

    def test_dtype_compatibility(self):
        """Test dtype compatibility with backend."""
        manager = BackendManager(verbose=False)

        # Skip if MLX backend
        if manager.backend != Backend.MLX:
            # Get recommended dtype
            dtype = manager.get_dtype()

            # Create tensor with recommended dtype
            x = torch.randn(10, 10, device=manager.device, dtype=dtype)

            # Verify dtype
            self.assertIn(x.dtype, [torch.float16, torch.bfloat16])

    def test_memory_info_retrieval(self):
        """Test memory information retrieval."""
        manager = BackendManager(verbose=False)
        info = manager.get_memory_info()

        # Should always return a dict
        self.assertIsInstance(info, dict)

        # Check backend-specific info
        if manager.backend == Backend.CUDA and torch.cuda.is_available():
            self.assertIn('total_memory', info)
            self.assertIn('allocated_memory', info)
            self.assertGreater(info['total_memory'], 0)

    def test_model_size_recommendations(self):
        """Test model size validation and batch size recommendations."""
        manager = BackendManager(verbose=False)

        # Test various model sizes
        model_sizes = [7, 13, 30, 70]

        for size in model_sizes:
            try:
                manager.validate_model_size(size)
                # If validation passes, get batch size recommendation
                batch_size = manager.get_optimal_batch_size(size)
                self.assertGreater(batch_size, 0)
                print(f"\n{manager.backend} - {size}B model: batch_size={batch_size}")
            except ValueError as e:
                # Model too large for backend
                print(f"\n{manager.backend} - {size}B model: {str(e)}")

    def test_quantization_support(self):
        """Test quantization bit width support."""
        manager = BackendManager(verbose=False)

        # Common quantization bit widths
        bit_widths = [4, 8, 16, 32]

        print(f"\n{manager.backend} quantization support:")
        for bits in bit_widths:
            supported = manager.supports_quantization(bits)
            print(f"  {bits}-bit: {'✓' if supported else '✗'}")

    def test_distributed_backend_configuration(self):
        """Test distributed training backend configuration."""
        manager = BackendManager(verbose=False)

        dist_backend = manager.get_distributed_backend()

        if dist_backend:
            print(f"\n{manager.backend} distributed backend: {dist_backend}")

            # Verify it's a valid distributed backend
            self.assertIn(dist_backend, ["nccl", "gloo", "mpi"])

            # Check FSDP support
            if manager.capabilities.supports_fsdp:
                print("FSDP supported: ✓")
            else:
                print("FSDP supported: ✗")

    def test_backend_switching(self):
        """Test switching between different backends."""
        available_backends = []

        # Try each backend
        for backend in [Backend.CPU, Backend.CUDA, Backend.MPS, Backend.MLX]:
            try:
                manager = BackendManager(backend=backend, verbose=False)
                available_backends.append(backend)
            except ValueError:
                # Backend not available
                pass

        print(f"\nAvailable backends on this system: {[str(b) for b in available_backends]}")

        # Ensure at least CPU is available
        self.assertIn(Backend.CPU, available_backends)

    def test_environment_variable_configuration(self):
        """Test configuration via environment variables."""
        # Test with CPU backend (always available)
        with patch.dict(os.environ, {'FSDP_BACKEND': 'cpu'}):
            manager = BackendManager.from_env(verbose=False)
            self.assertEqual(manager.backend, Backend.CPU)

        # Test with invalid backend
        with patch.dict(os.environ, {'FSDP_BACKEND': 'invalid_backend'}):
            with self.assertRaises(ValueError):
                BackendManager.from_env(verbose=False)

    def test_model_training_simulation(self):
        """Simulate a simple training scenario with the detected backend."""
        manager = BackendManager(verbose=False)

        # Skip if MLX backend
        if manager.backend == Backend.MLX:
            print("\nSkipping training simulation for MLX backend")
            return

        # Create model and move to device
        model = DummyModel()
        
        # Get appropriate dtype
        dtype = manager.get_dtype()
        
        # Convert model to appropriate dtype and device
        model = model.to(dtype=dtype, device=manager.device)

        # Create dummy data
        batch_size = manager.get_optimal_batch_size(0.001)  # Small model
        x = torch.randn(batch_size, 768, device=manager.device, dtype=dtype)

        # Forward pass
        output = model(x)

        # Verify output
        self.assertEqual(output.shape, (batch_size, 768))
        self.assertEqual(output.device.type, manager.device.type)

        print(f"\nTraining simulation successful on {manager.backend}")
        print(f"  Batch size: {batch_size}")
        print(f"  Dtype: {dtype}")
        print(f"  Device: {manager.device}")

    def test_capability_matrix_consistency(self):
        """Test that capability matrix is consistent and complete."""
        # Check all backends have complete capability definitions
        for backend in Backend:
            caps = BackendManager.CAPABILITIES[backend]

            # Verify all fields are properly set
            self.assertIsNotNone(caps.quantization_bits)
            self.assertIsInstance(caps.supports_distributed, bool)
            self.assertIsInstance(caps.supports_fsdp, bool)
            self.assertIsInstance(caps.supports_bfloat16, bool)
            self.assertIsInstance(caps.supports_flash_attention, bool)

            # Verify quantization bits are valid
            for bits in caps.quantization_bits:
                self.assertIn(bits, [4, 8, 16, 32])

            # Verify distributed backend consistency
            if caps.supports_distributed:
                self.assertIsNotNone(caps.distributed_backend)
            else:
                self.assertIsNone(caps.distributed_backend)

    def test_verbose_output(self):
        """Test verbose output functionality."""
        print("\n" + "="*60)
        print("Testing verbose output")
        print("="*60)

        # Capture verbose output by creating manager with verbose=True
        manager = BackendManager(verbose=True)

        # Verify manager was created successfully
        self.assertIsInstance(manager.backend, Backend)


class TestBackendManagerPerformance(unittest.TestCase):
    """Performance-related tests for BackendManager."""

    def test_initialization_speed(self):
        """Test that BackendManager initializes quickly."""
        import time

        start_time = time.time()
        manager = BackendManager(verbose=False)
        end_time = time.time()

        initialization_time = end_time - start_time

        # Should initialize in less than 1 second
        self.assertLess(initialization_time, 1.0)
        print(f"\nBackendManager initialization time: {initialization_time:.3f}s")

    def test_multiple_instances(self):
        """Test creating multiple BackendManager instances."""
        managers = []

        # Create multiple instances
        for i in range(5):
            manager = BackendManager(verbose=False)
            managers.append(manager)

        # All should detect the same backend
        backends = [m.backend for m in managers]
        self.assertEqual(len(set(backends)), 1)

        print(f"\nCreated {len(managers)} BackendManager instances")
        print(f"All detected backend: {backends[0]}")


if __name__ == "__main__":
    # Run with more verbose output
    unittest.main(verbosity=2)