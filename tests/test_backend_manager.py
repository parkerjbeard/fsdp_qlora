"""
Unit tests for the backend_manager module.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.backend_manager import Backend, BackendCapabilities, BackendManager, detect_backend, get_device


class TestBackend(unittest.TestCase):
    """Test the Backend enum."""

    def test_backend_enum_values(self):
        """Test that Backend enum has correct values."""
        self.assertEqual(Backend.CUDA.value, "cuda")
        self.assertEqual(Backend.MPS.value, "mps")
        self.assertEqual(Backend.MLX.value, "mlx")
        self.assertEqual(Backend.CPU.value, "cpu")

    def test_backend_string_representation(self):
        """Test string representation of Backend enum."""
        self.assertEqual(str(Backend.CUDA), "cuda")
        self.assertEqual(str(Backend.MPS), "mps")
        self.assertEqual(str(Backend.MLX), "mlx")
        self.assertEqual(str(Backend.CPU), "cpu")


class TestBackendCapabilities(unittest.TestCase):
    """Test the BackendCapabilities dataclass."""

    def test_default_initialization(self):
        """Test default initialization of BackendCapabilities."""
        caps = BackendCapabilities()
        self.assertEqual(caps.quantization_bits, [])
        self.assertFalse(caps.supports_distributed)
        self.assertFalse(caps.supports_fsdp)
        self.assertFalse(caps.supports_bfloat16)
        self.assertFalse(caps.supports_flash_attention)
        self.assertIsNone(caps.max_model_size)
        self.assertIsNone(caps.distributed_backend)
        self.assertEqual(caps.notes, [])

    def test_custom_initialization(self):
        """Test custom initialization of BackendCapabilities."""
        caps = BackendCapabilities(
            quantization_bits=[4, 8],
            supports_distributed=True,
            supports_fsdp=True,
            supports_bfloat16=True,
            supports_flash_attention=True,
            max_model_size=70,
            distributed_backend="nccl",
            notes=["Test note"]
        )
        self.assertEqual(caps.quantization_bits, [4, 8])
        self.assertTrue(caps.supports_distributed)
        self.assertTrue(caps.supports_fsdp)
        self.assertTrue(caps.supports_bfloat16)
        self.assertTrue(caps.supports_flash_attention)
        self.assertEqual(caps.max_model_size, 70)
        self.assertEqual(caps.distributed_backend, "nccl")
        self.assertEqual(caps.notes, ["Test note"])


class TestBackendManager(unittest.TestCase):
    """Test the BackendManager class."""

    def test_capabilities_matrix_completeness(self):
        """Test that all backends have defined capabilities."""
        for backend in Backend:
            self.assertIn(backend, BackendManager.CAPABILITIES)
            caps = BackendManager.CAPABILITIES[backend]
            self.assertIsInstance(caps, BackendCapabilities)

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    @patch('src.core.backend_manager.BackendManager._is_mlx_available')
    def test_backend_detection_cuda_only(self, mock_mlx, mock_mps, mock_cuda):
        """Test backend detection when only CUDA is available."""
        mock_cuda.return_value = True
        mock_mps.return_value = False
        mock_mlx.return_value = False

        manager = BackendManager(verbose=False)
        available = manager._available_backends

        self.assertIn(Backend.CUDA, available)
        self.assertIn(Backend.CPU, available)
        self.assertNotIn(Backend.MPS, available)
        self.assertNotIn(Backend.MLX, available)
        self.assertEqual(manager.backend, Backend.CUDA)

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    @patch('src.core.backend_manager.BackendManager._is_mlx_available')
    def test_backend_detection_mps_only(self, mock_mlx, mock_mps, mock_cuda):
        """Test backend detection when only MPS is available."""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        mock_mlx.return_value = False

        manager = BackendManager(verbose=False)
        available = manager._available_backends

        self.assertNotIn(Backend.CUDA, available)
        self.assertIn(Backend.MPS, available)
        self.assertIn(Backend.CPU, available)
        self.assertEqual(manager.backend, Backend.MPS)

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    @patch('src.core.backend_manager.BackendManager._is_mlx_available')
    def test_backend_detection_mlx_available(self, mock_mlx, mock_mps, mock_cuda):
        """Test backend detection when MLX is available."""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        mock_mlx.return_value = True

        manager = BackendManager(verbose=False)
        available = manager._available_backends

        self.assertIn(Backend.MLX, available)
        self.assertIn(Backend.CPU, available)
        self.assertEqual(manager.backend, Backend.MLX)

    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    @patch('src.core.backend_manager.BackendManager._is_mlx_available')
    def test_backend_detection_cpu_only(self, mock_mlx, mock_mps, mock_cuda):
        """Test backend detection when only CPU is available."""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        mock_mlx.return_value = False

        manager = BackendManager(verbose=False)
        available = manager._available_backends

        self.assertNotIn(Backend.CUDA, available)
        self.assertNotIn(Backend.MPS, available)
        self.assertNotIn(Backend.MLX, available)
        self.assertIn(Backend.CPU, available)
        self.assertEqual(manager.backend, Backend.CPU)

    @patch('torch.cuda.is_available')
    def test_manual_backend_selection(self, mock_cuda):
        """Test manual backend selection."""
        mock_cuda.return_value = True

        # Test with string
        manager = BackendManager(backend="cuda", verbose=False)
        self.assertEqual(manager.backend, Backend.CUDA)

        # Test with enum
        manager = BackendManager(backend=Backend.CUDA, verbose=False)
        self.assertEqual(manager.backend, Backend.CUDA)

    @patch('torch.cuda.is_available')
    def test_invalid_backend_selection(self, mock_cuda):
        """Test invalid backend selection raises error."""
        mock_cuda.return_value = False

        with self.assertRaises(ValueError) as cm:
            BackendManager(backend="cuda", verbose=False)

        self.assertIn("not available", str(cm.exception))

    def test_get_device(self):
        """Test device retrieval for different backends."""
        # Create manager with specific backend (mocking availability)
        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.CUDA, Backend.MPS, Backend.MLX, Backend.CPU]):

            # Test CUDA device
            manager = BackendManager(backend=Backend.CUDA, verbose=False)
            self.assertEqual(manager.device.type, "cuda")

            # Test MPS device
            manager = BackendManager(backend=Backend.MPS, verbose=False)
            self.assertEqual(manager.device.type, "mps")

            # Test CPU device
            manager = BackendManager(backend=Backend.CPU, verbose=False)
            self.assertEqual(manager.device.type, "cpu")

            # Test MLX (returns CPU for compatibility)
            manager = BackendManager(backend=Backend.MLX, verbose=False)
            self.assertEqual(manager.device.type, "cpu")

    def test_supports_quantization(self):
        """Test quantization support checking."""
        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.CUDA, Backend.CPU]):

            # CUDA supports 4, 8, 16 bit
            manager = BackendManager(backend=Backend.CUDA, verbose=False)
            self.assertTrue(manager.supports_quantization(4))
            self.assertTrue(manager.supports_quantization(8))
            self.assertTrue(manager.supports_quantization(16))
            self.assertFalse(manager.supports_quantization(32))

            # CPU doesn't support 4-bit
            manager = BackendManager(backend=Backend.CPU, verbose=False)
            self.assertFalse(manager.supports_quantization(4))
            self.assertTrue(manager.supports_quantization(8))
            self.assertTrue(manager.supports_quantization(16))

    def test_get_dtype(self):
        """Test dtype selection based on backend capabilities."""
        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.CUDA, Backend.MPS]):

            # CUDA supports bfloat16
            manager = BackendManager(backend=Backend.CUDA, verbose=False)
            self.assertEqual(manager.get_dtype(prefer_bfloat16=True), torch.bfloat16)
            self.assertEqual(manager.get_dtype(prefer_bfloat16=False), torch.float16)

            # MPS doesn't support bfloat16
            manager = BackendManager(backend=Backend.MPS, verbose=False)
            self.assertEqual(manager.get_dtype(prefer_bfloat16=True), torch.float16)
            self.assertEqual(manager.get_dtype(prefer_bfloat16=False), torch.float16)

    def test_get_distributed_backend(self):
        """Test distributed backend selection."""
        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.CUDA, Backend.MPS, Backend.MLX]):

            # CUDA uses nccl
            manager = BackendManager(backend=Backend.CUDA, verbose=False)
            self.assertEqual(manager.get_distributed_backend(), "nccl")

            # MPS uses gloo
            manager = BackendManager(backend=Backend.MPS, verbose=False)
            self.assertEqual(manager.get_distributed_backend(), "gloo")

            # MLX doesn't support distributed
            manager = BackendManager(backend=Backend.MLX, verbose=False)
            self.assertIsNone(manager.get_distributed_backend())

    def test_validate_model_size(self):
        """Test model size validation."""
        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.MPS, Backend.CPU]):

            # MPS has 70B limit
            manager = BackendManager(backend=Backend.MPS, verbose=False)
            manager.validate_model_size(70)  # Should not raise

            with self.assertRaises(ValueError) as cm:
                manager.validate_model_size(100)
            self.assertIn("exceeds the recommended limit", str(cm.exception))

            # CPU has 7B limit
            manager = BackendManager(backend=Backend.CPU, verbose=False)
            manager.validate_model_size(7)  # Should not raise

            with self.assertRaises(ValueError):
                manager.validate_model_size(13)

    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.CUDA, Backend.MLX]):

            # Test CUDA batch sizes
            manager = BackendManager(backend=Backend.CUDA, verbose=False)
            self.assertEqual(manager.get_optimal_batch_size(7), 4)
            self.assertEqual(manager.get_optimal_batch_size(13), 2)
            self.assertEqual(manager.get_optimal_batch_size(70), 1)

            # Test with longer sequence length
            self.assertEqual(manager.get_optimal_batch_size(7, sequence_length=4096), 2)

            # Test MLX batch sizes (generally higher due to unified memory)
            manager = BackendManager(backend=Backend.MLX, verbose=False)
            self.assertEqual(manager.get_optimal_batch_size(7), 8)
            self.assertEqual(manager.get_optimal_batch_size(13), 4)
            self.assertEqual(manager.get_optimal_batch_size(70), 2)

    @patch.dict(os.environ, {'FSDP_BACKEND': 'cuda'})
    @patch('torch.cuda.is_available', return_value=True)
    def test_from_env(self, mock_cuda):
        """Test creating BackendManager from environment variables."""
        manager = BackendManager.from_env(verbose=False)
        self.assertEqual(manager.backend, Backend.CUDA)

    @patch('platform.system')
    @patch('platform.processor')
    def test_mlx_availability_check(self, mock_processor, mock_system):
        """Test MLX availability checking."""
        # Test when MLX is available on Apple Silicon
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "arm"

        with patch('builtins.__import__') as mock_import:
            # Simulate successful MLX import
            manager = BackendManager(verbose=False)
            result = manager._is_mlx_available()
            # This will depend on whether MLX can actually be imported
            # In test environment, it's likely False
            self.assertIsInstance(result, bool)

        # Test when not on Apple Silicon
        mock_system.return_value = "Linux"
        mock_processor.return_value = "x86_64"

        manager = BackendManager(verbose=False)
        self.assertFalse(manager._is_mlx_available())

    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.is_available', return_value=True)
    def test_get_memory_info_cuda(self, mock_available, mock_reserved, mock_allocated, mock_props):
        """Test memory info retrieval for CUDA."""
        # Mock CUDA memory values
        mock_props.return_value = MagicMock(total_memory=16_000_000_000)
        mock_allocated.return_value = 4_000_000_000
        mock_reserved.return_value = 5_000_000_000

        with patch.object(BackendManager, '_detect_available_backends',
                         return_value=[Backend.CUDA]):
            manager = BackendManager(backend=Backend.CUDA, verbose=False)
            info = manager.get_memory_info()

            self.assertEqual(info['total_memory'], 16_000_000_000)
            self.assertEqual(info['allocated_memory'], 4_000_000_000)
            self.assertEqual(info['reserved_memory'], 5_000_000_000)
            self.assertEqual(info['free_memory'], 12_000_000_000)


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""

    @patch('torch.cuda.is_available', return_value=True)
    def test_detect_backend(self, mock_cuda):
        """Test the detect_backend convenience function."""
        backend = detect_backend()
        self.assertEqual(backend, Backend.CUDA)

    @patch('torch.cuda.is_available', return_value=True)
    def test_get_device(self, mock_cuda):
        """Test the get_device convenience function."""
        device = get_device("cuda")
        self.assertEqual(device.type, "cuda")

        # Test auto-detection
        device = get_device()
        self.assertEqual(device.type, "cuda")


if __name__ == "__main__":
    unittest.main()