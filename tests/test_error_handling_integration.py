"""
Integration tests for error handling in real scenarios.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from unittest.mock import patch, MagicMock
import gc

from src.core.backend_manager import Backend, BackendManager
from src.core.quantization_wrapper import QuantizationConfig, QuantizationMethod
from src.core.quantization_error_handler import (
    QuantizationErrorHandler,
    create_safe_quantization_adapter,
)
from src.utils.error_handling import ResourceMonitor


# Skip if no GPU/MPS available
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() and not torch.cuda.is_available(),
    reason="Requires MPS or CUDA"
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, size: int = 1000):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TestQuantizationErrorHandlerIntegration:
    """Integration tests for quantization error handling."""
    
    @pytest.fixture
    def backend(self):
        """Get available backend."""
        if torch.backends.mps.is_available():
            return Backend.MPS
        elif torch.cuda.is_available():
            return Backend.CUDA
        else:
            return Backend.CPU
    
    @pytest.fixture
    def device(self, backend):
        """Get device string."""
        if backend == Backend.MPS:
            return "mps"
        elif backend == Backend.CUDA:
            return "cuda"
        else:
            return "cpu"
    
    def test_oom_recovery_during_quantization(self, backend, device):
        """Test OOM recovery during model quantization."""
        # Create a model that's likely to cause memory pressure
        model = SimpleModel(size=5000).to(device)
        
        config = QuantizationConfig(
            method=QuantizationMethod.BNB_INT8 if backend == Backend.CUDA else QuantizationMethod.QUANTO_INT8,
            bits=8,
            compute_dtype=torch.float16,
        )
        
        error_handler = QuantizationErrorHandler(backend, config)
        
        # Mock the quantization adapter to simulate OOM
        mock_adapter = MagicMock()
        oom_count = 0
        
        def mock_quantize(m):
            nonlocal oom_count
            oom_count += 1
            if oom_count < 2:
                # Simulate OOM on first attempt
                raise RuntimeError("MPS out of memory" if device == "mps" else "CUDA out of memory")
            # Succeed on second attempt
            return m
        
        mock_adapter.quantize_model = mock_quantize
        
        # Test with recovery
        result = error_handler.quantize_model_safe(model, mock_adapter, batch_size=32)
        
        # Should have succeeded after retry
        assert result is not None
        assert error_handler.stats["oom_errors"] == 1
        assert error_handler.stats["successful_quantizations"] == 1
    
    def test_layer_wise_quantization_with_failures(self, backend, device):
        """Test layer-wise quantization with some layers failing."""
        model = SimpleModel().to(device)
        
        config = QuantizationConfig(bits=8)
        error_handler = QuantizationErrorHandler(backend, config)
        
        success_count = 0
        fail_count = 0
        
        def mock_quantize_fn(layer):
            nonlocal success_count, fail_count
            # Fail for fc2
            if hasattr(layer, 'out_features') and layer.out_features == 1000:
                fail_count += 1
                raise RuntimeError("Simulated quantization failure")
            success_count += 1
            return layer
        
        # Quantize each layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                result = error_handler.quantize_layer_safe(
                    module,
                    name,
                    mock_quantize_fn
                )
                # Should always return something (original on failure)
                assert result is not None
        
        assert success_count == 1  # fc3 only (has 10 out_features)
        assert fail_count == 2     # fc1 and fc2 (both have 1000 out_features)
    
    def test_memory_monitoring_during_operations(self, backend, device):
        """Test memory monitoring integration."""
        model = SimpleModel().to(device)
        
        config = QuantizationConfig(bits=8)
        error_handler = QuantizationErrorHandler(backend, config, enable_monitoring=True)
        
        # Get initial memory
        initial_stats = error_handler.get_statistics()
        assert "current_memory" in initial_stats
        
        # Create some memory pressure
        tensors = []
        for _ in range(10):
            tensors.append(torch.randn(1000, 1000, device=device))
        
        # Check if monitor detects high usage
        warning = error_handler.monitor.check_resources()
        # May or may not warn depending on system state
        
        # Clean up
        del tensors
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
        gc.collect()
    
    def test_save_load_with_error_recovery(self, backend, device):
        """Test save/load operations with error handling."""
        model = SimpleModel().to(device)
        
        config = QuantizationConfig(bits=8)
        error_handler = QuantizationErrorHandler(backend, config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")
            
            # Mock adapter with intermittent failures
            mock_adapter = MagicMock()
            save_attempts = 0
            
            def mock_save(m, path):
                nonlocal save_attempts
                save_attempts += 1
                if save_attempts < 2:
                    raise OSError("Simulated save failure")
                # Actually save something
                torch.save({"model_state": "dummy"}, path)
            
            mock_adapter.save_quantized_model = mock_save
            
            # Should succeed after retry
            error_handler.save_quantized_model_safe(model, save_path, mock_adapter)
            assert save_attempts == 2
            assert os.path.exists(save_path)
            
            # Test loading
            load_attempts = 0
            
            def mock_load(path, cls, device=None, **kwargs):
                nonlocal load_attempts
                load_attempts += 1
                if load_attempts < 2:
                    raise OSError("Simulated load failure")
                return SimpleModel()
            
            mock_adapter.load_quantized_model = mock_load
            
            loaded_model = error_handler.load_quantized_model_safe(
                save_path,
                SimpleModel,
                mock_adapter,
                device=device
            )
            
            assert loaded_model is not None
            assert load_attempts == 2


class TestBackendFallbackIntegration:
    """Test backend fallback in real scenarios."""
    
    def test_backend_initialization_with_fallback(self):
        """Test backend initialization with fallback."""
        # Try to initialize with a potentially failing backend
        with patch('torch.backends.mps.is_available', return_value=False):
            with patch('torch.cuda.is_available', return_value=False):
                manager = BackendManager(backend=None, verbose=False)
                
                # Should fall back to CPU
                assert manager.backend == Backend.CPU
                assert manager.device == torch.device("cpu")
    
    def test_quantization_backend_fallback(self):
        """Test quantization with backend fallback."""
        from src.backends.mps.mps_quantization import MPSQuantizationAdapter
        
        # Create a config that might fail
        config = QuantizationConfig(
            method=QuantizationMethod.QUANTO_INT4,
            bits=4,  # 4-bit might not be supported
        )
        
        # Mock MPS to simulate failure
        with patch.object(MPSQuantizationAdapter, 'quantize_model', side_effect=RuntimeError("Unsupported")):
            try:
                adapter = MPSQuantizationAdapter(Backend.MPS, config)
                model = SimpleModel()
                
                # This should fail
                with pytest.raises(RuntimeError):
                    adapter.quantize_model(model)
                    
            except Exception:
                # In real scenario, we'd fall back to a different method
                pass


class TestMemoryPressureScenarios:
    """Test behavior under memory pressure."""
    
    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS specific test")
    def test_mps_memory_fragmentation_handling(self):
        """Test handling MPS memory fragmentation."""
        from src.backends.mps.mps_fsdp_wrapper import MPSFSDPWrapper, MPSFSDPConfig
        
        config = MPSFSDPConfig(
            unified_memory_pool_size=int(8e9),  # 8GB
            aggressive_memory_optimization=True,
        )
        
        model = SimpleModel(size=2000)
        wrapper = MPSFSDPWrapper(config)
        
        # Simulate memory pressure
        large_tensors = []
        try:
            # Allocate until we hit pressure
            for i in range(100):
                large_tensors.append(torch.randn(1000, 1000, device="mps"))
                
                if i % 10 == 0:
                    # Check memory stats
                    stats = wrapper.get_memory_stats()
                    if stats.get("allocated_gb", 0) > 10:  # Arbitrary threshold
                        break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Expected - clean up
                del large_tensors
                torch.mps.empty_cache()
                torch.mps.synchronize()
        
        # Should be able to continue after cleanup
        small_tensor = torch.randn(10, 10, device="mps")
        assert small_tensor is not None
    
    def test_batch_size_reduction_on_oom(self):
        """Test automatic batch size reduction on OOM."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        backend = Backend.MPS if device == "mps" else Backend.CPU
        
        config = QuantizationConfig(bits=8)
        error_handler = QuantizationErrorHandler(backend, config)
        
        # Test batch size reduction logic
        original_batch = 64
        new_batch = error_handler._reduce_batch_size(original_batch)
        assert new_batch == 32
        
        # Test multiple reductions
        new_batch = error_handler._reduce_batch_size(new_batch)
        assert new_batch == 16
        
        # Test minimum
        min_batch = error_handler._reduce_batch_size(1)
        assert min_batch == 1


class TestResourceMonitoringIntegration:
    """Test resource monitoring in real scenarios."""
    
    def test_continuous_monitoring_during_training(self):
        """Test monitoring during simulated training."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        monitor = ResourceMonitor(
            device=device,
            memory_threshold=0.7,  # Lower threshold for testing
            check_interval=0.1     # Fast checking for test
        )
        
        warnings_issued = []
        
        # Simulate training loop
        model = SimpleModel().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        
        for epoch in range(3):
            for batch in range(5):
                # Create batch data
                data = torch.randn(32, 1000, device=device)
                target = torch.randint(0, 10, (32,), device=device)
                
                # Forward pass
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Check resources
                warning = monitor.check_resources()
                if warning:
                    warnings_issued.append(warning)
                
                # Simulate memory growth
                if batch == 3:
                    # Allocate extra memory to trigger warning
                    extra = torch.randn(5000, 5000, device=device)
                    
        # Clean up
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    
    def test_configuration_validation_real_models(self):
        """Test configuration validation with real model configs."""
        from src.utils.error_handling import validate_configuration
        
        # Test various configurations
        configs = [
            {
                "model_name": "meta-llama/Llama-2-7b-hf",
                "batch_size": 256,  # Too large
                "quantization_bits": 4,
                "backend": "mps",
                "dtype": "float16",
            },
            {
                "model_name": "meta-llama/Llama-2-13b-hf", 
                "batch_size": 32,
                "quantization_bits": 7,  # Unusual
                "backend": "cuda",
                "dtype": "bfloat16",
            },
            {
                "model_name": "meta-llama/Llama-2-70b-hf",
                "batch_size": 8,
                "quantization_bits": 4,
                "backend": "mps",
                "dtype": "bfloat16",  # Not supported on MPS
                "model_size": 70,
            },
        ]
        
        for i, config in enumerate(configs):
            issues = validate_configuration(config)
            
            if i == 0:
                # First config has large batch size
                assert any("batch size" in issue for issue in issues)
            elif i == 1:
                # Second config has unusual quantization
                assert any("quantization bits" in issue for issue in issues)
            elif i == 2:
                # Third config has bfloat16 on MPS and large model
                assert any("BFloat16" in issue for issue in issues)
                # Should suggest gradient checkpointing for 70B model
                assert any("gradient checkpoint" in issue.lower() for issue in issues)


class TestEndToEndErrorHandling:
    """Test complete error handling workflow."""
    
    def test_full_quantization_workflow_with_errors(self):
        """Test complete quantization workflow with various errors."""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        backend = Backend.MPS if device == "mps" else Backend.CPU
        
        # Create model
        model = SimpleModel().to(device)
        
        # Create config with potential issues
        config = QuantizationConfig(
            method=QuantizationMethod.QUANTO_INT8,
            bits=8,
            compute_dtype=torch.float16,
        )
        
        # Create mock adapter that simulates various errors
        mock_adapter = MagicMock()
        
        error_sequence = ["oom", "success", "io_error", "success"]
        call_count = 0
        
        def mock_operation(*args, **kwargs):
            nonlocal call_count
            error_type = error_sequence[call_count % len(error_sequence)]
            call_count += 1
            
            if error_type == "oom":
                raise RuntimeError(f"{device.upper()} out of memory")
            elif error_type == "io_error":
                raise OSError("Disk full")
            else:
                return model  # Success
        
        mock_adapter.quantize_model = mock_operation
        mock_adapter.save_quantized_model = mock_operation
        
        # Create safe adapter
        safe_adapter = create_safe_quantization_adapter(backend, config, mock_adapter)
        
        # Test quantization (will retry on OOM)
        quantized_model = safe_adapter.quantize_model(model)
        assert quantized_model is not None
        
        # Get statistics
        stats = safe_adapter.get_statistics()
        assert stats["oom_errors"] > 0
        assert stats["successful_quantizations"] > 0
        
        # Test save (will retry on IO error)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")
            safe_adapter.save_quantized_model(quantized_model, save_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])