"""
Unit tests for error handling utilities.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from src.utils.error_handling import (
    ErrorContext,
    MemoryError,
    BackendError,
    get_memory_info,
    detect_oom_error,
    suggest_memory_optimization,
    format_error_message,
    error_handler,
    with_recovery,
    ResourceMonitor,
    validate_configuration,
    handle_import_error,
    handle_backend_error,
)


class TestErrorContext:
    """Test ErrorContext dataclass."""
    
    def test_error_context_creation(self):
        """Test creating error context with various fields."""
        context = ErrorContext(
            operation="test_operation",
            backend="cuda",
            model_size=7.5,
            quantization_bits=8,
            batch_size=32,
            device="cuda:0",
            additional_info={"key": "value"}
        )
        
        assert context.operation == "test_operation"
        assert context.backend == "cuda"
        assert context.model_size == 7.5
        assert context.quantization_bits == 8
        assert context.batch_size == 32
        assert context.device == "cuda:0"
        assert context.additional_info == {"key": "value"}
    
    def test_error_context_defaults(self):
        """Test default values for optional fields."""
        context = ErrorContext(operation="test")
        
        assert context.operation == "test"
        assert context.backend is None
        assert context.model_size is None
        assert context.quantization_bits is None
        assert context.batch_size is None
        assert context.device is None
        assert context.additional_info is None


class TestMemoryInfo:
    """Test memory information functions."""
    
    def test_get_memory_info_cpu(self):
        """Test getting CPU memory info."""
        info = get_memory_info("cpu")
        
        assert "total_gb" in info
        assert "available_gb" in info
        assert "used_gb" in info
        assert "percent" in info
        
        # Sanity checks
        assert info["total_gb"] > 0
        assert info["used_gb"] >= 0
        assert 0 <= info["percent"] <= 100
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_info_cuda(self):
        """Test getting CUDA memory info."""
        info = get_memory_info("cuda")
        
        assert "allocated_gb" in info
        assert "reserved_gb" in info
        assert "total_gb" in info
        assert "available_gb" in info
    
    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_get_memory_info_mps(self):
        """Test getting MPS memory info."""
        info = get_memory_info("mps")
        
        # MPS might fail or succeed depending on the system
        if "error" not in info:
            assert "allocated_gb" in info
            assert "reserved_gb" in info


class TestOOMDetection:
    """Test OOM error detection."""
    
    def test_detect_oom_error_positive_cases(self):
        """Test detecting OOM errors from various exception messages."""
        oom_exceptions = [
            RuntimeError("CUDA out of memory"),
            RuntimeError("MPS out of memory error"),
            MemoryError("Cannot allocate memory"),
            Exception("Insufficient memory for allocation"),
            RuntimeError("OOM: Failed to allocate tensor"),
        ]
        
        for exc in oom_exceptions:
            assert detect_oom_error(exc) is True
    
    def test_detect_oom_error_negative_cases(self):
        """Test non-OOM errors are not detected as OOM."""
        non_oom_exceptions = [
            ValueError("Invalid value"),
            RuntimeError("Device mismatch"),
            TypeError("Type error"),
            Exception("Random error"),
        ]
        
        for exc in non_oom_exceptions:
            assert detect_oom_error(exc) is False


class TestMemoryOptimizationSuggestions:
    """Test memory optimization suggestions."""
    
    def test_suggest_memory_optimization_batch_size(self):
        """Test batch size reduction suggestions."""
        context = ErrorContext(
            operation="training",
            batch_size=64,
            quantization_bits=8,
        )
        memory_info = {"available_gb": 2.0}
        
        suggestions = suggest_memory_optimization(context, memory_info)
        
        assert any("batch size" in s.lower() for s in suggestions)
        assert any("32" in s for s in suggestions)  # Half of 64
    
    def test_suggest_memory_optimization_quantization(self):
        """Test quantization suggestions."""
        context = ErrorContext(
            operation="quantization",
            quantization_bits=16,
            model_size=10.0,
        )
        memory_info = {"available_gb": 4.0}
        
        suggestions = suggest_memory_optimization(context, memory_info)
        
        assert any("quantization" in s.lower() for s in suggestions)
        assert any("gradient checkpoint" in s.lower() for s in suggestions)
    
    def test_suggest_memory_optimization_backend_specific(self):
        """Test backend-specific suggestions."""
        # MPS suggestions
        context = ErrorContext(operation="training", backend="mps")
        suggestions = suggest_memory_optimization(context, {})
        
        assert any("PYTORCH_MPS_HIGH_WATERMARK_RATIO" in s for s in suggestions)
        assert any("torch.mps.empty_cache()" in s for s in suggestions)
        
        # CUDA suggestions
        context = ErrorContext(operation="training", backend="cuda")
        suggestions = suggest_memory_optimization(context, {})
        
        assert any("PYTORCH_CUDA_ALLOC_CONF" in s for s in suggestions)
        assert any("torch.cuda.empty_cache()" in s for s in suggestions)


class TestErrorFormatting:
    """Test error message formatting."""
    
    def test_format_error_message_basic(self):
        """Test basic error message formatting."""
        exc = ValueError("Test error")
        context = ErrorContext(
            operation="Test Operation",
            backend="cpu",
            device="cpu",
        )
        
        message = format_error_message(exc, context, include_traceback=False)
        
        assert "Test Operation" in message
        assert "ValueError" in message
        assert "Test error" in message
        assert "Backend: cpu" in message
        assert "Device: cpu" in message
    
    def test_format_error_message_with_memory_info(self):
        """Test error message with memory information."""
        exc = RuntimeError("Out of memory")
        context = ErrorContext(
            operation="Model Loading",
            device="cpu",
            model_size=7.5,
            batch_size=32,
        )
        
        message = format_error_message(exc, context, include_traceback=False)
        
        assert "Memory Status:" in message
        assert "Model Size: 7.5GB" in message
        assert "Batch Size: 32" in message
    
    def test_format_error_message_with_suggestions(self):
        """Test error message includes suggestions for OOM."""
        exc = RuntimeError("CUDA out of memory")
        context = ErrorContext(
            operation="Training",
            backend="cuda",
            device="cuda",
            batch_size=64,
        )
        
        message = format_error_message(exc, context, include_traceback=False)
        
        assert "Suggested Solutions:" in message
        assert "batch size" in message.lower()


class TestErrorHandler:
    """Test error handler context manager."""
    
    def test_error_handler_success(self):
        """Test error handler with successful operation."""
        context = ErrorContext(operation="test_success")
        
        with error_handler(context, reraise=False):
            result = 1 + 1
        
        assert result == 2
    
    def test_error_handler_exception_logging(self):
        """Test error handler logs exceptions properly."""
        context = ErrorContext(operation="test_error", device="cpu")
        
        with patch('src.utils.error_handling.logger') as mock_logger:
            with pytest.raises(ValueError):
                with error_handler(context, reraise=True):
                    raise ValueError("Test exception")
            
            # Check that error was logged
            mock_logger.error.assert_called_once()
            logged_message = mock_logger.error.call_args[0][0]
            assert "test_error" in logged_message
            assert "ValueError" in logged_message
    
    @patch('src.utils.error_handling.get_memory_info')
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.is_available')
    def test_error_handler_oom_cleanup_cuda(self, mock_is_available, mock_sync, mock_empty, mock_get_memory):
        """Test OOM cleanup for CUDA."""
        # Mock CUDA as available for this test
        mock_is_available.return_value = True
        # Mock memory info to avoid actual CUDA calls
        mock_get_memory.return_value = {
            "allocated_gb": 5.0,
            "reserved_gb": 6.0,
            "total_gb": 8.0,
            "available_gb": 3.0
        }
        
        context = ErrorContext(operation="test_oom", device="cuda")
        
        with error_handler(context, reraise=False):
            raise RuntimeError("CUDA out of memory")
        
        # Check cleanup was called
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()
    
    @patch('torch.mps.empty_cache')
    @patch('torch.mps.synchronize')
    def test_error_handler_oom_cleanup_mps(self, mock_sync, mock_empty):
        """Test OOM cleanup for MPS."""
        context = ErrorContext(operation="test_oom", device="mps")
        
        with error_handler(context, reraise=False):
            raise RuntimeError("MPS out of memory")
        
        # Check cleanup was called
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()


class TestRecoveryDecorator:
    """Test automatic retry decorator."""
    
    def test_with_recovery_success_first_try(self):
        """Test function succeeds on first try."""
        call_count = 0
        
        @with_recovery(max_retries=3)
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count == 1
    
    def test_with_recovery_success_after_retry(self):
        """Test function succeeds after retries."""
        call_count = 0
        
        @with_recovery(max_retries=3, backoff_factor=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count == 3
    
    def test_with_recovery_all_retries_fail(self):
        """Test all retries fail."""
        call_count = 0
        
        @with_recovery(max_retries=2, backoff_factor=0.1)
        def test_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")
        
        with pytest.raises(ValueError, match="Persistent error"):
            test_func()
        
        assert call_count == 2
    
    def test_with_recovery_with_recovery_function(self):
        """Test recovery function is called."""
        recovery_called = False
        
        def recovery_fn(exc):
            nonlocal recovery_called
            recovery_called = True
        
        @with_recovery(max_retries=2, backoff_factor=0.1, recover_fn=recovery_fn)
        def test_func():
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            test_func()
        
        assert recovery_called


class TestResourceMonitor:
    """Test resource monitoring."""
    
    def test_resource_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = ResourceMonitor(device="cpu", memory_threshold=0.8)
        
        assert monitor.device == "cpu"
        assert monitor.memory_threshold == 0.8
        assert monitor._last_check == 0
    
    @patch('src.utils.error_handling.get_memory_info')
    def test_resource_monitor_check_below_threshold(self, mock_get_info):
        """Test monitoring when below threshold."""
        mock_get_info.return_value = {"percent": 50.0}
        
        monitor = ResourceMonitor(device="cpu", check_interval=0)
        warning = monitor.check_resources()
        
        assert warning is None
    
    @patch('src.utils.error_handling.get_memory_info')
    @patch('src.utils.error_handling.logger')
    def test_resource_monitor_check_above_threshold(self, mock_logger, mock_get_info):
        """Test monitoring when above threshold."""
        mock_get_info.return_value = {"percent": 95.0}
        
        monitor = ResourceMonitor(device="cpu", memory_threshold=0.9, check_interval=0)
        warning = monitor.check_resources()
        
        assert warning is not None
        assert "95.0%" in warning
        mock_logger.warning.assert_called_once()
    
    def test_resource_monitor_rate_limiting(self):
        """Test check interval rate limiting."""
        monitor = ResourceMonitor(check_interval=10.0)
        
        # First check should work
        monitor._last_check = 0
        with patch('time.time', return_value=5.0):
            with patch('src.utils.error_handling.get_memory_info', return_value={"percent": 50}):
                result1 = monitor.check_resources()
        
        # Second check too soon should be skipped
        with patch('time.time', return_value=6.0):
            result2 = monitor.check_resources()
        
        assert result1 is None  # Below threshold
        assert result2 is None  # Skipped due to rate limit


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_validate_configuration_valid(self):
        """Test validation with valid config."""
        config = {
            "batch_size": 32,
            "quantization_bits": 8,
            "dtype": "float16",
            "gradient_checkpointing": True,
        }
        
        issues = validate_configuration(config)
        assert len(issues) == 0
    
    def test_validate_configuration_large_batch_size(self):
        """Test validation with large batch size."""
        config = {"batch_size": 256}
        
        issues = validate_configuration(config)
        assert len(issues) == 1
        assert "batch size" in issues[0].lower()
    
    def test_validate_configuration_invalid_quantization(self):
        """Test validation with unusual quantization bits."""
        config = {"quantization_bits": 7}
        
        issues = validate_configuration(config)
        assert len(issues) == 1
        assert "quantization bits" in issues[0].lower()
    
    def test_validate_configuration_mps_bfloat16(self):
        """Test validation with incompatible dtype for MPS."""
        config = {
            "dtype": "bfloat16",
            "backend": "mps",
        }
        
        issues = validate_configuration(config)
        assert len(issues) == 1
        assert "BFloat16" in issues[0]
        assert "MPS" in issues[0]


class TestImportErrorHandling:
    """Test import error handling."""
    
    def test_handle_import_error_success(self):
        """Test successful import."""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_import.return_value = mock_module
            
            result = handle_import_error("test_module")
            assert result == mock_module
    
    def test_handle_import_error_with_fallback(self):
        """Test import error with fallback."""
        fallback_obj = Mock()
        
        with patch('importlib.import_module', side_effect=ImportError("Not found")):
            result = handle_import_error(
                "missing_module",
                fallback=lambda: fallback_obj
            )
            
            assert result == fallback_obj
    
    def test_handle_import_error_with_install_cmd(self):
        """Test import error with install command."""
        with patch('importlib.import_module', side_effect=ImportError("Not found")):
            with pytest.raises(ImportError) as exc_info:
                handle_import_error(
                    "missing_module",
                    install_cmd="pip install missing-module"
                )
            
            assert "pip install missing-module" in str(exc_info.value)


class TestBackendErrorHandling:
    """Test backend error handling."""
    
    def test_handle_backend_error_with_fallback(self):
        """Test backend error with fallback."""
        exc = RuntimeError("Backend failed")
        
        result = handle_backend_error(exc, "cuda", fallback_backend="cpu")
        assert result == "cpu"
    
    def test_handle_backend_error_no_fallback_cuda(self):
        """Test CUDA error suggests alternatives."""
        exc = RuntimeError("CUDA error")
        
        with patch('src.utils.error_handling.logger') as mock_logger:
            with pytest.raises(BackendError):
                handle_backend_error(exc, "cuda")
            
            # Check suggestions were logged
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("mps" in call or "cpu" in call for call in info_calls)
    
    def test_handle_backend_error_no_fallback_mps(self):
        """Test MPS error suggests CPU."""
        exc = RuntimeError("MPS error")
        
        with patch('src.utils.error_handling.logger') as mock_logger:
            with pytest.raises(BackendError):
                handle_backend_error(exc, "mps")
            
            # Check CPU was suggested
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("cpu" in call for call in info_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])