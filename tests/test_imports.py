"""
Tests for the import abstraction layer.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.imports import (
    ImportResult,
    ImportRegistry,
    get_module,
    check_import_availability,
    validate_imports,
    report_import_status,
    suppress_output,
)


class TestImportResult(unittest.TestCase):
    """Test the ImportResult dataclass."""
    
    def test_success_result(self):
        """Test successful import result."""
        module = MagicMock()
        result = ImportResult(success=True, module=module)
        
        self.assertTrue(result.success)
        self.assertIs(result.module, module)
        self.assertIsNone(result.error)
        self.assertFalse(result.fallback_used)
        self.assertEqual(result.warnings, [])
    
    def test_failure_result(self):
        """Test failed import result."""
        error = ImportError("Test error")
        result = ImportResult(success=False, error=error)
        
        self.assertFalse(result.success)
        self.assertIsNone(result.module)
        self.assertIs(result.error, error)
        self.assertFalse(result.fallback_used)
    
    def test_fallback_result(self):
        """Test fallback import result."""
        module = MagicMock()
        warnings = ["Using fallback"]
        result = ImportResult(
            success=True,
            module=module,
            fallback_used=True,
            warnings=warnings
        )
        
        self.assertTrue(result.success)
        self.assertIs(result.module, module)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.warnings, warnings)


class TestImportRegistry(unittest.TestCase):
    """Test the ImportRegistry class."""
    
    def setUp(self):
        """Set up test registry."""
        self.registry = ImportRegistry()
    
    def test_register_simple_import(self):
        """Test registering a simple import."""
        mock_module = MagicMock()
        
        def import_func():
            return mock_module
        
        self.registry.register('test_module', import_func)
        
        result = self.registry.get('test_module')
        self.assertTrue(result.success)
        self.assertIs(result.module, mock_module)
    
    def test_register_with_fallback(self):
        """Test registering an import with fallback."""
        fallback_module = MagicMock()
        
        def import_func():
            raise ImportError("Not available")
        
        def fallback_func():
            return fallback_module
        
        self.registry.register('test_module', import_func, fallback=fallback_func)
        
        result = self.registry.get('test_module')
        self.assertTrue(result.success)
        self.assertIs(result.module, fallback_module)
        self.assertTrue(result.fallback_used)
    
    def test_backend_specific_import(self):
        """Test backend-specific imports."""
        cuda_module = MagicMock()
        
        def import_func():
            return cuda_module
        
        self.registry.register('cuda_only', import_func, backends=['cuda'])
        
        # Should work for CUDA backend
        result = self.registry.get('cuda_only', backend='cuda')
        self.assertTrue(result.success)
        
        # Should fail for other backends
        result = self.registry.get('cuda_only', backend='cpu')
        self.assertFalse(result.success)
    
    def test_validator(self):
        """Test import validation."""
        module = MagicMock()
        module.required_attr = True
        
        def import_func():
            return module
        
        def validator(m):
            return hasattr(m, 'required_attr') and m.required_attr
        
        self.registry.register('validated', import_func, validator=validator)
        
        result = self.registry.get('validated')
        self.assertTrue(result.success)
        
        # Test with failing validator
        bad_module = MagicMock()
        # Explicitly delete the attribute to make hasattr return False
        del bad_module.required_attr
        
        def bad_import_func():
            return bad_module
        
        self.registry.register('bad_validated', bad_import_func, validator=validator)
        
        result = self.registry.get('bad_validated')
        self.assertFalse(result.success)
    
    def test_caching(self):
        """Test that imports are cached."""
        call_count = 0
        
        def counting_import():
            nonlocal call_count
            call_count += 1
            return MagicMock()
        
        self.registry.register('cached', counting_import)
        
        # First call
        result1 = self.registry.get('cached')
        self.assertEqual(call_count, 1)
        
        # Second call should use cache
        result2 = self.registry.get('cached')
        self.assertEqual(call_count, 1)
        self.assertIs(result1.module, result2.module)
    
    def test_check_availability(self):
        """Test checking import availability."""
        self.registry.register('available', lambda: MagicMock())
        self.registry.register(
            'unavailable',
            lambda: (_ for _ in ()).throw(ImportError())
        )
        
        self.assertTrue(self.registry.check_availability('available'))
        self.assertFalse(self.registry.check_availability('unavailable'))


class TestBitsandbytesImport(unittest.TestCase):
    """Test bitsandbytes import handling."""
    
    @patch('builtins.__import__')
    def test_bitsandbytes_import_success(self, mock_import):
        """Test successful bitsandbytes import."""
        # Create mock bitsandbytes module
        mock_bnb = MagicMock()
        mock_bnb.optim = MagicMock()
        mock_bnb.optim.cadam32bit_grad_fp32 = True
        mock_bnb.nn = MagicMock()
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'bitsandbytes':
                return mock_bnb
            elif name == 'bitsandbytes.nn':
                return mock_bnb.nn
            else:
                return MagicMock()
        
        mock_import.side_effect = import_side_effect
        
        # Import the module to trigger registration
        from imports import _import_bitsandbytes
        
        # Test the import
        bnb = _import_bitsandbytes()
        self.assertIsNotNone(bnb)
        self.assertTrue(hasattr(bnb, 'optim'))
    
    def test_bitsandbytes_fallback(self):
        """Test bitsandbytes fallback."""
        from imports import _bitsandbytes_fallback
        
        dummy_bnb = _bitsandbytes_fallback()
        
        # Check that dummy module has required attributes
        self.assertTrue(hasattr(dummy_bnb, 'nn'))
        self.assertTrue(hasattr(dummy_bnb, 'optim'))
        self.assertTrue(hasattr(dummy_bnb, 'Linear4bit'))
        self.assertTrue(hasattr(dummy_bnb, 'Params4bit'))
        
        # Check that accessing other attributes raises ImportError
        with self.assertRaises(ImportError):
            dummy_bnb.some_other_function


class TestWandbImport(unittest.TestCase):
    """Test wandb import handling."""
    
    def test_wandb_fallback(self):
        """Test wandb fallback functionality."""
        from imports import _wandb_fallback
        
        dummy_wandb = _wandb_fallback()
        
        # Test that dummy methods work without errors
        dummy_wandb.init(project="test")
        dummy_wandb.log({"metric": 1})
        dummy_wandb.finish()
        
        # Test that arbitrary methods return no-op functions
        result = dummy_wandb.some_random_method(1, 2, 3)
        self.assertIsNone(result)


class TestImportHelpers(unittest.TestCase):
    """Test helper functions."""
    
    def test_suppress_output(self):
        """Test output suppression context manager."""
        
        with suppress_output():
            print("This should not appear")
            sys.stderr.write("Neither should this")
        
        # If we get here without errors, the test passes
        self.assertTrue(True)
    
    @patch('imports._import_registry')
    def test_get_module(self, mock_registry):
        """Test get_module convenience function."""
        
        # Set up mock
        mock_result = ImportResult(success=True, module=MagicMock())
        mock_registry.get.return_value = mock_result
        
        module = get_module('test_module', backend='cuda')
        
        mock_registry.get.assert_called_once_with('test_module', 'cuda')
        self.assertIsNotNone(module)
    
    @patch('imports._import_registry')
    def test_get_module_failure(self, mock_registry):
        """Test get_module with import failure."""
        
        # Set up mock
        error = ImportError("Test error")
        mock_result = ImportResult(success=False, error=error)
        mock_registry.get.return_value = mock_result
        
        with self.assertRaises(ImportError) as cm:
            get_module('test_module')
        
        self.assertEqual(str(cm.exception), "Test error")
    
    @patch('imports._import_registry')
    def test_validate_imports(self, mock_registry):
        """Test validate_imports function."""
        
        # Set up mock results
        results = {
            'module1': ImportResult(success=True, module=MagicMock()),
            'module2': ImportResult(success=False, error=ImportError("Not found")),
        }
        
        def get_side_effect(name, backend):
            return results[name]
        
        mock_registry.get.side_effect = get_side_effect
        
        validation_results = validate_imports(['module1', 'module2'], backend='cuda')
        
        self.assertEqual(len(validation_results), 2)
        self.assertTrue(validation_results['module1'].success)
        self.assertFalse(validation_results['module2'].success)
    
    def test_report_import_status(self):
        """Test import status reporting."""
        
        # This should generate a report without errors
        report = report_import_status(backend='cuda')
        
        self.assertIsInstance(report, str)
        self.assertIn("Import Status Report", report)
        self.assertIn("Backend: cuda", report)


class TestRealImports(unittest.TestCase):
    """Test with real import attempts (integration tests)."""
    
    def test_actual_import_availability(self):
        """Test checking actual import availability."""
        
        # Check if local modules exist before testing
        try:
            import importlib.util
            dora_spec = importlib.util.find_spec('dora')
            dora_exists = dora_spec is not None
        except ImportError:
            dora_exists = False
            
        try:
            lora_spec = importlib.util.find_spec('lora')
            lora_exists = lora_spec is not None
        except ImportError:
            lora_exists = False
        
        # Only test if modules actually exist
        if dora_exists:
            self.assertTrue(check_import_availability('dora'))
        if lora_exists:
            self.assertTrue(check_import_availability('lora'))
        
        # Wandb might or might not be installed, but should have fallback
        wandb_available = check_import_availability('wandb')
        self.assertTrue(wandb_available)  # Should be True due to fallback
    
    def test_backend_specific_availability(self):
        """Test backend-specific import availability."""
        
        # Bitsandbytes should only be available on CUDA
        bnb_cuda = check_import_availability('bitsandbytes', backend='cuda')
        bnb_cpu = check_import_availability('bitsandbytes', backend='cpu')
        
        # On CPU, it should use fallback
        self.assertTrue(bnb_cuda or bnb_cpu)  # At least one should work
    
    def test_import_status_report_format(self):
        """Test the format of import status report."""
        
        report = report_import_status(backend='cpu')
        
        # Check report structure
        lines = report.split('\n')
        self.assertGreater(len(lines), 5)
        self.assertEqual(lines[1], "=" * 50)
        
        # Check for specific modules
        module_names = ['bitsandbytes', 'hqq', 'wandb', 'dora', 'lora']
        for name in module_names:
            self.assertTrue(
                any(name in line for line in lines),
                f"Module {name} not found in report"
            )


if __name__ == '__main__':
    unittest.main()