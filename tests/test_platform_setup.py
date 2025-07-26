"""
Tests for platform detection and setup functionality.
"""

import os
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup import PlatformDetector, DependencyInstaller


class TestPlatformDetector(unittest.TestCase):
    """Test the PlatformDetector class."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = PlatformDetector()
    
    def test_initialization(self):
        """Test that PlatformDetector initializes correctly."""
        self.assertIsNotNone(self.detector.system)
        self.assertIsNotNone(self.detector.machine)
        self.assertIsNotNone(self.detector.python_version)
    
    @patch('subprocess.run')
    def test_detect_cuda_success(self, mock_run):
        """Test CUDA detection when CUDA is available."""
        # Mock nvidia-smi success
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout='515.65.01'),  # nvidia-smi
            MagicMock(returncode=0, stdout='nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2022 NVIDIA Corporation\nBuilt on Thu_Feb_10_18:23:41_Pacific_Standard_Time_2022\nCuda compilation tools, release 11.8, V11.8.89')  # nvcc
        ]
        
        cuda_version = self.detector.detect_cuda()
        self.assertEqual(cuda_version, '11.8')
    
    @patch('subprocess.run')
    def test_detect_cuda_no_nvcc(self, mock_run):
        """Test CUDA detection when nvcc is not available."""
        # Mock nvidia-smi success but nvcc failure
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout='515.65.01'),  # nvidia-smi
            MagicMock(returncode=1, stdout='')  # nvcc not found
        ]
        
        cuda_version = self.detector.detect_cuda()
        self.assertEqual(cuda_version, '11.8')  # Default version
    
    @patch('subprocess.run')
    def test_detect_cuda_not_available(self, mock_run):
        """Test CUDA detection when CUDA is not available."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'nvidia-smi')
        
        cuda_version = self.detector.detect_cuda()
        self.assertIsNone(cuda_version)
    
    @patch.object(PlatformDetector, 'detect_cuda')
    def test_detect_platform_cuda(self, mock_cuda):
        """Test platform detection returns CUDA when available."""
        mock_cuda.return_value = '11.8'
        
        platform_type, metadata = self.detector.detect_platform()
        
        self.assertEqual(platform_type, 'cuda')
        self.assertEqual(metadata['cuda_version'], '11.8')
        self.assertEqual(metadata['cuda_short'], '118')
    
    @patch.object(PlatformDetector, 'detect_cuda')
    @patch('platform.system')
    @patch('platform.machine')
    def test_detect_platform_mac(self, mock_machine, mock_system, mock_cuda):
        """Test platform detection for Apple Silicon."""
        mock_cuda.return_value = None
        mock_system.return_value = 'Darwin'
        mock_machine.return_value = 'arm64'
        
        detector = PlatformDetector()
        platform_type, metadata = detector.detect_platform()
        
        self.assertEqual(platform_type, 'mac')
    
    @patch.object(PlatformDetector, 'detect_cuda')
    @patch('platform.system')
    @patch('platform.machine')
    def test_detect_platform_cpu(self, mock_machine, mock_system, mock_cuda):
        """Test platform detection defaults to CPU."""
        mock_cuda.return_value = None
        mock_system.return_value = 'Linux'
        mock_machine.return_value = 'x86_64'
        
        detector = PlatformDetector()
        platform_type, metadata = detector.detect_platform()
        
        self.assertEqual(platform_type, 'cpu')


class TestDependencyInstaller(unittest.TestCase):
    """Test the DependencyInstaller class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create dummy requirements files
        (self.project_root / 'requirements.txt').write_text('torch>=2.2.0\n')
        (self.project_root / 'requirements-cuda.txt').write_text('torch>=2.2.0\nbitsandbytes>=0.43.0\n')
        (self.project_root / 'requirements-mac.txt').write_text('torch>=2.2.0\nmlx>=0.9.0\n')
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_requirements_file_cuda(self):
        """Test getting CUDA requirements file."""
        installer = DependencyInstaller('cuda', {'cuda_version': '11.8'})
        installer.project_root = self.project_root
        
        req_file = installer.get_requirements_file()
        self.assertEqual(req_file.name, 'requirements-cuda.txt')
    
    def test_get_requirements_file_mac(self):
        """Test getting Mac requirements file."""
        installer = DependencyInstaller('mac', {})
        installer.project_root = self.project_root
        
        req_file = installer.get_requirements_file()
        self.assertEqual(req_file.name, 'requirements-mac.txt')
    
    def test_get_requirements_file_cpu(self):
        """Test getting CPU requirements file."""
        installer = DependencyInstaller('cpu', {})
        installer.project_root = self.project_root
        
        req_file = installer.get_requirements_file()
        self.assertEqual(req_file.name, 'requirements.txt')
    
    @patch('subprocess.run')
    @patch('sys.executable', '/usr/bin/python3')
    def test_install_requirements_cuda(self, mock_run):
        """Test installing CUDA requirements."""
        installer = DependencyInstaller('cuda', {'cuda_short': '118'})
        installer.project_root = self.project_root
        
        installer.install_requirements()
        
        # Check that subprocess.run was called with correct arguments
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        
        self.assertIn('/usr/bin/python3', args)
        self.assertIn('-m', args)
        self.assertIn('pip', args)
        self.assertIn('install', args)
        self.assertIn(str(self.project_root / 'requirements-cuda.txt'), ' '.join(args))
        self.assertIn('https://download.pytorch.org/whl/cu118', ' '.join(args))
    
    @patch('subprocess.run')
    def test_install_requirements_with_upgrade(self, mock_run):
        """Test installing requirements with upgrade flag."""
        installer = DependencyInstaller('cpu', {})
        installer.project_root = self.project_root
        
        installer.install_requirements(upgrade=True)
        
        args = mock_run.call_args[0][0]
        self.assertIn('--upgrade', args)
    
    def test_verify_installation_success(self):
        """Test verification when all dependencies are installed."""
        # Import needed modules before patching
        from io import StringIO
        import sys
        
        # Mock successful imports
        mock_torch = MagicMock()
        mock_torch.__version__ = '2.2.0'
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        
        mock_transformers = MagicMock()
        mock_transformers.__version__ = '4.40.0'
        
        # Store original __import__ for non-mocked imports
        original_import = __builtins__['__import__']
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'torch':
                return mock_torch
            elif name == 'transformers':
                return mock_transformers
            elif name in ['accelerate', 'safetensors']:
                mock_module = MagicMock()
                mock_module.__version__ = '1.0.0'
                return mock_module
            else:
                return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=import_side_effect):
            installer = DependencyInstaller('cuda', {})
            
            # Capture print output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            installer.verify_installation()
            
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            self.assertIn('PyTorch 2.2.0 installed', output)
            self.assertIn('CUDA support detected', output)
            self.assertIn('transformers 4.40.0 installed', output)
    
    def test_verify_installation_missing_deps(self):
        """Test verification when dependencies are missing."""
        # Import needed modules before patching
        from io import StringIO
        import sys
        
        # Store original __import__ for non-mocked imports
        original_import = __builtins__['__import__']
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'torch':
                mock_torch = MagicMock()
                mock_torch.__version__ = '2.2.0'
                mock_torch.cuda.is_available.return_value = False
                return mock_torch
            elif name in ['transformers', 'accelerate', 'safetensors']:
                raise ImportError(f"No module named '{name}'")
            else:
                return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=import_side_effect):
            installer = DependencyInstaller('cuda', {})
            
            # Capture print output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            installer.verify_installation()
            
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            self.assertIn('PyTorch 2.2.0 installed', output)
            self.assertIn('CUDA support not detected', output)
            self.assertIn('transformers not installed', output)


class TestRequirementsFiles(unittest.TestCase):
    """Test that requirements files are properly formatted."""
    
    def test_requirements_txt_exists(self):
        """Test that requirements.txt exists."""
        req_file = Path(__file__).parent.parent / 'requirements.txt'
        self.assertTrue(req_file.exists(), "requirements.txt should exist")
    
    def test_requirements_mac_txt_exists(self):
        """Test that requirements-mac.txt exists."""
        req_file = Path(__file__).parent.parent / 'requirements-mac.txt'
        self.assertTrue(req_file.exists(), "requirements-mac.txt should exist")
    
    def test_requirements_cuda_txt_exists(self):
        """Test that requirements-cuda.txt exists."""
        req_file = Path(__file__).parent.parent / 'requirements-cuda.txt'
        self.assertTrue(req_file.exists(), "requirements-cuda.txt should exist")
    
    def test_requirements_syntax(self):
        """Test that requirements files have valid syntax."""
        req_files = [
            'requirements.txt',
            'requirements-mac.txt',
            'requirements-cuda.txt'
        ]
        
        for filename in req_files:
            req_file = Path(__file__).parent.parent / filename
            if req_file.exists():
                content = req_file.read_text()
                lines = content.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Skip special pip options
                    if line.startswith('--'):
                        continue
                    
                    # Check that the line contains a package name
                    self.assertTrue(
                        any(c.isalnum() for c in line),
                        f"Invalid line in {filename}: {line}"
                    )


class TestSetupScript(unittest.TestCase):
    """Test the setup.py script functionality."""
    
    def test_setup_script_exists(self):
        """Test that setup.py exists and is executable."""
        setup_file = Path(__file__).parent.parent / 'setup.py'
        self.assertTrue(setup_file.exists(), "setup.py should exist")
        
        # Check if file is executable (on Unix-like systems)
        if platform.system() != 'Windows':
            import stat
            mode = setup_file.stat().st_mode
            self.assertTrue(
                mode & stat.S_IXUSR,
                "setup.py should be executable"
            )
    
    def test_setup_script_imports(self):
        """Test that setup.py can be imported without errors."""
        setup_file = Path(__file__).parent.parent / 'setup.py'
        
        # Read the file and check for syntax errors
        try:
            compile(setup_file.read_text(), 'setup.py', 'exec')
        except SyntaxError as e:
            self.fail(f"Syntax error in setup.py: {e}")


if __name__ == '__main__':
    unittest.main()