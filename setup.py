#!/usr/bin/env python3
"""
Platform-aware setup script for FSDP QLoRA

This script detects your system platform and installs the appropriate dependencies.
It handles CUDA, Apple Silicon (MPS), and CPU installations.
"""

import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


class PlatformDetector:
    """Detect platform and available hardware accelerators."""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.python_version = sys.version_info
        
    def detect_cuda(self) -> Optional[str]:
        """Detect CUDA installation and version."""
        try:
            # Try nvidia-smi first
            subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Try to get CUDA version
            cuda_result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True
            )
            
            if cuda_result.returncode == 0:
                # Parse CUDA version from nvcc output
                output = cuda_result.stdout
                for line in output.split('\n'):
                    if 'release' in line:
                        parts = line.split('release')[-1].split(',')[0].strip()
                        return parts
            
            # If nvcc not found, assume CUDA 11.8 as default
            return "11.8"
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def detect_platform(self) -> tuple[str, dict]:
        """
        Detect platform and return platform type and metadata.
        
        Returns:
            Tuple of (platform_type, metadata_dict)
        """
        metadata = {
            'system': self.system,
            'machine': self.machine,
            'python_version': f"{self.python_version.major}.{self.python_version.minor}"
        }
        
        # Check for CUDA first (highest priority)
        cuda_version = self.detect_cuda()
        if cuda_version:
            metadata['cuda_version'] = cuda_version
            metadata['cuda_short'] = cuda_version.replace('.', '')[:3]  # e.g., "11.8" -> "118"
            return 'cuda', metadata
        
        # Check for Apple Silicon
        if self.system == 'Darwin' and self.machine == 'arm64':
            return 'mac', metadata
        
        # Default to CPU
        return 'cpu', metadata


class DependencyInstaller:
    """Install platform-specific dependencies."""
    
    def __init__(self, platform_type: str, metadata: dict):
        self.platform_type = platform_type
        self.metadata = metadata
        self.project_root = Path(__file__).parent
        
    def get_requirements_file(self) -> Path:
        """Get the appropriate requirements file for the platform."""
        if self.platform_type == 'cuda':
            return self.project_root / 'requirements-cuda.txt'
        elif self.platform_type == 'mac':
            return self.project_root / 'requirements-mac.txt'
        else:
            return self.project_root / 'requirements.txt'
    
    def install_requirements(self, upgrade: bool = False) -> None:
        """Install requirements from the appropriate file."""
        req_file = self.get_requirements_file()
        
        if not req_file.exists():
            print(f"Warning: {req_file} not found. Using base requirements.txt")
            req_file = self.project_root / 'requirements.txt'
        
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(req_file)]
        
        if upgrade:
            cmd.append('--upgrade')
        
        # Add CUDA-specific index URL if needed
        if self.platform_type == 'cuda':
            cuda_short = self.metadata.get('cuda_short', '118')
            cmd.extend(['--extra-index-url', f'https://download.pytorch.org/whl/cu{cuda_short}'])
        
        print(f"Installing dependencies from {req_file.name}...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print("✓ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing dependencies: {e}")
            sys.exit(1)
    
    def install_optional_dependencies(self) -> None:
        """Install optional platform-specific dependencies."""
        optional_deps = {
            'cuda': [
                ('flash-attn', 'Flash Attention 2 for improved performance'),
                ('apex', 'NVIDIA Apex for mixed precision training'),
            ],
            'mac': [
                ('mlx', 'MLX framework for Apple Silicon'),
                ('mlx-lm', 'MLX language modeling utilities'),
            ]
        }
        
        deps = optional_deps.get(self.platform_type, [])
        
        if deps:
            print("\nOptional dependencies available:")
            for dep, desc in deps:
                response = input(f"Install {dep} ({desc})? [y/N]: ").strip().lower()
                if response == 'y':
                    try:
                        subprocess.run(
                            [sys.executable, '-m', 'pip', 'install', dep],
                            check=True
                        )
                        print(f"✓ {dep} installed successfully!")
                    except subprocess.CalledProcessError:
                        print(f"✗ Failed to install {dep}")
    
    def verify_installation(self) -> None:
        """Verify that key dependencies are properly installed."""
        print("\nVerifying installation...")
        
        # Check PyTorch
        try:
            import torch
            print(f"✓ PyTorch {torch.__version__} installed")
            
            if self.platform_type == 'cuda':
                if torch.cuda.is_available():
                    print(f"✓ CUDA support detected (GPUs: {torch.cuda.device_count()})")
                else:
                    print("✗ CUDA support not detected in PyTorch")
            elif self.platform_type == 'mac':
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    print("✓ MPS (Metal Performance Shaders) support detected")
                else:
                    print("✗ MPS support not detected")
        except ImportError:
            print("✗ PyTorch not installed properly")
        
        # Check other key dependencies
        deps_to_check = ['transformers', 'accelerate', 'safetensors']
        for dep in deps_to_check:
            try:
                module = __import__(dep)
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {dep} {version} installed")
            except ImportError:
                print(f"✗ {dep} not installed")


def main():
    """Main setup function."""
    print("=" * 60)
    print("FSDP QLoRA Platform-Aware Setup")
    print("=" * 60)
    
    # Detect platform
    detector = PlatformDetector()
    platform_type, metadata = detector.detect_platform()
    
    print(f"\nDetected Platform: {platform_type.upper()}")
    print(f"System: {metadata['system']}")
    print(f"Architecture: {metadata['machine']}")
    print(f"Python: {metadata['python_version']}")
    
    if 'cuda_version' in metadata:
        print(f"CUDA Version: {metadata['cuda_version']}")
    
    # Ask user to confirm or override
    print("\nAvailable platforms:")
    print("1. CUDA (NVIDIA GPUs)")
    print("2. Mac (Apple Silicon)")
    print("3. CPU (Generic)")
    
    choice = input(f"\nPress Enter to continue with {platform_type.upper()} or select a number to override: ").strip()
    
    if choice:
        platform_map = {'1': 'cuda', '2': 'mac', '3': 'cpu'}
        if choice in platform_map:
            platform_type = platform_map[choice]
            print(f"\nOverriding to: {platform_type.upper()}")
        else:
            print("Invalid choice, continuing with detected platform")
    
    # Install dependencies
    installer = DependencyInstaller(platform_type, metadata)
    
    # Check if we should upgrade
    upgrade = '--upgrade' in sys.argv or '-U' in sys.argv
    
    installer.install_requirements(upgrade=upgrade)
    installer.install_optional_dependencies()
    installer.verify_installation()
    
    # Print backend manager info
    print("\n" + "=" * 60)
    print("Backend Manager Configuration")
    print("=" * 60)
    
    try:
        from backend_manager import BackendManager
        BackendManager(verbose=True)
    except Exception as e:
        print(f"Note: Backend manager not available yet: {e}")
    
    print("\n✓ Setup complete! You can now run training with:")
    print(f"  python train.py --backend {platform_type if platform_type != 'cpu' else 'auto'}")
    
    # Create a setup marker file
    marker_file = Path('.setup_complete')
    with open(marker_file, 'w') as f:
        f.write(f"platform={platform_type}\n")
        f.write(f"timestamp={subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n")


if __name__ == "__main__":
    main()