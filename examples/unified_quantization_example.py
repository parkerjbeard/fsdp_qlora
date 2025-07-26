#!/usr/bin/env python3
"""
Example of using the unified quantization interface.

This example demonstrates:
1. Basic tensor quantization
2. Model quantization
3. Backend selection
4. Mixed precision quantization
"""

import torch
import torch.nn as nn
from src.utils.unified_quantization import (
    QuantizationBackend,
    UnifiedQuantizationConfig,
    UnifiedQuantizer,
    quantize_model,
)


def example_basic_quantization():
    """Example of basic tensor quantization."""
    print("=== Basic Tensor Quantization ===")
    
    # Create a simple tensor
    tensor = torch.randn(256, 128)
    
    # Configure quantization
    config = UnifiedQuantizationConfig(
        backend=QuantizationBackend.AUTO,  # Automatically select best backend
        bits=8,
        group_size=64,
    )
    
    # Create quantizer
    quantizer = UnifiedQuantizer(config)
    print(f"Selected backend: {quantizer.backend}")
    
    # Quantize tensor
    quantized, params = quantizer.quantize(tensor)
    print(f"Original tensor size: {tensor.numel() * tensor.element_size()} bytes")
    print(f"Quantized tensor size: {quantized.numel() * quantized.element_size()} bytes")
    print(f"Compression ratio: {tensor.numel() * tensor.element_size() / (quantized.numel() * quantized.element_size()):.2f}x")
    
    # Dequantize
    dequantized = quantizer.dequantize(quantized, params)
    error = torch.mean(torch.abs(tensor - dequantized)).item()
    print(f"Mean absolute error: {error:.6f}\n")


def example_model_quantization():
    """Example of quantizing a neural network model."""
    print("=== Model Quantization ===")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    
    # Print original model size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
    
    # Quantize model
    quantized_model, quantizer = quantize_model(
        model,
        bits=8,
        backend=QuantizationBackend.AUTO,
        skip_modules=["4"],  # Skip the output layer
    )
    
    # Test the quantized model
    x = torch.randn(32, 784)
    with torch.no_grad():
        original_output = model(x)
        quantized_output = quantized_model(x)
    
    # Compare outputs
    output_error = torch.mean(torch.abs(original_output - quantized_output)).item()
    print(f"Mean output error: {output_error:.6f}\n")


def example_mixed_precision():
    """Example of mixed precision quantization."""
    print("=== Mixed Precision Quantization ===")
    
    # Create a transformer-like model
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 256)
            self.attention = nn.MultiheadAttention(256, 8)
            self.mlp = nn.Sequential(
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
            )
            self.output = nn.Linear(256, 1000)
        
        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.attention(x, x, x)
            x = self.mlp(x)
            return self.output(x)
    
    model = SimpleTransformer()
    
    # Configure mixed precision
    config = UnifiedQuantizationConfig(
        backend=QuantizationBackend.AUTO,
        bits=4,  # Default 4-bit
        layer_bits={
            "embedding": 8,  # 8-bit for embeddings
            "attention": 8,  # 8-bit for attention
            "output": 16,    # 16-bit for output layer
        },
        skip_modules=["output"],  # Don't quantize output layer
        group_size=128,
    )
    
    # Quantize
    quantizer = UnifiedQuantizer(config)
    quantized_model = quantizer.quantize_model(model)
    
    print("Mixed precision configuration applied:")
    print("- Embedding: 8-bit")
    print("- Attention: 8-bit") 
    print("- MLP: 4-bit (default)")
    print("- Output: Not quantized (skipped)\n")


def example_save_load():
    """Example of saving and loading quantized models."""
    print("=== Save/Load Quantized Model ===")
    
    # Create and quantize a model
    model = nn.Linear(128, 64)
    config = UnifiedQuantizationConfig(bits=8)
    quantizer = UnifiedQuantizer(config)
    quantized_model = quantizer.quantize_model(model)
    
    # Save
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "quantized_model")
        quantizer.save_model(quantized_model, save_path)
        print(f"Model saved to {save_path}")
        
        # Load
        loaded_model = quantizer.load_model(
            save_path,
            model_class=lambda: nn.Linear(128, 64)
        )
        print("Model loaded successfully!")
        
        # Verify
        x = torch.randn(10, 128)
        with torch.no_grad():
            out1 = quantized_model(x)
            out2 = loaded_model(x)
        
        if torch.allclose(out1, out2):
            print("✓ Loaded model produces identical outputs\n")


if __name__ == "__main__":
    # Run all examples
    example_basic_quantization()
    example_model_quantization()
    example_mixed_precision()
    example_save_load()
    
    print("All examples completed successfully!")