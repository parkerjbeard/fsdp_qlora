"""
Example: Using the Model Loading Abstraction

This example demonstrates how to use the new model loading abstraction
to simplify model loading across different backends and configurations.
"""

import torch
from backend_manager import BackendManager
from model_loader import (
    ModelLoadingConfig,
    ModelLoaderFactory,
    load_model_and_tokenizer,
    get_recommended_loader_config,
)
from quantization_wrapper import QuantizationConfig, QuantizationMethod


def example_simple_loading():
    """Example 1: Simple model loading with auto-detection."""
    print("Example 1: Simple Model Loading")
    print("-" * 50)
    
    # Simplest way - auto-detects backend and loads model
    model, tokenizer = load_model_and_tokenizer(
        "meta-llama/Llama-2-7b-hf",
        backend="auto",  # Auto-detect best backend
        dtype=torch.float16,
    )
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print()


def example_quantized_loading():
    """Example 2: Loading with quantization."""
    print("Example 2: Quantized Model Loading")
    print("-" * 50)
    
    # Create quantization config
    quant_config = QuantizationConfig(
        method=QuantizationMethod.BNB_NF4,
        bits=4,
        compute_dtype=torch.bfloat16,
        double_quant=True,
    )
    
    # Load quantized model
    model, tokenizer = load_model_and_tokenizer(
        "meta-llama/Llama-2-7b-hf",
        backend="cuda",
        quantization_config=quant_config,
        low_memory=True,
        verbose=True,
    )
    
    print("Quantized model loaded with 4-bit precision")
    print()


def example_backend_specific():
    """Example 3: Backend-specific loading."""
    print("Example 3: Backend-Specific Loading")
    print("-" * 50)
    
    # Detect backend
    backend_manager = BackendManager(verbose=True)
    
    # Get recommended configuration for the backend
    config = get_recommended_loader_config(
        "meta-llama/Llama-2-7b-hf",
        backend=backend_manager.backend,
        available_memory_gb=8.0,  # Simulate 8GB available
    )
    
    print(f"Recommended loading strategy: {config.loading_strategy}")
    if config.quantization_config:
        print(f"Recommended quantization: {config.quantization_config.method}")
    
    # Create loader and load model
    loader = ModelLoaderFactory.create_loader(config)
    model = loader.load_model()
    tokenizer = loader.load_tokenizer()
    
    print("Model loaded with recommended configuration")
    print()


def example_custom_config():
    """Example 4: Custom configuration."""
    print("Example 4: Custom Configuration")
    print("-" * 50)
    
    # Create custom configuration
    config = ModelLoadingConfig(
        model_name="meta-llama/Llama-2-70b-hf",
        backend=BackendManager().backend,
        loading_strategy="low_memory",
        quantization_config=QuantizationConfig(
            method=QuantizationMethod.HQQ,
            bits=8,
            group_size=128,
        ),
        dtype=torch.float16,
        low_memory=True,
        loading_workers=4,
        verbose=True,
    )
    
    # Create loader
    loader = ModelLoaderFactory.create_loader(config)
    
    print(f"Using loader: {loader.__class__.__name__}")
    print(f"Loading strategy: {config.loading_strategy}")
    print(f"Quantization: {config.quantization_config.method}")
    print(f"Loading workers: {config.loading_workers}")
    print()


def example_train_py_integration():
    """Example 5: Integration with train.py workflow."""
    print("Example 5: Train.py Integration")
    print("-" * 50)
    
    # Simulate train.py arguments
    class Args:
        model_name = "meta-llama/Llama-2-7b-hf"
        train_type = "qlora"
        precision = "bf16"
        low_memory = True
        q_bits = 4
        q_group_size = 64
        loading_workers = -1
        rank = 0
        world_size = 1
        verbose = True
    
    args = Args()
    
    # Map train_type to quantization config
    quant_config = None
    if args.train_type in ["qlora", "custom_qlora"]:
        quant_config = QuantizationConfig(
            method=QuantizationMethod.BNB_NF4,
            bits=args.q_bits,
            group_size=args.q_group_size,
            compute_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16,
        )
    elif args.train_type in ["hqq_lora", "hqq_dora"]:
        quant_config = QuantizationConfig(
            method=QuantizationMethod.HQQ,
            bits=args.q_bits,
            group_size=args.q_group_size,
        )
    
    # Create loading config
    backend_manager = BackendManager(verbose=args.verbose)
    config = ModelLoadingConfig(
        model_name=args.model_name,
        backend=backend_manager.backend,
        quantization_config=quant_config,
        dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16,
        low_memory=args.low_memory,
        loading_workers=args.loading_workers,
        rank=args.rank,
        world_size=args.world_size,
        verbose=args.verbose,
    )
    
    # This single line replaces ~100+ lines of model loading logic
    loader = ModelLoaderFactory.create_loader(config)
    
    print(f"Train type: {args.train_type}")
    print(f"Selected loader: {loader.__class__.__name__}")
    print(f"Backend: {config.backend}")
    print(f"Quantization: {config.quantization_config.method if config.quantization_config else 'None'}")
    print()
    
    # Load model and tokenizer
    # model = loader.load_model()
    # tokenizer = loader.load_tokenizer()
    print("Model would be loaded here in actual usage")


def main():
    """Run all examples."""
    print("Model Loading Abstraction Examples")
    print("=" * 50)
    print()
    
    # Note: These examples show the API usage but don't actually load models
    # to avoid downloading large model files
    
    try:
        # example_simple_loading()
        # example_quantized_loading()
        example_backend_specific()
        example_custom_config()
        example_train_py_integration()
    except Exception as e:
        print(f"Note: Examples are for demonstration. Actual loading would require model files.")
        print(f"Error: {e}")
    
    print("\nFor actual usage, uncomment the model loading lines in the examples.")


if __name__ == "__main__":
    main()