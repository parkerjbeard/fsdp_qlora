"""
MPS Quantization Example

Demonstrates how to use MPS-optimized quantization for training models on Apple Silicon.
This example shows:
- Dynamic quantization strategy selection
- Training with quantized models
- Memory profiling and optimization
- Performance benchmarking
- Integration with existing training pipelines
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mps_quantization import (
    MPSQuantizationConfig,
    MPSQuantizationMethod,
    DynamicQuantizationStrategy,
    MPSQuantizationAdapter,
    MPSPerformanceOptimizer,
    create_mps_quantized_model,
    benchmark_quantization_methods,
)
from backend_manager import Backend, BackendManager


class TextClassificationDataset(Dataset):
    """Simple text classification dataset."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_sample_data(num_samples=1000):
    """Create sample text classification data."""
    # Sample texts and labels
    texts = [
        f"This is a {'positive' if i % 2 == 0 else 'negative'} example {i}."
        for i in range(num_samples)
    ]
    labels = [i % 2 for i in range(num_samples)]  # Binary classification
    
    return texts, labels


def demonstrate_dynamic_strategy():
    """Demonstrate dynamic quantization strategy selection."""
    print("\n" + "=" * 60)
    print("Dynamic Quantization Strategy Demo")
    print("=" * 60)
    
    # Create strategy
    strategy = DynamicQuantizationStrategy()
    
    # Test different model sizes
    model_sizes = [
        ("Small (350M)", 0.35),
        ("Medium (1.3B)", 1.3),
        ("Large (7B)", 7.0),
        ("XLarge (13B)", 13.0),
    ]
    
    print("\nQuantization recommendations based on model size:")
    print("-" * 60)
    
    for name, size_gb in model_sizes:
        config = strategy.select_quantization_config(
            model_size_gb=size_gb,
            target_dtype=torch.float16,
        )
        
        print(f"\n{name}:")
        print(f"  Method: {config.mps_method}")
        print(f"  Bits: {config.bits}")
        print(f"  Memory efficient: {config.memory_efficient}")
        print(f"  Chunk size: {config.chunk_size}")
    
    # Test model-specific optimizations
    print("\n\nModel-specific optimizations:")
    print("-" * 60)
    
    for model_type in ["llama", "gpt", "bert"]:
        config = strategy.optimize_for_model_type(model_type, 7.0)
        print(f"\n{model_type.upper()}:")
        print(f"  Per-channel quantization: {config.use_per_channel}")
        print(f"  Skip modules: {config.skip_modules}")


def demonstrate_quantization_methods(model):
    """Demonstrate different quantization methods."""
    print("\n" + "=" * 60)
    print("Quantization Methods Demo")
    print("=" * 60)
    
    methods = [
        ("No Quantization", None),
        ("PyTorch Dynamic (INT8)", MPSQuantizationMethod.PYTORCH_DYNAMIC),
        ("Fake Quantization", MPSQuantizationMethod.FAKE_QUANT),
    ]
    
    input_shape = (4, 128)  # batch_size, seq_length
    
    for name, method in methods:
        print(f"\n{name}:")
        
        if method is None:
            # Baseline - no quantization
            test_model = model.to('mps')
        else:
            # Apply quantization
            config = MPSQuantizationConfig(
                mps_method=method,
                bits=8,
                memory_efficient=True,
            )
            
            adapter = MPSQuantizationAdapter(Backend.MPS, config)
            test_model = adapter.quantize_model(model.clone())
            test_model = test_model.to('mps')
        
        # Measure performance
        optimizer = MPSPerformanceOptimizer(MPSQuantizationConfig())
        stats = optimizer.profile_quantized_model(
            test_model,
            input_shape,
            num_runs=20,
        )
        
        print(f"  Forward time: {stats['avg_forward_time_ms']:.2f}ms")
        print(f"  Memory allocated: {stats['allocated_memory_mb']:.1f}MB")
        print(f"  Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec")


def train_with_quantization(model, train_loader, config, num_epochs=3):
    """Train model with quantization."""
    print("\n" + "=" * 60)
    print(f"Training with {config.bits}-bit {config.mps_method} quantization")
    print("=" * 60)
    
    # Apply quantization
    adapter = MPSQuantizationAdapter(Backend.MPS, config)
    quantized_model = adapter.quantize_model(model)
    quantized_model = adapter.prepare_model_for_training(quantized_model)
    quantized_model = quantized_model.to('mps')
    
    # Training setup
    optimizer = torch.optim.AdamW(quantized_model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        quantized_model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to MPS
            input_ids = batch['input_ids'].to('mps')
            attention_mask = batch['attention_mask'].to('mps')
            labels = batch['labels'].to('mps')
            
            # Forward pass
            optimizer.zero_grad()
            outputs = quantized_model(input_ids, attention_mask=attention_mask)
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"  Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        # Epoch summary
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Memory stats
        memory_stats = adapter.strategy._get_memory_info()
        print(f"  Memory allocated: {memory_stats['allocated_gb']:.2f}GB")
    
    return quantized_model


def main(args):
    """Main demo function."""
    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("Error: MPS is not available on this system!")
        return
    
    print("MPS Quantization Demo")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Set device
    device = torch.device('mps')
    
    # Demonstrate dynamic strategy
    if args.demo_strategy:
        demonstrate_dynamic_strategy()
    
    # Create or load model
    if args.model_type == "transformer":
        # Simple transformer for demo
        from examples.mps_fsdp_example import SimpleTransformer
        model = SimpleTransformer(
            vocab_size=30000,
            hidden_size=768,
            num_layers=6,
            num_heads=12,
        )
        tokenizer = None
    else:
        # Load from HuggingFace
        print(f"\nLoading {args.model_name}...")
        try:
            model = AutoModel.from_pretrained(args.model_name)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Using simple transformer instead")
            from examples.mps_fsdp_example import SimpleTransformer
            model = SimpleTransformer()
            tokenizer = None
    
    # Calculate model size
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"\nModel size: {model_size_gb:.2f}GB")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Demonstrate quantization methods
    if args.demo_methods:
        demonstrate_quantization_methods(model)
    
    # Training demo
    if args.train:
        # Create sample dataset
        texts, labels = create_sample_data(args.num_samples)
        
        # Use simple tokenization if no tokenizer
        if tokenizer is None:
            # Simple word-to-id mapping
            vocab = list(set(' '.join(texts).split()))
            word_to_id = {word: i for i, word in enumerate(vocab)}
            
            class SimpleTokenizer:
                def __call__(self, text, **kwargs):
                    ids = [word_to_id.get(word, 0) for word in text.split()]
                    # Pad or truncate
                    max_len = kwargs.get('max_length', 128)
                    if len(ids) < max_len:
                        ids += [0] * (max_len - len(ids))
                    else:
                        ids = ids[:max_len]
                    
                    return {
                        'input_ids': torch.tensor([ids]),
                        'attention_mask': torch.tensor([[1 if i < len(ids) else 0 
                                                       for i in range(max_len)]]),
                    }
            
            tokenizer = SimpleTokenizer()
        
        dataset = TextClassificationDataset(texts, labels, tokenizer)
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # MPS works best with single process
        )
        
        # Select quantization configuration
        if args.auto_select:
            # Use dynamic strategy
            strategy = DynamicQuantizationStrategy()
            config = strategy.optimize_for_model_type(
                args.model_type,
                model_size_gb,
            )
            print(f"\nAuto-selected quantization: {config.bits}-bit {config.mps_method}")
        else:
            # Manual configuration
            config = MPSQuantizationConfig(
                mps_method=getattr(MPSQuantizationMethod, args.method),
                bits=args.bits,
                memory_efficient=args.memory_efficient,
                use_fast_math=args.fast_math,
            )
        
        # Train with quantization
        quantized_model = train_with_quantization(
            model,
            train_loader,
            config,
            num_epochs=args.epochs,
        )
        
        # Save model if requested
        if args.save_model:
            save_path = f"quantized_model_{config.bits}bit_{config.mps_method}.pt"
            adapter = MPSQuantizationAdapter(Backend.MPS, config)
            adapter.save_quantized_model(quantized_model, save_path)
            print(f"\nModel saved to {save_path}")
    
    # Benchmark if requested
    if args.benchmark:
        print("\nBenchmarking quantization methods...")
        results = benchmark_quantization_methods(
            model,
            input_shape=(args.batch_size, 128),
            methods=[
                MPSQuantizationMethod.PYTORCH_DYNAMIC,
                MPSQuantizationMethod.FAKE_QUANT,
                "none",
            ],
        )
        
        print("\nBenchmark Results:")
        print("-" * 60)
        for method, stats in results.items():
            if 'error' not in stats:
                print(f"\n{method}:")
                print(f"  Forward time: {stats['avg_forward_time_ms']:.2f}ms")
                print(f"  Min time: {stats['min_forward_time_ms']:.2f}ms")
                print(f"  Max time: {stats['max_forward_time_ms']:.2f}ms")
                print(f"  Memory: {stats['allocated_memory_mb']:.1f}MB")
                print(f"  Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec")
            else:
                print(f"\n{method}: Error - {stats['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MPS Quantization Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="transformer",
        choices=["transformer", "huggingface"],
        help="Model type to use",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace model name",
    )
    
    # Quantization arguments
    parser.add_argument(
        "--method",
        type=str,
        default="PYTORCH_DYNAMIC",
        choices=["PYTORCH_DYNAMIC", "PYTORCH_STATIC", "HQQ_MPS", "FAKE_QUANT"],
        help="Quantization method",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[4, 8, 16],
        help="Quantization bits",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Automatically select quantization based on model size",
    )
    parser.add_argument(
        "--memory-efficient",
        action="store_true",
        help="Enable memory efficient mode",
    )
    parser.add_argument(
        "--fast-math",
        action="store_true",
        help="Enable fast math optimizations",
    )
    
    # Training arguments
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training demo",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of training samples",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save quantized model",
    )
    
    # Demo arguments
    parser.add_argument(
        "--demo-strategy",
        action="store_true",
        help="Demonstrate dynamic quantization strategy",
    )
    parser.add_argument(
        "--demo-methods",
        action="store_true",
        help="Demonstrate different quantization methods",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all demos",
    )
    
    args = parser.parse_args()
    
    # Enable all demos if --all is specified
    if args.all:
        args.demo_strategy = True
        args.demo_methods = True
        args.train = True
        args.benchmark = True
    
    # Run demo
    main(args)