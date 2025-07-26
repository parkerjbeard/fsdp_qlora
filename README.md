# fsdp_qlora

Training LLMs with Quantized LoRA + FSDP.

Read our [announcement blog post](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html).

You should treat this script as an alpha/preview release. If you're not comfortable with testing and debugging models, we'd suggest holding off for a few months while the community more fully tests the approach.

## New: Multi-Backend Support

FSDP+QLoRA now supports multiple compute backends beyond CUDA:
- **Apple Silicon** (MPS/MLX) - Train on M1/M2/M3 Macs
- **CPU** - For testing and development
- **Auto-detection** - Automatically selects the best available backend

### MLX Framework Integration

New MLX support provides optimized training on Apple Silicon:
- **4-bit Quantization**: Memory-efficient training with MLX quantization
- **LoRA Fine-tuning**: Native MLX LoRA implementation
- **Unified Memory**: Leverages Apple Silicon's unified memory architecture
- **PyTorch Compatibility**: Seamless integration with existing code

See [MLX Integration Documentation](docs/mlx_integration.md) for details.

### Advanced MPS Quantization

We now provide comprehensive MPS quantization with multiple backends:
- **MLX Backend**: Native 1-8 bit quantization for Apple Silicon
- **Quanto Backend**: HuggingFace's cross-platform quantization with MPS support
- **Custom MPS Backend**: Fallback implementation for PyTorch compatibility
- **Unified API**: Automatic backend selection based on hardware and requirements

Key features:
- **Mixed Precision**: Different quantization bits per layer
- **QLoRA Support**: Fine-tune quantized models on MPS
- **Memory Optimization**: Automatic memory management for large models
- **Performance Profiling**: Built-in benchmarking tools

See [MPS Quantization Guide](docs/mps_quantization_guide.md) and [Migration Guide](docs/migration_guide.md) for details.

See [Backend Usage Documentation](docs/backend_usage.md) for general backend information.

## Integrations

FSDP+QLoRA has been integrated into:
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1378): experimental support

## Installation

### Quick Start (Automatic Platform Detection)

The easiest way to install FSDP+QLoRA is using our automatic setup script:

```bash
# Clone the repository
git clone https://github.com/AnswerDotAI/fsdp_qlora
cd fsdp_qlora

# Run the platform-aware setup script
python setup.py

# Login to Hugging Face (for model access)
huggingface-cli login
```

The setup script will:
- Automatically detect your platform (CUDA, Apple Silicon, or CPU)
- Install appropriate dependencies
- Verify the installation
- Configure the backend manager

### Platform-Specific Installation

#### NVIDIA GPUs (CUDA)

For systems with NVIDIA GPUs (tested on CUDA 11.7, 11.8, and 12.1):

```bash
# Install CUDA-specific requirements
pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Optional: Install Flash Attention 2 for better performance
pip install flash-attn>=2.5.0

# Optional: HQQ quantization with CUDA kernels
# Follow HQQ installation instructions at https://github.com/mobiusml/hqq
cd hqq/kernels && python setup_cuda.py install
```

#### Apple Silicon (M1/M2/M3 Macs)

For Apple Silicon Macs with MPS/MLX support:

```bash
# Install Mac-specific requirements
pip install -r requirements-mac.txt

# Optional: Install MLX for optimized performance
pip install mlx mlx-lm

# Note: bitsandbytes is not supported on Mac, HQQ is recommended for quantization
```

#### CPU Installation

For CPU-only systems (useful for development/testing):

```bash
# Install base requirements
pip install -r requirements.txt

# Note: Training will be significantly slower on CPU
```

### Manual Installation

If you prefer manual installation or need specific versions:

```bash
# Core dependencies
pip install torch>=2.2.0 transformers>=4.40.0 accelerate>=0.30.0

# Platform-specific quantization
# For CUDA:
pip install bitsandbytes>=0.43.0

# For Mac/CPU:
pip install hqq>=0.1.7

# Additional dependencies
pip install llama-recipes fastcore safetensors tqdm packaging

# Optional: Logging
pip install wandb
```

### Verify Installation

After installation, verify your setup:

```bash
# Check backend detection
python -c "from backend_manager import BackendManager; BackendManager(verbose=True)"

# Run tests
pytest tests/test_backend_manager.py
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.2.0 (recommended for Flash Attention 2 support)
- CUDA 11.7+ (for NVIDIA GPUs)
- macOS 12.3+ (for Apple Silicon)
- 16GB+ RAM recommended

## Finetune Llama-2 70B on Dual 24GB GPUs

Once installed, run `cd fsdp_qlora` and then run the following command to begin finetuning Llama-2 70B on [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned) at a maximum sequence length of 512 tokens.

```bash
python train.py \
--model_name meta-llama/Llama-2-70b-hf \
--batch_size 2 \
--context_length 512 \
--precision bf16 \
--train_type qlora \
--use_gradient_checkpointing true \
--use_cpu_offload true \
--dataset alpaca \
--reentrant_checkpointing true
```

This example command currently uses just over 128GB of CPU RAM. If you only have 128GB available, we recommend making a 10-20GB swap file to accommodate the initial spike in usage.

### Apple Silicon Example

To train on Apple Silicon Macs (M1/M2/M3), use the `--backend` flag:

```bash
python train.py \
--backend mps \
--model_name meta-llama/Llama-2-7b-hf \
--batch_size 2 \
--context_length 512 \
--precision fp16 \
--train_type qlora \
--use_gradient_checkpointing true \
--dataset alpaca_sample
```

Or use `--backend auto` to automatically detect and use the best available backend.

## Training Options

### Quantization Abstraction

FSDP+QLoRA now includes a unified quantization abstraction layer that automatically handles backend-specific differences:

- **Automatic Method Selection**: Chooses the best quantization method based on your hardware
- **Backend Compatibility**: Ensures quantization methods work correctly with your backend
- **Configuration Validation**: Validates settings to prevent incompatible configurations
- **Fallback Support**: Gracefully handles missing dependencies

See [Quantization Abstraction Documentation](docs/quantization_abstraction.md) for detailed information.

### Model Loading Abstraction

The new model loading abstraction simplifies and unifies model loading across different backends:

- **Unified Interface**: Single API for all model loading scenarios
- **Backend Optimizations**: Tailored loading strategies for CUDA, MPS, MLX, and CPU
- **Memory-Efficient Loading**: Support for low-memory and unified memory architectures
- **Parallel Weight Loading**: Efficient parallel loading for large models
- **Quantization Integration**: Seamless integration with quantization methods

See [Model Loading Abstraction Documentation](docs/model_loading_abstraction.md) for detailed information.

For quantization we support HQQ and bitsandbytes. We're currently doing benchmarking to help you decide which to use. If you do use bitsandbytes, be sure to pass `--reentrant_checkpointing True` to avoid triggering a bug in bitsandbytes which results in high memory usage (a fix is in progress).

### `--train_type full`

Full params fine-tuning.

```bash
export CUDA_VISIBLE_DEVICES=4,5 # optionally set devices
python train.py \
--world_size 2 \ # optional, on a single machine will be set automatically
--master_port 12356 \ # optional, defaults to 12355
--model_name meta-llama/Llama-2-7b-hf \
--gradient_accumulation_steps 4 \
--batch_size 8 \
--context_length 512 \
--precision bf16 \
--train_type full \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--use_activation_cpu_offload false \
--log_to wandb \
--dataset alpaca
```

### `--train_type lora`

LoRA fine-tuning using HF PEFT library.

```diff
- --train_type full \
+ --train_type lora \
```

### `--train_type custom_lora`

LoRA fine-tuning using a custom LoRA module.

```diff
- --train_type full \
+ --train_type custom_lora \
```

### `--train_type qlora`

4-bit quantized LoRA fine-tuning using bitsanbytes Linear4bit layer with NF4 quantization and HF PEFT library.

```diff
- --train_type full \
+ --train_type qlora \
+ --reentrant_checkpointing true \
```

### `--train_type custom_qlora`

4-bit quantized LoRA fine-tuning using bitsanbytes Linear4bit layer with NF4 quantization and a custom LoRA module.

```diff
- --train_type full \
+ --train_type custom_qlora \
+ --reentrant_checkpointing true \
```

### `--train_type hqq_lora`

4-bit quantized LoRA fine-tuning using HQQ library and a custom LoRA module.

```diff
- --train_type full \
+ --train_type hqq_lora \
```

### `--train_type bnb_dora`

4-bit quantized DoRA fine-tuning using bitsanbytes Linear4bit layer with NF4 quantization and a custom DoRA module.

```diff
- --train_type full \
+ --train_type bnb_dora \
```

### `--train_type hqq_dora`

4-bit quantized DoRA fine-tuning using HQQ library and a custom DoRA module.

```diff
- --train_type full \
+ --train_type hqq_dora \
```

### `--train_type bnb_llama_pro`

4-bit quantized Llama-Pro fine-tuning using bitsanbytes Linear4bit layer with NF4 quantization.

To create llama-pro weights, run the following command:

```bash
python scripts/block_expansion.py \
--model_name meta-llama/Llama-2-7b-hf \
--output_dir /path/to/llama_pro_weights_directory \
--expansion_rate 0.1
```

```diff
- --train_type full \
+ --train_type bnb_llama_pro \
+ --llama_pro_path /path/to/llama_pro_weights_directory \
```

### `--train_type hqq_llama_pro`

4-bit quantized Llama-Pro fine-tuning using HQQ library.

To create llama-pro weights, run the following command:

```bash
python scripts/block_expansion.py \
--model_name meta-llama/Llama-2-7b-hf \
--output_dir /path/to/llama_pro_weights_directory \
--expansion_rate 0.1
```

```diff
- --train_type full \
+ --train_type hqq_llama_pro \
+ --llama_pro_path /path/to/llama_pro_weights_directory \
```

## Learning Rate Schedulers

FSDP+QLoRA now supports advanced learning rate scheduling with configurable warmup:

### Available Schedulers

- **`constant`**: Fixed learning rate (default, no warmup)
- **`linear`**: Linear decay to 0 after warmup
- **`cosine`**: Cosine annealing with warmup
- **`cosine_with_restarts`**: Cosine with hard restarts
- **`polynomial`**: Polynomial decay (customizable power)
- **`exponential`**: Exponential decay after warmup

### Basic Usage

```bash
# Linear schedule with default 10% warmup
python train.py \
--lr_scheduler linear \
--lr 1e-4

# Cosine schedule with custom warmup
python train.py \
--lr_scheduler cosine \
--lr 5e-5 \
--warmup_steps 1000 \
--cosine_min_lr_ratio 0.1
```

### Warmup Configuration

Configure warmup using one of three methods:

```bash
--warmup_steps 1000        # Exact number of steps
--warmup_ratio 0.1         # Percentage of total steps
--warmup_epochs 2          # Number of epochs
```

By default, all schedulers except `constant` use 10% warmup if not specified.

See [Learning Rate Scheduler Examples](examples/lr_scheduler_example.md) for detailed usage.

## Logging Control

Control the frequency of training metrics logging to reduce output verbosity:

```bash
# Log every 10 steps instead of every step
python train.py \
--log_every_n_steps 10 \
--log_to wandb
```

This only affects training metrics (loss, learning rate). Memory and other system metrics are always logged.

## Low Memory Loading

During quantized LoRA training we use a custom quantization and loading code to avoid loading the entire model into GPU memory before sharding it across GPUs. This is the default behavior of our training script when any of the following training options `"qlora", "custom_qlora", "hqq_lora"` is used. Other training options are already optimized for low memory loading to their best extent.

We load the weights iteratively, quantize them on the GPU and place them back to CPU or meta device (based on their rank) concurrently a few layers at a time. We do this across all GPUs to initialize the quantization parameters, such as zero and scale, while using `sync_module_states=True` to sync the model parameters and buffers across all GPUs during FSDP initialization.

## Mixed Precision Training

### `--precision bf16` (pure bfloat16)

This will cast all the model parameters to `torch.bfloat16` before training and won't use FSDP mixed precision. As a result, sharded and unsharded params will be stored in bf16, forward and backward passes will be done in bf16, and gradient reduction and updates will be done in bf16.

### `--precision fp32` (pure float32)

This will cast all the model parameters to `torch.float32` before training and won't use FSDP mixed precision. As a result, sharded and unsharded params will be stored in fp32, forward and backward passes will be done in fp32, and gradient reduction and updates will be done in fp32.


### `--precision mp_fp16_autocast` (mixed float16 with autocast)

This will cast all the model parameters to `torch.float32` before training and will use FSDP mixed precision with

```
mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
```

As a results, sharded and unsharded params will be stored in fp32. It will use `autocast(torch.float16)` for forward and backward passes, and `autocast(torch.float16)` for gradient reduction and updates.


### `--precision mp_bf16_autocast` (mixed bfloat16 with autocast)

This will cast all the model parameters to `torch.float32` before training and will use FSDP mixed precision with

```
mp_policy = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
```

As a results, sharded and unsharded params will be stored in fp32. It will use `autocast(torch.bfloat16)` for forward and backward passes, and `autocast(torch.bfloat16)` for gradient reduction and updates.


### `--precision mp_bf16_buffers_autocast` (bfloat16 params and float32 buffers with autocast)

This will cast all the model parameters to `torch.bfloat16` before training but will keep the buffers in `torch.float32` and will use FSDP mixed precision with

```
mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32)
```

As a results, sharded and unsharded params will be stored in bf16. It will use `autocast(torch.bfloat16)` for forward and backward passes, and `autocast(torch.bfloat16)` for gradient reduction and updates. Buffers and only [eligible operations](https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16) in autocast will be performed in bf16.

This option is important for RoPE layer which gives incorrect results when cast to lower precision especially with longer context lengths.

## Comparison to an existing trainer

![Screenshot 2024-02-01 083222](https://github.com/AnswerDotAI/fsdp_qlora/assets/6575163/97bb03fb-c2bb-4679-83ff-63a2e202826f)
`hf_train.py` uses TRL's SFTTrainer for a comparison run. To match with our script, modify the dataloading code to train on everything (not just completions) and then run `train.py --train_type qlora --dataset guanaco --batch_size 8 --lr_scheduler cosine --log_to wandb --save_model True --output_dir guanaco_7B --gradient_accumulation_steps 2 --lr 2e-4`. The SFTTrainer version has to run with a lower batch size (4 vs 8) so we only do 2 gradient accumulation steps vs 4 in the QLoRA+FSDP version.

## Converting Saved Models

If you specify `--save_model True` the adapter layers will be saved as a state dict. To convert to the regular Hugging Face format and upload to the hub, see: **Converting the State Dict.ipynb**

If `"custom_qlora", "hqq_lora"` training options are used, then only the trainable LoRA parameters will be saved. Before inference, you need to load and quantize the base model again, and separately load the saved LoRA parameters.

You can alternatively test to see if merging base model weights and trained LoRA weights and then quantizing them performs similar to keeping the parameters separately as done during training. To make use of `torch.compile` with HQQ, see https://github.com/mobiusml/hqq/issues/18.

## Limitations

While QLoRA finetuning works with FSDP, there are some rough edges to be aware of with this alpha release and our example script.

First, the current release of Transformer `AutoModel.from_pretrained` cannot be used to load models into quantized weights, as it does not support the new quant_storage or quantization flag. Loading pretrained models requires writing or using custom model loading code. We provide an example of how to load and quantize a QLoRA model for finetuning in our demo script.

We are actively working with Hugging Face to resolve this incompatibility in future Transformers and PEFT releases.

Second, while FSDP’s Mixed Precision works with QLoRA, practitioners need to be careful to set the `MixedPrecision.param_type` to match the `Linear4Bit.quant_storage` dtype. Otherwise, FSDP’s Mixed Precision could cast the quantized weights to a different precision, essentially turning them into random weights. Our example script shows how to avoid this potential pitfall, and we will be happy to assist model training libraries in correctly exposing FSDP’s Mixed Precision options to users when training with QLoRA

## Example: Llama 70B 4-A100 40GB Training

```bash
# BnB QLoRA
export CUDA_VISIBLE_DEVICES=4,5,6,7
python train.py \
--world_size 4 \
--master_port 12356 \
--model_name meta-llama/Llama-2-70b-hf \
--gradient_accumulation_steps 4 \
--batch_size 2 \
--context_length 512 \
--precision bf16_buffers_autocast \
--train_type custom_qlora \
--use_gradient_checkpointing true \
--reentrant_checkpointing true
--use_cpu_offload false \
--log_to stdout \
--dataset alpaca

# HQQ QLoRA
export CUDA_VISIBLE_DEVICES=4,5,6,7
python train.py \
--world_size 4 \
--master_port 12356 \
--model_name meta-llama/Llama-2-70b-hf \
--gradient_accumulation_steps 4 \
--batch_size 2 \
--context_length 512 \
--precision bf16_buffers_autocast \
--train_type hqq_lora \
--use_gradient_checkpointing true \
--use_cpu_offload false \
--log_to stdout \
--dataset alpaca
```

**Note:** For large batch size or long context training HQQ LoRA is a bit more memory efficient compared to BnB LoRA with re-entrant checkpointing. So if you are running into OOM issues, try using HQQ LoRA.


## SLURM Training

See `fsdp_multi_node.sh` for an example training script using multi-node training with SLURM.

## Add support for a new model

First, import the new model's transformer, attention, and MLP layers from Transformers:

```python
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MISTRAL_ATTENTION_CLASSES, MistralMLP
```

Then in the `get_wrapping_policy` function, add the attention, MLP, and transformer layers to the `self_attn_policy_fn`, `mlp_policy_fn`, and `transformer_wrap_policy` wrapping policy methods:

```python
def get_wrapping_policy(custom_policy:bool=False):

    def self_attn_policy_fn(module):
        return isinstance(module, tuple(*LLAMA_ATTENTION_CLASSES.values(), *MISTRAL_ATTENTION_CLASSES.values()))

    def mlp_policy_fn(module):
        return isinstance(module, (LlamaMLP, MistralMLP))

    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(LlamaDecoderLayer, MistralDecoderLayer),
    )
```

Finally, add gradient checkpointing support by adding the transformer layer to `check_fn`:

```python
if args["use_gradient_checkpointing"]:
    check_fn = lambda submodule: isinstance(submodule, (LlamaDecoderLayer, MistralDecoderLayer))
```
