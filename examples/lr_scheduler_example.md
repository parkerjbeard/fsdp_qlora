# Learning Rate Scheduler Examples

This guide shows how to use the learning rate schedulers in the fsdp_qlora training framework.

## Basic Usage

### Constant Learning Rate (Default)
```bash
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --lr_scheduler constant \
  --lr 1e-5
```

### Linear Schedule with Warmup
```bash
# 10% warmup by default
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --lr_scheduler linear \
  --lr 1e-4

# Explicit warmup steps
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --lr_scheduler linear \
  --lr 1e-4 \
  --warmup_steps 1000

# Warmup as ratio of total steps
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --lr_scheduler linear \
  --lr 1e-4 \
  --warmup_ratio 0.2  # 20% warmup
```

### Cosine Annealing with Warmup
```bash
# Basic cosine schedule (10% warmup by default)
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --lr_scheduler cosine \
  --lr 5e-5

# Cosine with minimum learning rate
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --lr_scheduler cosine \
  --lr 5e-5 \
  --warmup_ratio 0.1 \
  --cosine_min_lr_ratio 0.05  # Final LR = 0.05 * initial LR
```

### Cosine with Hard Restarts
```bash
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --lr_scheduler cosine_with_restarts \
  --lr 5e-5 \
  --warmup_steps 500 \
  --cosine_cycles 3  # Number of restart cycles
```

### Polynomial Decay
```bash
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --lr_scheduler polynomial \
  --lr 1e-4 \
  --warmup_ratio 0.1
```

### Exponential Decay
```bash
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --lr_scheduler exponential \
  --lr 1e-4 \
  --warmup_steps 1000
```

## Scheduler Comparison

| Scheduler | Description | Best Use Case |
|-----------|-------------|---------------|
| `constant` | Fixed learning rate | Debugging, baseline experiments |
| `linear` | Linear decay to 0 | Standard fine-tuning |
| `cosine` | Smooth cosine decay | Longer training runs |
| `cosine_with_restarts` | Cosine with periodic restarts | Very long training, exploration |
| `polynomial` | Polynomial decay curve | Custom decay profiles |
| `exponential` | Exponential decay | Rapid initial decay needed |

## Warmup Configuration

You can configure warmup in three ways (only use one):

1. **Warmup steps**: Exact number of steps
   ```bash
   --warmup_steps 1000
   ```

2. **Warmup ratio**: Percentage of total training steps
   ```bash
   --warmup_ratio 0.1  # 10% of total steps
   ```

3. **Warmup epochs**: Number of epochs (requires consistent batch size)
   ```bash
   --warmup_epochs 2
   ```

## Default Behavior

- For `constant` scheduler: No warmup by default
- For all other schedulers: 10% warmup by default if not specified
- Minimum LR for cosine: 0 by default (can be changed with `--cosine_min_lr_ratio`)

## Example: Full QLoRA Training with Cosine Schedule

```bash
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --train_type qlora \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --lr 2e-4 \
  --lr_scheduler cosine \
  --warmup_ratio 0.03 \
  --cosine_min_lr_ratio 0.1 \
  --dataset alpaca \
  --context_length 2048 \
  --use_gradient_checkpointing true
```

This configuration:
- Uses cosine annealing with 3% warmup
- Decays from 2e-4 to 2e-5 (10% of initial)
- Suitable for longer QLoRA fine-tuning runs