"""
Read our announcement blog post: https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html.

This script trains a model using FSDP with LoRA & QLoRA. It pulls inspiration from
- llama-recipes (https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/finetuning.py)
- PyTorch FSDP docs (https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- bitsandbytes (https://github.com/TimDettmers/bitsandbytes)

For information on the different arguments, run `python train.py --help`

You should treat this script as an alpha/preview release. If you're not comfortable with testing and debugging
models, we'd suggest holding off for a few months while the community more fully tests the approach.
"""

# Imports

# General
import copy
import functools
import logging
import os
import sys
import time
import types
from contextlib import nullcontext
from glob import glob
from pathlib import Path

# Import abstraction layer for conditional imports
from src.core.imports import get_module, check_import_availability

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Keep warnings and contextlib for other uses


# Conditional imports will be handled after backend detection
# Global variables for conditional imports
bnb = None
Linear4bit = None
Params4bit = None

import safetensors
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from accelerate import init_empty_weights
from accelerate.utils import set_seed

# Model loading
from fastcore.parallel import parallel

# Argument parsing
from fastcore.script import Param, bool_arg, call_parse
from packaging.version import parse
from safetensors.torch import save_file

# Torch + distributed training
from torch import Tensor, nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)

# FSDP
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import CPUOffload, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from src.utils.lr_scheduler import get_scheduler_with_config
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, hub

try:
    from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear
except ImportError:
    HQQLinear = None
    pass

# To add a new model, import the transformer, attention, & MLP layers
# for the wrapping policy and `check_fn` in activation checkpointing
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaAttention,
)
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralMLP,
    MistralAttention,
)

# Create attention class dictionaries for compatibility
LLAMA_ATTENTION_CLASSES = {"LlamaAttention": LlamaAttention}
MISTRAL_ATTENTION_CLASSES = {"MistralAttention": MistralAttention}

# To get rid of tokenizers warnings for now
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For logging things during training
try:
    import wandb  # noqa: F401
except ImportError:
    pass

# LoRA and DORA modules
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.append(scripts_dir)
# These will be imported conditionally later
BNBDORA = None
HQQDORA = None
DORALayer = None
MagnitudeLayer = None
LORA = None

from src.utils.profiling_utils import profiling_context

# Backend management
from src.core.backend_manager import BackendManager, Backend


def load_conditional_imports(backend_str):
    """Load conditional imports based on the detected backend.
    
    Args:
        backend_str: String representation of the backend (cuda, mps, mlx, cpu)
    """
    global bnb, Linear4bit, Params4bit, BNBDORA, HQQDORA, DORALayer, MagnitudeLayer, LORA
    
    # Import bitsandbytes (only for CUDA)
    if backend_str == 'cuda' or check_import_availability('bitsandbytes', backend_str):
        bnb_module = get_module('bitsandbytes', backend_str)
        bnb = bnb_module
        Linear4bit = bnb_module.Linear4bit
        Params4bit = bnb_module.Params4bit
    else:
        # Use dummy fallback for non-CUDA backends
        bnb_module = get_module('bitsandbytes', backend_str)
        bnb = bnb_module
        Linear4bit = bnb_module.Linear4bit
        Params4bit = bnb_module.Params4bit
    
    # Import DORA modules
    try:
        dora_module = get_module('dora', backend_str)
        BNBDORA = dora_module.BNBDORA
        HQQDORA = dora_module.HQQDORA
        DORALayer = dora_module.DORALayer
        MagnitudeLayer = dora_module.MagnitudeLayer
    except ImportError as e:
        print(f"Warning: DORA modules not available: {e}")
    
    # Import LoRA module
    try:
        lora_module = get_module('lora', backend_str)
        LORA = lora_module.LORA
    except ImportError as e:
        print(f"Warning: LoRA module not available: {e}")


class Logger:
    def __init__(
        self,
        args,
        log_to="stdout",
        project_name="fsdp_qlora",
        entity=None,
        group=None,
        name=None,
        rank=0,
        log_every_n_steps=1,
    ):
        self.log_every_n_steps = log_every_n_steps
        self.log_to = log_to
        self.step_count = 0
        if self.log_to == "wandb" and rank == 0:
            wandb = get_module('wandb')
            wandb.init(
                project=project_name, entity=entity, group=group, name=name, config=args
            )

    def log(self, d: dict, rank: int, force=False):
        if rank != 0:
            return
        
        # Check if this is a loss/lr log (training step log)
        is_step_log = "loss" in d or "lr" in d
        
        if is_step_log:
            self.step_count += 1
            # Only log every n steps unless forced
            if not force and self.step_count % self.log_every_n_steps != 0:
                return
        
        if self.log_to == "tqdm":
            for k, v in d.items():
                tqdm.write(f"{k}: {v}")
        elif self.log_to == "wandb":
            wandb = get_module('wandb')
            wandb.log(d)
        elif self.log_to == "stdout":
            for k, v in d.items():
                print(f"{k}: {v}")

    def finish(self, rank=0):
        if self.log_to == "wandb" and rank == 0:
            wandb = get_module('wandb')
            wandb.finish()


def update_progress_bar(
    progress_bar: tqdm, epoch: int, log_loss: float, log_lr: float, rank: int
):
    """Updates the progress bar with the current epoch, loss, and learning rate"""
    if rank == 0:
        if log_lr >= 0:
            progress_bar.set_description(
                f"Epoch {epoch}, Loss {log_loss:.3f}, LR {log_lr:.2e}", refresh=True
            )
        else:
            progress_bar.set_description(
                f"Epoch {epoch}, Loss {log_loss:.3f}", refresh=True
            )


def n_loading_workers(
    quant_method: str, param_count: float, backend_manager: BackendManager
):
    if backend_manager.backend == Backend.CUDA and torch.cuda.is_available():
        devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
        left = int(os.cpu_count() / torch.cuda.device_count())
        right = int(
            (4 if quant_method == "hqq" else 8)
            * (devprops.total_memory / 1e9 / 40)
            * (70 / (param_count / 1e9))
        )
        return min(left, right)
    else:
        # For non-CUDA backends, use a simpler heuristic
        return min(4, os.cpu_count())


def get_memory_stats(backend_manager: BackendManager, device_idx: int = 0):
    """Get memory statistics for the current backend."""
    if backend_manager.backend == Backend.CUDA:
        return {
            "allocated": torch.cuda.memory_allocated(device_idx),
            "reserved": torch.cuda.memory_reserved(device_idx),
        }
    else:
        # For non-CUDA backends, use backend manager's memory info
        mem_info = backend_manager.get_memory_info()
        return {
            "allocated": mem_info.get("used_memory", 0),
            "reserved": mem_info.get("total_memory", 0),
        }


# Utilities related to model loading
def replace_linear(
    model: nn.Module,
    linear_replacement: nn.Module,
    quant_config: dict | None = None,
    skip_modules: list[str] = ["lm_head"],
    **kwargs,
):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
    """
    for name, module in model.named_children():
        if name in skip_modules:
            continue

        if len(list(module.children())) > 0:
            replace_linear(
                module, linear_replacement, quant_config, skip_modules, **kwargs
            )

        if isinstance(module, torch.nn.Linear):
            if issubclass(linear_replacement, Linear4bit):
                model._modules[name] = linear_replacement(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    **kwargs,
                )
            elif issubclass(linear_replacement, HQQLinear):
                model._modules[name] = linear_replacement(
                    module, quant_config, **kwargs
                )
            else:
                raise ValueError(
                    f"Unsupported linear replacement: {type(linear_replacement)}"
                )
    return model


def setup_quantized_meta_for_peft(model: nn.Module):
    """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""

    def temp_to_method(self, *args, **kwargs):
        return self

    for param in model.parameters():
        if isinstance(param, Params4bit):
            param.quant_state._orig_to = param.quant_state.to
            param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)


def setup_quantized_peft_meta_for_training(model: nn.Module):
    """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, "_orig_to"):
            param.quant_state.to = param.quant_state._orig_to
            param.quant_state._orig_to = None


def load_and_quantize(
    module: nn.Module,
    name: str,
    value: Tensor,
    device: torch.device = None,
    dtype: torch.dtype = None,
    skip_names: list[str] = [],
    to_cpu: bool = False,
    to_meta: bool = False,
    verbose: bool = False,
    quant_method: str = "bnb",
    is_dora: bool = False,
):
    """
    Loads `value` tensor into submodule of `module`, optionally skipping `skip_names` and converting to `dtype`.

    Quantizes `Params4bit` on `device` then places on "cpu" if to_cpu=True or "meta" if to_meta=True.
    """

    def place_on_device(value):
        if to_meta:
            device = "meta"
        elif to_cpu:
            device = "cpu"
        return value.to(device=device, dtype=dtype)

    if any([skip_name in name for skip_name in skip_names]):
        if verbose:
            print(f"Skipping {name} because it is in skip_names")
        return

    module_key, _, value_key = name.rpartition(".")
    try:
        submodule = module.get_submodule(module_key)
    except AttributeError as e:
        print(f"Module {module_key} not found:\n{e}")
        return

    try:
        if quant_method == "bnb":
            param = submodule.get_parameter(value_key)
            if isinstance(param, Params4bit):
                # With `sync_module_states=True`, a meta device Params4bit needs to be the same
                # shape as the quantized Params4bit with an initialized quant_state. However,
                # FSDP only syncs parameters and buffers, so the quant_state isn't copied. This
                # workaround quantizes Params4bit to initialize quant_state on all ranks, then
                # replaces Params4bit's data with a meta tensor to free memory on non-rank 0.
                if is_dora:
                    setattr(
                        submodule,
                        "dora_scale",
                        value.norm(p=2, dim=1).to(dtype=dtype).to("cpu"),
                    )
                value = type(param)(
                    value.to(device=device, dtype=dtype).data, **param.__dict__
                ).cuda(device)
                if to_meta:
                    value = type(param)(value.data.to("meta"), **value.__dict__)
                elif to_cpu:
                    value = type(param)(value.data.to("cpu"), **value.__dict__)
            else:
                value = type(param)(place_on_device(value).data)
        elif quant_method == "hqq":
            if isinstance(submodule, HQQLinear):
                if value_key == "weight":
                    # Like `Params4bit`, this workaround quantizes `HQQLinear`` per device so the quantization
                    # meta dictionary is created on all ranks, before converting to meta on non-rank 0.
                    submodule.linear_layer.to_empty(device=device)
                    submodule.linear_layer.weight.data.copy_(
                        value.to(device=device, dtype=dtype)
                    )
                    if is_dora:
                        setattr(
                            submodule,
                            "dora_scale",
                            value.norm(p=2, dim=1).to(dtype=dtype).to("cpu"),
                        )
                    submodule.initialize()

                    if to_meta:
                        setattr(
                            submodule, "W_q", nn.Parameter(submodule.W_q.to("meta"))
                        )
                    elif to_cpu:
                        setattr(submodule, "W_q", nn.Parameter(submodule.W_q.to("cpu")))
                    submodule.in_gpu = False

                if value_key == "bias":
                    raise ValueError("Bias not supported in HQQLinear yet!")
            else:
                param = submodule.get_parameter(value_key)
                value = type(param)(place_on_device(value).data)

    except AttributeError:
        # it's a buffer
        value = place_on_device(value)
        pass
    if HQQLinear is None or not isinstance(submodule, HQQLinear):
        setattr(submodule, value_key, value)


# DATASET + DATALOADERS (modified from llama recipes)
# Formatting prompts in alpaca
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


# Dataset class
class InstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, style="alpaca"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.style = style

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        if self.style == "guanaco":
            prompt = self.dataset[index]["text"].split("### Assistant: ")[0]
            example = self.dataset[index]["text"]
        elif self.style == "qna":
            prompt_template = (
                "###Context:\n{context}\n###Question:\n{question}\n###Answer:\n"
            )
            sample = self.dataset[index]
            prompt = prompt_template.format_map(sample)
            example = prompt + sample["answer"]
        elif self.style == "qna_no_ctx":
            prompt_template = "###Question:\n{question}\n###Answer:\n"
            sample = self.dataset[index]
            prompt = prompt_template.format_map(sample)
            example = prompt + sample["answer"]
        else:  # Alpaca
            ann = self.dataset[index]
            if ann.get("input", "") == "":
                prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
            else:
                prompt = PROMPT_DICT["prompt_input"].format_map(ann)
            example = prompt + ann["output"]

        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": example_mask.tolist(),
        }


# And to get the dataloader
def get_dataloader(tokenizer: PreTrainedTokenizerFast, args: dict):
    """Creates a dataset and appropriate dataloader with distributed sampler."""
    # Importing here rather than at the start to avoid multiprocessing issues
    from datasets import Dataset, load_dataset

    # Load the source dataset
    if args["dataset"] == "alpaca":
        dataset = load_dataset("yahma/alpaca-cleaned")["train"]
    elif args["dataset"] == "alpaca_sample":
        dataset = load_dataset(
            "yahma/alpaca-cleaned", split=f"train[:{args['dataset_samples']}]"
        )
    elif args["dataset"] == "dummy":
        dataset = Dataset.from_dict(
            {
                "instruction": ["instruction"] * args["dataset_samples"],
                "input": ["input"] * args["dataset_samples"],
                "output": ["output" * args["context_length"] * 2]
                * args["dataset_samples"],
            }  # A long output to test memory usage (gets truncated)
        )
    elif args["dataset"] == "guanaco":
        dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    elif args["dataset"] == "sql":
        dataset = load_dataset("knowrohit07/know_sql")["validation"]
        dataset = dataset.shuffle(seed=args["seed"])
        dataset = dataset.select(range(1000, len(dataset)))
    elif args["dataset"] == "orca_math":
        dataset = load_dataset("microsoft/orca-math-word-problems-200k")[
            "train"
        ].shuffle(seed=42)
        # train with 10k for starters. Then 100k.
        dataset = dataset.select(range(0, args["dataset_samples"]))

    # truncate dataset so it's evenly divisible by grad_accumulation_steps
    dataset = dataset.select(
        range(
            0,
            len(dataset)
            - len(dataset) % (args["batch_size"] * args["gradient_accumulation_steps"]),
        )
    )

    # # Create the InstructionDataset
    if args["dataset"] == "guanaco":
        dataset = InstructionDataset(dataset, tokenizer, style="guanaco")
    elif args["dataset"] == "sql":
        dataset = InstructionDataset(dataset, tokenizer, style="qna")
    elif args["dataset"] == "orca_math":
        dataset = InstructionDataset(dataset, tokenizer, style="qna_no_ctx")
    else:  # (w/ alpaca prompt formatting)
        dataset = InstructionDataset(dataset, tokenizer, style="alpaca")

    # Collate function
    def collate_fn(batch, with_attention_mask=False):
        # To list of tensors
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_masks = [torch.tensor(item["attention_mask"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]
        # Pad + truncate
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )[:, : args["context_length"]]
        if with_attention_mask:
            attention_masks = pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )[:, : args["context_length"]]
        else:
            attention_masks = None
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)[
            :, : args["context_length"]
        ]
        # Return dict
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

    # For distributed training, use DistributedSampler
    sampler = DistributedSampler(dataset, seed=args["seed"])

    # Use the custom collate function in DataLoader
    dataloader = DataLoader(
        dataset, batch_size=args["batch_size"], collate_fn=collate_fn, sampler=sampler
    )

    return dataloader


# LR scheduler.


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    gradient_accumulation_steps: int,
    args: dict,
):
    """Returns configured learning rate scheduler with warmup support"""
    num_training_steps = (
        args["num_epochs"] * len(dataloader) // gradient_accumulation_steps
    )
    
    # Determine warmup configuration
    # If neither warmup_steps nor warmup_ratio is set, default to 10% warmup for non-constant schedulers
    if args.get("warmup_steps") is None and args.get("warmup_ratio") is None:
        if args["lr_scheduler"] != "constant":
            warmup_ratio = 0.1  # Default 10% warmup for backward compatibility
        else:
            warmup_ratio = None
    else:
        warmup_ratio = args.get("warmup_ratio")
    
    # Prepare scheduler-specific kwargs
    scheduler_kwargs = {}
    if args["lr_scheduler"] == "cosine" or args["lr_scheduler"] == "cosine_with_restarts":
        scheduler_kwargs["min_lr_ratio"] = args.get("cosine_min_lr_ratio", 0.1)
        if args["lr_scheduler"] == "cosine_with_restarts":
            scheduler_kwargs["num_cycles"] = args.get("cosine_cycles", 3)
    
    # Get the scheduler
    lr_scheduler = get_scheduler_with_config(
        optimizer=optimizer,
        scheduler_type=args["lr_scheduler"],
        num_training_steps=num_training_steps,
        warmup_steps=args.get("warmup_steps"),
        warmup_ratio=warmup_ratio,
        steps_per_epoch=len(dataloader) // gradient_accumulation_steps,
        **scheduler_kwargs
    )
    
    return lr_scheduler, num_training_steps


# Optimizer
def get_optimizer(model: nn.Module, args: dict):
    """Returns an optimizer. We can add more options here if needed."""
    if args["optimizer"] in ["adam", "fused_adam"]:
        return optim.Adam(
            model.parameters(), lr=args["lr"], fused=args["optimizer"] == "fused_adam"
        )
    elif args["optimizer"] == "sgd":
        return optim.SGD(model.parameters(), lr=args["lr"])
    elif args["optimizer"] == "adadelta":
        return optim.Adadelta(model.parameters(), lr=args["lr"])
    elif args["optimizer"] in ["adamw", "fused_adamw"]:
        return torch.optim.AdamW(
            model.parameters(),
            lr=args["lr"],
            betas=(0.9, 0.95),
            eps=1e-5,
            weight_decay=args["wd"],
            fused=args["optimizer"] == "fused_adamw",
        )
    else:
        raise ValueError("Invalid optimizer")


# Wrap the model using LoRA policy from llama-recipes or custom policy:
# This checks for lora layers (has weight and requires_grad)
def get_wrapping_policy(custom_policy: bool = False, vanilla_policy: bool = False):

    if custom_policy:

        def lambda_policy_fn(module):
            # LoRA and DoRA trainable layers.
            return (
                isinstance(module, nn.Sequential)
                and all(m.weight.requires_grad for m in module)
            ) or (isinstance(module, (DORALayer, MagnitudeLayer)))

    else:

        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

    def self_attn_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(
            module,
            tuple(
                (*LLAMA_ATTENTION_CLASSES.values(), *MISTRAL_ATTENTION_CLASSES.values())
            ),
        )

    def mlp_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, (LlamaMLP, MistralMLP))

    lambda_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
    )
    self_attn_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn
    )
    mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(LlamaDecoderLayer, MistralDecoderLayer),
    )
    if vanilla_policy:
        return transformer_wrap_policy

    policies = [lambda_policy, transformer_wrap_policy]
    if custom_policy:
        policies.extend([self_attn_policy, mlp_policy])
    return functools.partial(_or_policy, policies=policies)


# Main function, run on each process
def fsdp_main(local_rank: int, world_size: int, args: dict):

    # Get backend manager from args
    backend_manager = args["backend_manager"]

    # Setup and initialize the process group
    os.environ["MASTER_ADDR"] = args["master_addr"]
    os.environ["MASTER_PORT"] = args["master_port"]

    # Determine rank based on backend
    if backend_manager.backend == Backend.CUDA:
        if "SLURM_PROCID" in os.environ:
            # assumes same number of GPUs per node.
            rank = (
                int(os.environ["SLURM_PROCID"]) * torch.cuda.device_count() + local_rank
            )
        else:
            rank = local_rank
    else:
        # Non-CUDA backends use single process
        rank = 0

    # Initialize distributed process group if supported
    if backend_manager.capabilities.supports_distributed and world_size > 1:
        dist_backend = backend_manager.get_distributed_backend()
        dist.init_process_group(dist_backend, rank=rank, world_size=world_size)

    # Set device based on backend
    if backend_manager.backend == Backend.CUDA:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif backend_manager.backend == Backend.MPS:
        device = torch.device("mps")
    elif backend_manager.backend == Backend.CPU:
        device = torch.device("cpu")
    elif backend_manager.backend == Backend.MLX:
        # MLX uses CPU device for PyTorch compatibility
        device = torch.device("cpu")

    # Set thread count for CPU operations
    if args["use_cpu_offload"]:
        if backend_manager.backend == Backend.CUDA:
            torch.set_num_threads(
                os.cpu_count() // (min(world_size, torch.cuda.device_count()))
            )
        else:
            torch.set_num_threads(os.cpu_count() // world_size)

    # Store device in args for later use
    args["device"] = device

    # Start logging
    logger = Logger(
        args,
        log_to=args["log_to"],
        project_name=args["project_name"],
        entity=args["entity"],
        group=args["group"],
        name=args["name"],
        rank=rank,
        log_every_n_steps=args.get("log_every_n_steps", 1),
    )

    # Timing stuff (only for CUDA backend)
    if backend_manager.backend == Backend.CUDA:
        init_start_event = torch.cuda.Event(enable_timing=True)
        init_end_event = torch.cuda.Event(enable_timing=True)
    else:
        init_start_event = None
        init_end_event = None

    # model precision, qlora compute precison, and FSDP mixed precision policy.
    # The Linear4Bit quant_storage dtype should always match the FSDP param_dtype. The compute_dtype should match the AMP compute dtype.
    # MixedPrecision(param_dtype=fp32, reduce_dtype=fp32, buffer_dtype=fp32) uses `torch.amp.autocast` to control precision.
    # Limited qlora testing shows that fp16 only works with autocast while bf16 trains with both pure and autocast modes.
    # Note: FP16 mixed precision requires autocast for numerical stability, while BF16 has better numerical range
    # and can work without autocast. This behavior is consistent with PyTorch's recommendations.
    mp_policy = None
    load_param_skip_names = []
    if args["precision"] == "bf16":
        torch_dtype, compute_dtype = torch.bfloat16, torch.bfloat16
    elif args["precision"] == "fp32":
        torch_dtype, compute_dtype = torch.float32, torch.float16
    elif args["precision"] == "fp16_autocast":
        compute_dtype, torch_dtype = torch.float16, torch.float32
        mp_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
    elif args["precision"] == "bf16_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.float32
        mp_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
    elif args["precision"] == "bf16_buffers_autocast":
        compute_dtype, torch_dtype = torch.bfloat16, torch.bfloat16
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,
        )
        load_param_skip_names = ["inv_freq"]
    else:
        raise ValueError("Invalid precision")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    # Set pad token to eos token if pad token doesn't exist
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set up dataloader
    dataloader = get_dataloader(tokenizer, args)

    # Create model
    cfg = None
    attn_impl = "sdpa"  # torch 2.2 sdpa uses flash attn 2
    if rank == 0 or args["verbose"]:
        print("Creating model", rank)
    if args["train_type"] in ["full", "lora", "custom_lora"]:
        if (args["low_memory"] and rank == 0) or (not args["low_memory"]):
            model = AutoModelForCausalLM.from_pretrained(
                args["model_name"],
                use_cache=False,
                torch_dtype=torch_dtype,
                _attn_implementation=attn_impl,
            )
            dtype = torch_dtype if args["precision"] == "bf16" else None
            model.to(dtype=dtype, device="cpu" if args["low_memory"] else device)
        else:
            cfg = AutoConfig.from_pretrained(args["model_name"])
            cfg.use_cache = False
            cfg._attn_implementation = attn_impl
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch_dtype)
            if args["precision"] == "bf16":
                model.to(torch_dtype)
    elif args["train_type"] in [
        "qlora",
        "custom_qlora",
        "hqq_lora",
        "hqq_dora",
        "bnb_dora",
        "bnb_llama_pro",
        "hqq_llama_pro",
    ]:  # Our custom loading
        cfg = AutoConfig.from_pretrained(args["model_name"])
        cfg.use_cache = False
        cfg._attn_implementation = attn_impl
        skip_modules = ["lm_head"]

        if args["train_type"] in ["bnb_llama_pro", "hqq_llama_pro"]:
            llama_pro_path = Path(args["llama_pro_path"])
            num_original_layers, num_expanded_layers = llama_pro_path.name.split(
                "blk_exp-"
            )[1].split("-")
            num_original_layers, num_expanded_layers = int(num_original_layers), int(
                num_expanded_layers
            )
            total_new_layers = num_expanded_layers - num_original_layers
            split = int(
                num_original_layers / (num_expanded_layers - num_original_layers)
            )
            new_layer_ids = [split + (split + 1) * n for n in range(total_new_layers)]
            new_layer_names = [f"layers.{i}" for i in new_layer_ids]
            skip_modules += [str(lid) for lid in new_layer_ids]
            cfg.num_hidden_layers = num_expanded_layers

        # load model on meta device without calling init and replace nn.Linear with Linear4bit
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
            if args["train_type"] in ["hqq_lora", "hqq_dora", "hqq_llama_pro"]:
                # HQQ quantization configuration
                # Default settings are optimized for 4-bit quantization with good quality/performance trade-off
                # group_size=64 is a good balance between quantization quality and memory usage
                # quant_zero=True and quant_scale=True enable full quantization of weights
                quant_config = BaseQuantizeConfig(
                    nbits=int(args["n_bits"]),
                    group_size=args.get("hqq_group_size", 64),
                    quant_zero=True,
                    quant_scale=True,
                    offload_meta=True,
                    view_as_float=True,
                )
                model.model = replace_linear(
                    model.model,
                    HQQLinear,
                    quant_config,
                    device=device,
                    compute_dtype=compute_dtype,
                    del_orig=True,
                    initialize=False,
                    skip_modules=skip_modules,
                )
                HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
            else:
                model.model = replace_linear(
                    model.model,
                    Linear4bit,
                    compute_dtype=compute_dtype,
                    quant_type="nf4",
                    quant_storage=torch_dtype,
                    skip_modules=skip_modules,
                )
        model.is_loaded_in_4bit = True

        # Grab the safetensors files that hold the weights
        if args["train_type"] in ["bnb_llama_pro", "hqq_llama_pro"]:
            files = glob(str(llama_pro_path / "*.safetensors"))
        else:
            try:
                idx = hub.cached_file(args["model_name"], SAFE_WEIGHTS_INDEX_NAME)
                files, _ = hub.get_checkpoint_shard_files(args["model_name"], idx)
            except OSError:
                try:
                    # This means the model doesn't have a model.safetensors.index.json because it is not sharded
                    files = []
                    files.append(hub.cached_file(args["model_name"], SAFE_WEIGHTS_NAME))
                except OSError as e:
                    # This means the model probably doesn't have a safetensors file
                    raise e

        # Load in the weights, using our custom load_and_quantize method which quantizes Params4bit on the fly
        # and then places each layer on CPU or meta if using low_memory to minimize GPU memory usage
        def load_and_quantize_parallel(name_param, model, **kwargs):
            name, param = name_param
            load_and_quantize(model, name, param, **kwargs)

        quant_method = (
            "hqq"
            if args["train_type"] in ["hqq_lora", "hqq_dora", "hqq_llama_pro"]
            else "bnb"
        )
        param_count = sum((p.numel() for n, p in model.named_parameters()))
        if rank == 0 or args["verbose"]:
            print("Loading model", rank)
        if rank == 0 and args["verbose"]:
            print(f"Total model params: {param_count}")

        n_workers = (
            n_loading_workers(quant_method, param_count, backend_manager)
            if args["loading_workers"] == -1
            else args["loading_workers"]
        )
        if rank == 0 and args["verbose"]:
            print(f"Using n_workers: {n_workers} for loading")

        start = time.time()
        for filename in tqdm(
            files,
            desc="Loading & Quantizing Model Shards",
            disable=rank != 0,
            position=0,
        ):
            weights = safetensors.torch.load_file(filename)
            parallel(
                load_and_quantize_parallel,
                iter(weights.items()),
                n_workers=n_workers,
                threadpool=True,
                model=model,
                dtype=torch_dtype,
                device=device,
                skip_names=load_param_skip_names,
                to_cpu=(args["low_memory"] and rank == 0),
                to_meta=(args["low_memory"] and rank != 0),
                verbose=args["verbose"],
                quant_method=quant_method,
                is_dora=(args["train_type"] in ["hqq_dora", "bnb_dora"]),
            )

        if rank == 0 and args["verbose"]:
            print(f"Loaded model weights in {time.time()-start:.3f} seconds")
        # cleanup any extra memory usage from parallel loading
        if backend_manager.backend == Backend.CUDA:
            torch.cuda.empty_cache()
    if rank == 0 or args["verbose"]:
        mem_stats = get_memory_stats(backend_manager, local_rank)
        print(f"Rank {rank}: Model created: {mem_stats['reserved']/2**30:.3f} GiB")

    # PEFT setup (LoRA and QLoRA)
    if args["train_type"] in ["lora", "qlora"]:
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args["lora_rank"],
            lora_alpha=args["lora_alpha"],
            lora_dropout=args["lora_dropout"],
            target_modules=args["lora_target_modules"],
        )
        # PEFT will move quant_state to meta device, so this method prevents that
        # from happening by replacing quant_state.to with a dummy function
        if rank != 0 and args["low_memory"]:
            setup_quantized_meta_for_peft(model)

        model = get_peft_model(model, peft_config)

        if rank == 0:
            model.print_trainable_parameters()
        elif args["low_memory"]:
            # And then setup_quantized_peft_meta_for_training sets quant_state.to back to normal
            setup_quantized_peft_meta_for_training(model)
    elif args["train_type"] in [
        "custom_qlora",
        "custom_lora",
        "hqq_lora",
        "hqq_dora",
        "bnb_dora",
    ]:
        if args["train_type"] == "hqq_dora":
            print("Using HQQDORA", rank)
            lora_cls = HQQDORA
        elif args["train_type"] == "bnb_dora":
            print("Using BNB DORA", rank)
            lora_cls = BNBDORA
        else:
            print("Using LORA", rank)
            lora_cls = LORA
        # Create LORA layers.
        for name, _ in model.named_modules():
            module_key, _, value_key = name.rpartition(".")
            if value_key in args["lora_target_modules"]:
                m = model.get_submodule(name)
                qlora_layer = lora_cls(
                    m, args["lora_rank"], args["lora_alpha"], args["lora_dropout"]
                )
                parent_module = model.get_submodule(module_key)
                setattr(parent_module, value_key, qlora_layer)
        for n, p in model.named_parameters():
            if any(
                [
                    lora_name in n
                    for lora_name in ["lora_AB", "lora_A", "lora_B", "magnitude"]
                ]
            ):
                p.requires_grad = True
                if args["verbose"]:
                    print("Trainable LORA layer", n)
            else:
                p.requires_grad = False
        if rank == 0 or args["verbose"]:
            mem_stats = get_memory_stats(backend_manager, local_rank)
            print(
                f"Rank {rank}: LoRA layers added: {mem_stats['reserved']/2**30:.3f} GiB"
            )

    elif args["train_type"] in ["bnb_llama_pro", "hqq_llama_pro"]:
        for n, p in model.named_parameters():
            if any([layer_name in n for layer_name in new_layer_names]):
                p.requires_grad = True
                if args["verbose"]:
                    print("Trainable Llama-Pro layer", n)
            else:
                p.requires_grad = False

    if args["log_to"] == "wandb":
        mem_stats = get_memory_stats(backend_manager, local_rank)
        logger.log(
            {"memory/allocated_after_model_created": mem_stats["allocated"]}, rank
        )
        logger.log(
            {"memory/reserved_after_model_creation": mem_stats["reserved"]}, rank
        )

    # Wrap model with llama-recipies or custom LoRA policy
    my_auto_wrap_policy = get_wrapping_policy(
        custom_policy=args["train_type"]
        in ["custom_qlora", "hqq_lora", "hqq_dora", "bnb_dora"],
        vanilla_policy=args["train_type"] in ["full", "bnb_llama_pro", "hqq_llama_pro"],
    )

    if rank == 0 or args["verbose"]:
        print("Wrapping model w/ FSDP", rank)

    if args["sharding_strategy"] == "full_shard":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif args["sharding_strategy"] == "shard_grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args["sharding_strategy"] == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    elif args["sharding_strategy"] == "hybrid_full_shard":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif args["sharding_strategy"] == "hybrid_shard_grad_op":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError("Invalid FSDP sharding strategy")

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=my_auto_wrap_policy,
        # backward_prefetch=None, #BackwardPrefetch.BACKWARD_PRE
        use_orig_params=False,
        cpu_offload=(
            CPUOffload(offload_params=True) if args["use_cpu_offload"] else None
        ),
        limit_all_gathers=True,  # See https://github.com/pytorch/pytorch/issues/91165
        device_id=(
            torch.cuda.current_device()
            if backend_manager.backend == Backend.CUDA
            else None
        ),
        sync_module_states=args["low_memory"],
        param_init_fn=lambda module: (
            module.to_empty(device=device, recurse=False)
            if (rank != 0 and args["low_memory"])
            else None
        ),  # When low_memory is True, rank 0 loads the full model while other ranks keep the model on meta device.
        # This param_init_fn moves parameters from meta device to the actual device (empty tensors) for non-rank-0 processes.
        # FSDP will then sync parameters from rank 0 to all other ranks during initialization.
        # This approach minimizes memory usage during model initialization for multi-GPU training.
        mixed_precision=mp_policy,
    )
    if rank == 0 or args["verbose"]:
        mem_stats = get_memory_stats(backend_manager, local_rank)
        print(f"Rank {rank}: Wrapped model: {mem_stats['reserved']/2**30:.3f} GiB")
    if args["log_to"] == "wandb":
        mem_stats = get_memory_stats(backend_manager, local_rank)
        logger.log({"memory/allocated_after_model_wrap": mem_stats["allocated"]}, rank)
        logger.log({"memory/reserved_after_model_wrap": mem_stats["reserved"]}, rank)

    # Synchronize at the start
    dist.barrier()

    # Apply activation checkpointing
    if args["use_gradient_checkpointing"]:
        if args["reentrant_checkpointing"]:
            model.enable_input_require_grads()
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=(
                CheckpointImpl.REENTRANT
                if args["reentrant_checkpointing"]
                else CheckpointImpl.NO_REENTRANT
            ),
        )

        def check_fn(submodule):
            return isinstance(submodule, (LlamaDecoderLayer, MistralDecoderLayer))

        if rank == 0 or args["verbose"]:
            print("Applying activation checkpointing", rank)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    if args["use_activation_cpu_offload"]:
        if rank == 0 or args["verbose"]:
            print("Applying activation offloading", rank)
        model = offload_wrapper(model)

    if rank == 0 and args["verbose"]:
        print("Config:")
        print(cfg)
        print("Model:")
        print(model)
        print("Starting training")

    # Create the optimizer
    optimizer = get_optimizer(model, args)

    # LR scheduler.
    gradient_accumulation_steps = max(1, args["gradient_accumulation_steps"])
    lr_scheduler, num_training_steps = get_lr_scheduler(
        optimizer, dataloader, gradient_accumulation_steps, args
    )

    # Sanity check: see what parameters the optimizer has and which require grad:
    if rank == 0 and args["verbose"]:
        print("Optimizer params:")
        for group in optimizer.param_groups:
            for param in group["params"]:
                print(
                    f"Shape: {param.shape}, Requires Grad: {param.requires_grad}, Dtype: {param.dtype}"
                )

    # Autocast for mixed precision with fp16/bf16 compute types with fp32 params
    if args["precision"] in ["fp16_autocast", "bf16_autocast", "bf16_buffers_autocast"]:
        if backend_manager.backend == Backend.CUDA:
            autocast = torch.cuda.amp.autocast(enabled=True, dtype=compute_dtype)
        else:
            # Use general autocast for non-CUDA backends
            autocast = torch.amp.autocast(
                device_type=str(device.type), enabled=True, dtype=compute_dtype
            )
    else:
        autocast = nullcontext()
    scaler = ShardedGradScaler() if args["precision"] == "fp16_autocast" else None
    scale_grads = scaler is not None

    if rank == 0:
        print("Total Training Steps:", num_training_steps)
    memory_stats = []
    progress_bar = tqdm(range(num_training_steps), disable=rank != 0)
    if init_start_event is not None:
        init_start_event.record()
    log_loss, log_lr = 0.0, -1
    # Reset peak memory to track that
    if backend_manager.backend == Backend.CUDA:
        torch.cuda.reset_peak_memory_stats(local_rank)
    with profiling_context(args, rank=rank) as prof:
        for epoch in range(args["num_epochs"]):
            update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
            model.train()
            ddp_loss = torch.zeros(2).to(device)

            for batch_idx, batch in enumerate(dataloader):

                accumulate_grads = (batch_idx + 1) % gradient_accumulation_steps == 0

                # Prevent gradient syncing until update step if using no_sync option.
                # Documentation states this should only be used on the root FSDP instance
                # We assume this is a one-node setup
                if args["no_sync"] and not accumulate_grads:
                    sync_context = model.no_sync()
                else:
                    sync_context = nullcontext()

                # Start logging memory (first iter) if requested
                if (
                    args["profile_memory"]
                    and batch_idx == 0
                    and rank == 0
                    and epoch == 0
                ):
                    if backend_manager.backend == Backend.CUDA:
                        torch.cuda.memory._record_memory_history()

                # Log memory usage
                if batch_idx == 0 and epoch == 0 and (rank == 0 or args["verbose"]):
                    mem_stats = get_memory_stats(backend_manager, local_rank)
                    reserved_before_forward = mem_stats["reserved"]
                    memory_stats.append(
                        f"Rank {rank}: Before forward: {reserved_before_forward/2**30:.2f} GiB"
                    )
                    if args["log_to"] == "wandb":
                        logger.log(
                            {"memory/allocated_before_forward": mem_stats["allocated"]},
                            rank,
                        )
                        logger.log(
                            {"memory/reserved_before_forward": reserved_before_forward},
                            rank,
                        )

                # Forward pass
                with sync_context:
                    with autocast:
                        output = model(
                            batch["input_ids"].to(device),
                            labels=batch["labels"].to(device),
                            attention_mask=None,
                        )
                        loss = output.loss

                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                    # Log memory usage
                    if batch_idx == 0 and epoch == 0 and (rank == 0 or args["verbose"]):
                        mem_stats = get_memory_stats(backend_manager, local_rank)
                        reserved_after_forward = mem_stats["reserved"]
                        memory_stats.append(
                            f"Rank {rank}: After forward: {reserved_after_forward/2**30:.2f} GiB"
                        )
                        if args["log_to"] == "wandb":
                            logger.log(
                                {
                                    "memory/allocated_after_forward": mem_stats[
                                        "allocated"
                                    ]
                                },
                                rank,
                            )
                            logger.log(
                                {
                                    "memory/reserved_after_forward": reserved_after_forward
                                },
                                rank,
                            )

                    # Backward pass
                    if scale_grads:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Record loss
                bs = batch["input_ids"].shape[0]
                ddp_loss[0] += loss.item() * bs * gradient_accumulation_steps
                ddp_loss[1] += bs

                # Step the optimizer (w/ gradient accumulation)
                if accumulate_grads:
                    if args["apply_gradient_clipping"] and (
                        args["grad_norm"] is not None
                    ):
                        model.clip_grad_norm_(args["grad_norm"], norm_type=2.0)
                    if scale_grads:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    # avoid overhead when lr is constant.
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    progress_bar.update(1)

                # Log memory usage after backward
                if batch_idx == 0 and epoch == 0 and (rank == 0 or args["verbose"]):
                    mem_stats = get_memory_stats(backend_manager, local_rank)
                    reserved_after_backward = mem_stats["reserved"]
                    memory_stats.append(
                        f"Rank {rank}: After backward: {reserved_after_backward/2**30:.2f} GiB"
                    )
                    if args["log_to"] == "wandb":
                        logger.log(
                            {"memory/allocated_after_backward": mem_stats["allocated"]},
                            rank,
                        )
                        logger.log(
                            {"memory/reserved_after_backward": reserved_after_backward},
                            rank,
                        )

                # Delete the output so more memory frees up before the next forward pass
                output = None
                loss = None

                # Stop logging memory (first iter)
                if (
                    args["profile_memory"]
                    and batch_idx == 0
                    and rank == 0
                    and epoch == 0
                ):
                    if backend_manager.backend == Backend.CUDA:
                        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
                        torch.cuda.memory._record_memory_history(
                            enabled=None
                        )  # Stop recording

                # Log loss every gradient update steps
                if accumulate_grads:
                    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
                    if rank == 0:
                        log_loss = ddp_loss[0] / ddp_loss[1]
                        if lr_scheduler is not None:
                            log_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            log_lr = args["lr"]
                        update_progress_bar(progress_bar, epoch, log_loss, log_lr, rank)
                        if args["log_to"] == "wandb":
                            logger.log({"loss": log_loss, "lr": log_lr}, rank)
                    ddp_loss = torch.zeros(2).to(device)

                if rank == 0 and args["verbose"]:
                    print(f"Batch idx {batch_idx}")

                prof.step()

                # Primarily for debugging
                if args["max_steps"] > 0 and batch_idx > args["max_steps"]:
                    if rank == 0:
                        print("Max steps reached, skipping rest of epoch")
                    break

            # Print + log peak memory usage for the whole fourth step of training
            if epoch == 0 and (rank == 0 or args["verbose"]):
                if backend_manager.backend == Backend.CUDA:
                    peak_allocated_memory = torch.cuda.max_memory_allocated(local_rank)
                    peak_reserved_memory = torch.cuda.max_memory_reserved(local_rank)
                    memory_stats.append(
                        f"Rank {rank}: Peak allocated memory: {peak_allocated_memory/2**30:.2f} GiB"
                    )
                    memory_stats.append(
                        f"Rank {rank}: Peak reserved memory:  {peak_reserved_memory/2**30:.2f} GiB"
                    )
                    if args["log_to"] == "wandb":
                        logger.log(
                            {"memory/allocated_peak": peak_allocated_memory}, rank
                        )
                        logger.log({"memory/reserved_peak": peak_reserved_memory}, rank)

    # Synchronize at the end and record time
    if init_end_event is not None:
        init_end_event.record()
    if backend_manager.capabilities.supports_distributed and world_size > 1:
        dist.barrier()
    if backend_manager.backend == Backend.CUDA:
        torch.cuda.synchronize()

    if rank == 0:
        print("Finished training", rank)

    # Print time, model, & memory stats
    if init_start_event is not None and init_end_event is not None:
        time_taken = init_start_event.elapsed_time(init_end_event) / 1000
        if backend_manager.capabilities.supports_distributed and world_size > 1:
            dist.barrier()
        if backend_manager.backend == Backend.CUDA:
            torch.cuda.synchronize()
        if rank == 0:
            print(f"Training elapsed time: {time_taken} sec")
            logger.log({"time_taken": time_taken}, rank)
    else:
        if rank == 0:
            print("Training completed (timing not available for non-CUDA backends)")
    for line in memory_stats:
        print(line)

    # End logging
    logger.finish(rank=rank)

    # Save model - ref: https://github.com/pytorch/pytorch/issues/98823
    # HQQLinear custom state_dict() method causes issues when saving.
    # Model is saved fine when `state_dict()` method is removed.
    # Non param/buffer types are not saved with FSDP.
    # It might be better to just save the trained lora layers.
    # summon_full_params on lora layers and save.
    if args["save_model"]:
        if rank == 0:
            os.makedirs(args["output_dir"], exist_ok=True)
        dist.barrier()
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        if args["train_type"] in [
            "custom_lora",
            "custom_qlora",
            "hqq_lora",
            "hqq_dora",
            "bnb_dora",
            "bnb_llama_pro",
            "hqq_llama_pro",
        ]:
            cpu_state_dict = {}
            if args["train_type"] in ["bnb_llama_pro", "hqq_llama_pro"]:
                trainable_fsdp_modules = [
                    (n, m)
                    for n, m in model.named_modules()
                    if n.endswith(tuple(new_layer_names))
                ]
            else:
                trainable_fsdp_modules = [
                    (n, m)
                    for n, m in model.named_modules()
                    if n.endswith(("lora_AB", "dora_layer", "magnitude_layer"))
                ]
            for prefix, module in trainable_fsdp_modules:
                prefix = (
                    prefix.replace("_fsdp_wrapped_module.", "")
                    .replace("_checkpoint_wrapped_module.", "")
                    .replace("_offload_wrapped_module.", "")
                )
                if args["verbose"]:
                    print(f"Saving {prefix}")
                with FSDP.state_dict_type(
                    module, StateDictType.FULL_STATE_DICT, save_policy
                ):
                    cpu_state_dict = {
                        **cpu_state_dict,
                        **{f"{prefix}.{k}": v for k, v in module.state_dict().items()},
                    }
                dist.barrier()
                torch.cuda.synchronize()
            if rank == 0:
                print("Saving trained LoRA weights.")
                save_file(
                    cpu_state_dict,
                    os.path.join(args["output_dir"], "model_state_dict.safetensors"),
                )
                print("Done", rank)
        else:
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state_dict = model.state_dict()
                if rank == 0:
                    print("Saving full model weights.")
                    save_file(
                        cpu_state_dict,
                        os.path.join(
                            args["output_dir"], "model_state_dict.safetensors"
                        ),
                    )
                    print("Done", rank)

    dist.barrier()  # Stop other processes ending while model saving - probably not needed?

    # Clean up
    dist.destroy_process_group()


def validate_args(args):
    if args["n_bits"] != 4 and args["train_type"] not in [
        "hqq_lora",
        "hqq_dora",
        "hqq_llama_pro",
    ]:
        raise ValueError(
            f"train_type={args['train_type']} doesn't support n_bits={args['n_bits']}. Either don't pass n_bits (to use default of 4) or use any of the hqq training types."
        )


# Main entry point, validate and package arguments, then call fsdp_main. Scripts importing train.py as an import can invoke this function (invoking main() directly leads to call_parse() overwriting arguments).
def fsdp_qlora(
    world_size: int = -1,  # Number of GPUs to use. -1 = all available GPUs.
    backend: str = "cuda",  # Backend to use for training. 'auto' for automatic detection.
    train_type: str = "qlora",  # "full", "lora", "qlora", or "custom_qlora"
    llama_pro_path: str = None,  # Path to the quantized llama pro model
    batch_size: int = 1,  # Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
    context_length: int = 512,  # Max length of input sequence (in tokens)
    gradient_accumulation_steps: int = 1,  # How many steps to accumulate gradients over (increases effective batch size)
    num_epochs: int = 1,  # How many epochs of training to do
    dataset: str = "alpaca_sample",  # alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
    dataset_samples: int = 512,  # Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
    sharding_strategy: str = "full_shard",  # Sharding strategy for FSDP
    use_gradient_checkpointing: bool = True,  # Use FSDP's activation checkpointing
    reentrant_checkpointing: bool = False,  # Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
    use_cpu_offload: bool = True,  # Use FSDP's CPU offloading
    use_activation_cpu_offload: bool = False,  # Use FSDP's activation CPU offloading
    low_memory: bool = True,  # Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
    no_sync: bool = False,  # Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
    precision: str = "bf16",  # Training precision. autocast precisions use mixed precision
    model_name: str = "meta-llama/Llama-2-7b-hf",  # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    save_model: bool = False,  # Save the resulting model
    output_dir: str = "output",  # Output directory to save the final model to
    lora_rank: int = 64,  # LoRA rank for lora/qlora
    lora_alpha: int = 16,  # LoRA alpha for lora/qlora
    lora_dropout: float = 0.1,  # LoRA dropout for lora/qlora
    lora_target_modules: str = "all",  # If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
    verbose: bool = False,  # Whether to print extra info for debugging
    lr: float = 1e-5,  # Learning rate
    apply_gradient_clipping: bool = False,  # Apply gradient norm clipping
    grad_norm: float = 0.3,  # Gradient norm clipping
    wd: float = 0.1,  # Weight decay
    profile_memory: bool = False,  # Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
    optimizer: str = "adamw",  # Optimizer. PyTorch 2.4 nightly adds CPU fused Adam/AdamW which should improve offload training speed.
    lr_scheduler: str = "constant",  # Learning Rate Scheduler. linear and cosine warm up for 10% of training steps.
    loading_workers: int = -1,  # Number of layers to load and quantize in parallel per GPU. Default of -1 uses heuristics to set worker count.
    log_to: str = "tqdm",  # Where to log output
    log_every_n_steps: int = 1,  # Log every n training steps
    master_addr: str = "localhost",  # For distributed training
    master_port: str = "12355",  # For distributed training, must be the same for all processes
    seed: int = 42,  # Random seed
    project_name: str = "fsdp_qlora",  # For wandb logging
    name: str = None,  # For wandb logging
    group: str = None,  # For wandb logging
    entity: str = None,  # For wandb logging
    n_bits: int = 4,  # passed to hqq
    hqq_group_size: int = 64,  # HQQ quantization group size
    # Profiling args
    profile: bool_arg = False,  # Whether to profile with torch.profiler
    profiling_output: str = "profiles",  # Output file prefix for profiling
    overwrite_profiling_output: bool = True,  # Overwrite output directory
    with_stack: bool_arg = False,  # Output stacks for profiling. Note that setting export_memory_timeline will automatically export traces since `with_stack` must be true to profile memory.
    with_shapes: bool_arg = False,  # Output shapes for profiling. Can impact performance.  Note that setting export_memory_timeline will automatically export traces since `with_shapes` must be true to profile memory.
    export_trace: bool_arg = True,  # Output trace for profiling
    export_memory_timeline: bool_arg = False,  # Output memory timelinefor profiling
    wait_steps: int = 1,  # Wait steps when running profiler.  Only used if repeat != 0.
    profiler_warmup_steps: int = 1,  # Warmup steps when running profiler
    active_steps: int = 2,  # Active steps when running profiler
    repeat: int = 0,  # Number of profiler cycles (wait + warmup + active) if > 0, else repeats forever
    profiling_frequency: int = 10,  # Profiling frequency in steps.  Only used if repeat == 0, in which case wait_steps will be set to profiling_frequency - (profiler_warmup_steps + active_steps) such that the effective cycle length == profiling_frequency
    max_steps: int = -1,  # Max number of training steps (in units of batches) per epoch. -1 means no max_steps, otherwise training loop breaks after `max_steps` each epoch.
):
    """
    Train a model with FSDP and QLoRA/QDoRA.

    Args:

        world_size: Number of GPUs to use. -1 = all available GPUs.
        backend: Backend to use for training. Options: 'cuda', 'mps', 'mlx', 'cpu', 'auto'
        train_type: "full", "lora", "qlora", or "custom_qlora"
        llama_pro_path: Path to the quantized llama pro model
        batch_size: Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
        context_length: Max length of input sequence (in tokens)
        gradient_accumulation_steps: How many steps to accumulate gradients over (increases effective batch size)
        num_epochs: How many epochs of training to do
        dataset: alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
        dataset_samples: Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
        sharding_strategy: Sharding strategy for FSDP
        use_gradient_checkpointing: Use FSDP's activation checkpointing
        reentrant_checkpointing: Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
        use_cpu_offload: Use FSDP's CPU offloading
        use_activation_cpu_offload: Use FSDP's activation CPU offloading
        low_memory: Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
        no_sync: Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
        precision: Training precision. autocast precisions use mixed precision
        model_name: Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        save_model: Save the resulting model
        output_dir: Output directory to save the final model to
        lora_rank: LoRA rank for lora/qlora
        lora_alpha: LoRA alpha for lora/qlora
        lora_dropout: LoRA dropout for lora/qlora
        lora_target_modules: If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
        verbose: Whether to print extra info for debugging
        lr: Learning rate
        apply_gradient_clipping: Apply gradient norm clipping
        grad_norm: Gradient norm clipping
        wd: Weight decay
        profile_memory: Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
        optimizer: Optimizer. PyTorch 2.4 nightly adds CPU fused Adam/AdamW which should improve offload training speed.
        lr_scheduler: Learning Rate Scheduler type. Options: constant, linear, cosine, cosine_with_restarts, polynomial, exponential
        warmup_steps: Number of warmup steps (highest priority if set)
        warmup_ratio: Ratio of total steps to use for warmup (e.g., 0.1 for 10%) 
        cosine_min_lr_ratio: Minimum learning rate as a ratio of initial LR for cosine scheduler
        loading_workers: Number of layers to load and quantize in parallel per GPU. Default of -1 uses heuristics to set worker count.
        log_to: Where to log output
        master_addr: For distributed training
        master_port: For distributed training, must be the same for all processes
        seed: Random seed
        project_name: For wandb logging
        name: For wandb logging
        group: For wandb logging
        entity: For wandb logging
        n_bits: passed to hqq
        profiling_output: Output file for profiling
    """

    # Initialize backend manager for validation and configuration
    backend_arg = backend if backend != "auto" else None
    backend_manager = BackendManager(backend=backend_arg, verbose=verbose)

    # Validate backend selection
    if backend != "auto" and backend_manager.backend != Backend(backend):
        raise ValueError(
            f"Requested backend '{backend}' is not available on this system."
        )
    
    # Load conditional imports based on detected backend
    load_conditional_imports(str(backend_manager.backend))

    # Set world size based on backend
    if world_size == -1:
        if backend_manager.backend == Backend.CUDA:
            world_size = torch.cuda.device_count()
        elif backend_manager.backend in [Backend.MPS, Backend.MLX]:
            world_size = 1  # Apple Silicon doesn't support multi-GPU
        else:  # CPU
            world_size = 1
    print(f"World size: {world_size}")
    print(f"Using backend: {backend_manager.backend}")

    # Get all args which will be passed to fsdp_main
    args = dict(locals())
    args["backend_manager"] = backend_manager  # Pass backend manager to fsdp_main
    set_seed(args["seed"])
    validate_args(args)
    if args["verbose"]:
        print(args)

    # If lora_target_modules is 'all', set sensible defaults for llama + mistral type modules
    # See peft.utils.constants -> TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING for the current defaults
    if lora_target_modules == "all":
        args["lora_target_modules"] = [
            "k_proj",
            "q_proj",
            "v_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    elif lora_target_modules.lower() == "default":
        args["lora_target_modules"] = None

    # Backend-specific configuration adjustments
    if backend_manager.backend == Backend.MPS:
        # MPS doesn't support bfloat16
        if args["precision"] in ["bf16", "bf16_autocast", "bf16_buffers_autocast"]:
            print("Warning: MPS backend doesn't support bfloat16. Switching to fp16.")
            args["precision"] = (
                "fp16_autocast" if "autocast" in args["precision"] else "fp16"
            )
        # MPS has limited quantization support
        if args["train_type"] in ["qlora", "custom_qlora"]:
            print(
                "Warning: MPS backend has limited 4-bit quantization support. Training may be slower."
            )
    elif backend_manager.backend == Backend.MLX:
        # MLX doesn't support distributed training
        if world_size > 1:
            raise ValueError(
                "MLX backend doesn't support distributed training. Set world_size=1."
            )
        # MLX uses different model loading mechanism
        args["backend_type"] = "mlx"
    elif backend_manager.backend == Backend.CPU:
        # CPU training is very slow, warn the user
        print(
            "Warning: CPU backend selected. Training will be significantly slower than GPU."
        )
        if args["train_type"] in ["qlora", "custom_qlora", "hqq_lora"]:
            print("Warning: Quantized training on CPU may have limited support.")

    # Original CUDA-specific check (now only for CUDA backend)
    if backend_manager.backend == Backend.CUDA:
        if (
            args["precision"] in ["bf16", "bf16_autocast", "bf16_buffers_autocast"]
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("Current CUDA device does not support bfloat16")

    # Set no_sync if using cpu_offload and gradient accumulation. Turn off if not using gradient accumulation
    if args["use_cpu_offload"] and args["gradient_accumulation_steps"] > 1:
        args["no_sync"] = True
    elif args["no_sync"] and args["gradient_accumulation_steps"] == 1:
        args["no_sync"] = False

    if args["train_type"] in ["hqq_lora"] and HQQLinear is None:
        raise ValueError(
            "HQQ is required to train with `train_type='hqq_lora'`. See ReadMe for details."
        )

    if (
        args["optimizer"] in ["fused_adam", "fused_adamw"]
        and args["use_cpu_offload"]
        and parse(torch.__version__) < parse("2.4dev")
    ):
        raise ValueError(
            f"Optimizer '{args['optimizer']}' with `use_cpu_offload=True` requires at least PyTorch 2.4 Nightly with fused Adam/AdamW CPU support."
        )

    # Run
    if backend_manager.backend == Backend.CUDA:
        mp.spawn(
            fsdp_main,
            args=(world_size, args),
            nprocs=torch.cuda.device_count(),
            join=True,
        )
    else:
        # For non-CUDA backends, run single process
        fsdp_main(0, world_size, args)


# Entry point, one line wrapper around fsdp_qlora(), use fastcore's call_parse to parse args from command line
@call_parse()
def main(
    world_size: int = -1,  # Number of GPUs to use. -1 = all available GPUs.
    backend: Param(
        "Backend to use for training. 'auto' for automatic detection.",
        choices=["cuda", "mps", "mlx", "cpu", "auto"],
    ) = "cuda",
    train_type: Param(
        "",
        choices=[
            "full",
            "lora",
            "qlora",
            "custom_qlora",
            "custom_lora",
            "hqq_lora",
            "hqq_dora",
            "bnb_dora",
            "bnb_llama_pro",
            "hqq_llama_pro",
        ],
    ) = "qlora",  # "full", "lora", "qlora", or "custom_qlora"
    llama_pro_path: str = None,  # Path to the quantized llama pro model
    batch_size: int = 1,  # Batch size per GPU. Effective BS = batch_size * world_size * gradient_accumulation_steps
    context_length: int = 512,  # Max length of input sequence (in tokens)
    gradient_accumulation_steps: int = 1,  # How many steps to accumulate gradients over (increases effective batch size)
    num_epochs: int = 1,  # How many epochs of training to do
    dataset: Param(
        "", choices=["alpaca", "alpaca_sample", "dummy", "guanaco", "sql", "orca_math"]
    ) = "alpaca_sample",  # alpaca, alpaca_sample (for a 128-sample test) or "dummy" for 16 long dummy samples
    dataset_samples: int = 512,  # Number of samples in an epoch if using "alpaca_sample" or "dummy" dataset
    sharding_strategy: Param(
        "",
        choices=[
            "full_shard",
            "shard_grad_op",
            "ddp",
            "hybrid_full_shard",
            "hybrid_shard_grad_op",
        ],
    ) = "full_shard",  # Sharding strategy for FSDP
    use_gradient_checkpointing: bool_arg = True,  # Use FSDP's activation checkpointing
    reentrant_checkpointing: bool_arg = False,  # Use re-entrant autograd activation checkpointing. Setting to True can use less GPU memory with BNB QLoRA
    use_cpu_offload: bool_arg = True,  # Use FSDP's CPU offloading
    use_activation_cpu_offload: bool_arg = False,  # Use FSDP's activation CPU offloading
    low_memory: bool_arg = True,  # Load one copy of the model into CPU memory before sharding with FSDP. For QLoRA, quantizes each layer individually on GPU before placing on CPU.
    no_sync: bool_arg = False,  # Prevent gradient sync until update step. Likely uses more memory. Required for `use_cpu_offload` and `gradient_accumulation_steps > 1`
    precision: Param(
        "",
        choices=[
            "fp32",
            "bf16",
            "fp16_autocast",
            "bf16_autocast",
            "bf16_buffers_autocast",
        ],
    ) = "bf16",  # Training precision. autocast precisions use mixed precision
    model_name: str = "meta-llama/Llama-2-7b-hf",  # Which model to train - e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    save_model: bool_arg = False,  # Save the resulting model
    output_dir: str = "output",  # Output directory to save the final model to
    lora_rank: int = 64,  # LoRA rank for lora/qlora
    lora_alpha: int = 16,  # LoRA alpha for lora/qlora
    lora_dropout: float = 0.1,  # LoRA dropout for lora/qlora
    lora_target_modules: Param(
        "", choices=["all", "default"]
    ) = "all",  # If 'default', uses peft defaults. Use 'all' for our best guess for Llama models
    verbose: bool_arg = False,  # Whether to print extra info for debugging
    lr: float = 1e-5,  # Learning rate
    apply_gradient_clipping: bool_arg = False,  # Apply gradient norm clipping
    grad_norm: float = 0.3,  # Gradient norm clipping
    wd: float = 0.1,  # Weight decay
    profile_memory: bool_arg = False,  # Profile memory usage for the first few batches. Keep false for training. May increase memory usage.
    optimizer: Param(
        "", choices=["adamw", "adam", "sgd", "adadelta"]
    ) = "adamw",  # Optimizer
    lr_scheduler: Param(
        "", choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "exponential"]
    ) = "constant",  # Learning Rate Scheduler type
    warmup_steps: int = None,  # Number of warmup steps (highest priority if set)
    warmup_ratio: float = None,  # Ratio of total steps to use for warmup (e.g., 0.1 for 10%)
    cosine_min_lr_ratio: float = 0.1,  # Minimum learning rate as a ratio of initial LR for cosine scheduler
    loading_workers: int = -1,  # Number of layers to load and quantize in parallel per GPU. Default of -1 uses heuristics to set worker count.
    log_to: Param(
        "", choices=["tqdm", "wandb", "stdout"]
    ) = "tqdm",  # Where to log output
    log_every_n_steps: int = 1,  # Log every n training steps
    master_addr: str = "localhost",  # For distributed training
    master_port: str = "12355",  # For distributed training, must be the same for all processes
    seed: int = 42,  # Random seed
    project_name: str = "fsdp_qlora",  # For wandb logging
    name: str = None,  # For wandb logging
    group: str = None,  # For wandb logging
    entity: str = None,  # For wandb logging
    n_bits: int = 4,  # passed to hqq
    hqq_group_size: int = 64,  # HQQ quantization group size
    profile: bool_arg = False,  # Whether to profile with torch.profiler
    profiling_output: str = "profiles",  # Output file prefix for profiling
    with_stack: bool_arg = False,  # Output stacks for profiling. Note that setting export_memory_timeline will automatically export traces since `with_stack` must be true to profile memory.
    with_shapes: bool_arg = False,  # Output shapes for profiling. Can impact performance.  Note that setting export_memory_timeline will automatically export traces since `with_shapes` must be true to profile memory.
    export_trace: bool_arg = True,  # Output trace for profiling
    export_memory_timeline: bool_arg = False,  # Output memory timelinefor profiling
    wait_steps: int = 0,  # Wait steps when running profiler.  Only used if repeat != 0.
    profiler_warmup_steps: int = 1,  # Warmup steps when running profiler
    active_steps: int = 2,  # Active steps when running profiler
    repeat: int = 0,  # Number of profiler cycles (wait + warmup + active) if > 0, else repeats forever
    profiling_frequency: int = 10,  # Profiling frequency in steps.  Only used if repeat == 0, in which case wait_steps will be set to profiling_frequency - (profiler_warmup_steps + active_steps) such that the effective cycle length == profiling_frequency
    max_steps: int = -1,  # Max number of training steps (in units of batches) per epoch. -1 means no max_steps, otherwise training loop breaks after `max_steps` each epoch.
):
    fsdp_qlora(**locals())
