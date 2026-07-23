# Adapted from https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/models/loader/

from typing import Generator
import os
import contextlib
from collections.abc import Generator, Callable

from tqdm import tqdm
import torch
from torch import nn
from torch.distributed import init_device_mesh, DeviceMesh
from torch.distributed.checkpoint.state_dict import set_model_state_dict, get_model_state_dict, StateDictOptions
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from safetensors.torch import safe_open
from pfm.utils.logging import get_logger


# TODO(PY): move this to utils elsewhere
@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


def safetensors_weights_iterator(hf_weights_files: list[str]) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    enable_tqdm = not torch.distributed.is_initialized(
    ) or torch.distributed.get_rank() == 0
    device = "cpu"
    for st_file in tqdm(
        hf_weights_files,
        desc="Loading safetensors checkpoint shards",
        disable=not enable_tqdm,
        bar_format=_BAR_FORMAT,
    ):
        with safe_open(st_file, framework="pt", device=device) as f:
            for name in f.keys():  # noqa: SIM118
                param = f.get_tensor(name)
                yield name, param


def pt_weights_iterator(hf_weights_files: list[str]) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""
    device = "cpu"
    enable_tqdm = not torch.distributed.is_initialized(
    ) or torch.distributed.get_rank() == 0
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm,
        bar_format=_BAR_FORMAT,
    ):
        state = torch.load(bin_file, map_location=device, weights_only=True)
        yield from state.items()
        del state


def maybe_load_fsdp_model(
    model: nn.Module,
    hsdp_shard_dim: int,
    reshard_after_forward: bool,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    fsdp_inference: bool = False,
    output_dtype: torch.dtype | None = None,
    training_mode: bool = True,
    pin_cpu_memory: bool = True,
) -> torch.nn.Module:
    """
    Load the model with FSDP if is training, else load the model without FSDP.
    """
    logger = get_logger()
    mp_policy = MixedPrecisionPolicy(param_dtype,
                                     reduce_dtype,
                                     output_dtype,
                                     cast_forward_inputs=True)

    # Check if we should use FSDP
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    assert world_size % hsdp_shard_dim == 0, f"world_size {world_size} must be divisible by hsdp_shard_dim {hsdp_shard_dim}"
    hsdp_replicate_dim = world_size // hsdp_shard_dim

    use_fsdp = training_mode or fsdp_inference
    if hsdp_shard_dim * hsdp_replicate_dim <= 1:
        use_fsdp = False
        logger.warning(
            f"hsdp_replicate_dim * hsdp_shard_dim = {hsdp_replicate_dim}x{hsdp_shard_dim} <= 1, not using FSDP.")

    if use_fsdp:
        device_mesh = init_device_mesh(
            "cuda",
            # (Replicate(), Shard(dim=0))
            mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
            mesh_dim_names=("replicate", "shard"),
        )
        shard_model(model,
                    cpu_offload=cpu_offload,
                    reshard_after_forward=reshard_after_forward,
                    mp_policy=mp_policy,
                    mesh=device_mesh,
                    fsdp_shard_conditions=model._fsdp_shard_conditions,
                    pin_cpu_memory=pin_cpu_memory)

    return model


def shard_model(
    model,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy | None = MixedPrecisionPolicy(),  # noqa
    mesh: DeviceMesh | None = None,
    fsdp_shard_conditions: list[Callable[[str, nn.Module], bool]] = [],  # noqa
    pin_cpu_memory: bool = True,
) -> None:
    """
    Utility to shard a model with FSDP using the PyTorch Distributed fully_shard API.

    This method will over the model's named modules from the bottom-up and apply shard modules
    based on whether they meet any of the criteria from shard_conditions.

    Args:
        model (TransformerDecoder): Model to shard with FSDP.
        shard_conditions (List[Callable[[str, nn.Module], bool]]): A list of functions to determine
            which modules to shard with FSDP. Each function should take module name (relative to root)
            and the module itself, returning True if FSDP should shard the module and False otherwise.
            If any of shard_conditions return True for a given module, it will be sharded by FSDP.
        cpu_offload (bool): If set to True, FSDP will offload parameters, gradients, and optimizer
            states to CPU.
        reshard_after_forward (bool): Whether to reshard parameters and buffers after
            the forward pass. Setting this to True corresponds to the FULL_SHARD sharding strategy
            from FSDP1, while setting it to False corresponds to the SHARD_GRAD_OP sharding strategy.
        mesh (Optional[DeviceMesh]): Device mesh to use for FSDP sharding under multiple parallelism.
            Default to None.
        fsdp_shard_conditions (List[Callable[[str, nn.Module], bool]]): A list of functions to determine
            which modules to shard with FSDP.
        pin_cpu_memory (bool): If set to True, FSDP will pin the CPU memory of the offloaded parameters.

    Raises:
        ValueError: If no layer modules were sharded, indicating that no shard_condition was triggered.
    """

    if fsdp_shard_conditions is None or len(fsdp_shard_conditions) == 0:
        logger = get_logger()
        logger.warning(
            "The FSDP shard condition list is empty or None. No modules will be sharded in %s",
            type(model).__name__)
        return

    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": mesh,
        "mp_policy": mp_policy,
    }
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(
            pin_memory=pin_cpu_memory)

    # iterating in reverse to start with
    # lowest-level modules first
    num_layers_sharded = 0
    # TODO(will): don't reshard after forward for the last layer to save on the
    # all-gather that will immediately happen Shard the model with FSDP,
    for n, m in reversed(list(model.named_modules())):
        if any([
                shard_condition(n, m)
                for shard_condition in fsdp_shard_conditions
        ]):
            fully_shard(m, **fsdp_kwargs)
            num_layers_sharded += 1

    if num_layers_sharded == 0:
        raise ValueError(
            "No layer modules were sharded. Please check if shard conditions are working as expected."
        )

    # Finally shard the entire model to account for any stragglers
    fully_shard(model, **fsdp_kwargs)
