from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_distributed(config: dict) -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = config.get("ddp", {}).get("backend", "nccl")
        if not torch.cuda.is_available():
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return distributed, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        distributed_barrier()
        dist.destroy_process_group()


def is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def distributed_barrier(local_rank: int | None = None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    backend = dist.get_backend()
    if backend == "nccl" and torch.cuda.is_available():
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
