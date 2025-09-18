from typing import List, Dict, Any, Tuple

import torch
import torch.distributed as dist


def create_zero_param_groups(
    param_groups: List[Dict[str, Any]], rank: int, world_size: int
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Create parameter groups for ZeRO-1 optimizer sharding and generate a map
    of parameter owners for broadcasting.

    Args:
        param_groups: Original parameter groups (list of dicts with 'params' key)
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        A tuple containing:
        - List of parameter groups containing only parameters owned by this rank.
        - A list where the index corresponds to a parameter's global index and
          the value is the rank of the process that owns it. This is the pre-computed
          index for the broadcast function.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")

    sharded_groups = []
    owner_ranks = []
    global_param_idx = 0

    for group in param_groups:
        # Copy all group settings except params
        sharded_group = {k: v for k, v in group.items() if k != "params"}
        sharded_group["params"] = []

        # Add only parameters owned by this rank
        for param in group["params"]:
            owner_rank = global_param_idx % world_size
            owner_ranks.append(owner_rank)

            if owner_rank == rank:
                sharded_group["params"].append(param)
            global_param_idx += 1

        # Only add group if it has parameters for this rank
        if sharded_group["params"]:
            sharded_groups.append(sharded_group)

    return sharded_groups, owner_ranks


def broadcast_zero_params(all_params: List[torch.Tensor], owner_ranks: List[int]):
    """
    Broadcast updated parameters from their owning ranks to all other ranks
    using a pre-computed index of owner ranks.

    Args:
        all_params: List of all model parameters (in the same order across all ranks)
        owner_ranks: A list mapping each parameter's global index to its owner rank.
    """
    with torch.no_grad():
        for i, param in enumerate(all_params):
            owner_rank = owner_ranks[i]
            dist.broadcast(param.data, src=owner_rank)
