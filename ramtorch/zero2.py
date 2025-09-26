import torch
import torch.distributed as dist


def setup_grad_sharding_hooks(rank_param_groups, current_rank):
    """
    Setup backward hooks for gradient sharding using the same parameter
    assignment structure as ZeRO-1.

    Args:
        rank_param_groups: Dict from create_zero_param_groups()
        current_rank: Current process rank
    """
    # Create a mapping from parameter to its owner rank
    param_to_owner = {}

    for owner_rank, param_groups in rank_param_groups.items():
        for group in param_groups:
            for param in group["params"]:
                param_to_owner[param] = owner_rank

    def create_grad_hook(param):

        # this grad hook will fire after grad is computed
        # and it will immediately reduce the grad towards the owner rank
        def hook(grad):
            owner_rank = param_to_owner[param]

            # reduce towards the owner rank
            dist.reduce(grad, dst=owner_rank, op=dist.ReduceOp.SUM)

            if current_rank == owner_rank:
                # owner rank keeps the reduced gradient
                return grad
            else:
                # boot grad on non-owner ranks
                return None

        return hook

    # register hooks for all pre sharded parameters
    for owner_rank, param_groups in rank_param_groups.items():
        for group in param_groups:
            for param in group["params"]:
                param.register_hook(create_grad_hook(param))
