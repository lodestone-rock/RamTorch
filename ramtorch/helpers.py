from collections import OrderedDict
from typing import Callable, Dict
import torch


def add_custom_hooks(tensor: torch.Tensor, hook_name: str = "_custom_hooks"):
    """
    Add a custom hook dictionary to a tensor, similar to _post_accumulate_grad_hooks

    Args:
        tensor: The tensor to add hooks to
        hook_name: Name of the hook attribute (default: "_custom_hooks")

    Returns:
        The tensor with the hook attribute added
    """
    if not hasattr(tensor, hook_name):
        setattr(tensor, hook_name, OrderedDict())
        setattr(tensor, f"{hook_name}_counter", 0)
    return tensor


def register_ramtorch_hook(tensor: torch.Tensor, hook: Callable, hook_name: str) -> int:
    """
    Register a hook to the tensor

    Args:
        tensor: The tensor to register the hook on
        hook: Callable to register
        hook_name: Name of the hook attribute

    Returns:
        hook_id: Integer ID to remove the hook later
    """
    # Ensure hook dict exists
    if not hasattr(tensor, hook_name):
        add_custom_hooks(tensor, hook_name)

    hooks = getattr(tensor, hook_name)
    counter_name = f"{hook_name}_counter"
    counter = getattr(tensor, counter_name)

    # Add hook with unique ID
    hook_id = counter
    hooks[hook_id] = hook
    setattr(tensor, counter_name, counter + 1)

    return hook_id


def register_ramtorch_grad_hook(module, hook_fn, param_names=None):

    handles = []

    for name, param in module.named_parameters():
        if param.requires_grad:
            # Filter by parameter names if specified
            if param_names is None or name in param_names:
                if hasattr(param, "is_ramtorch") and param.is_ramtorch:
                    handle = register_ramtorch_hook(
                        param, hook_fn, "_ramtorch_backward_hooks"
                    )
                else:
                    handle = param.register_hook(hook_fn)
                # TODO this works but if it not ramtorch then i need to add handles
                handles.append(handle)

    return handles


def register_ramtorch_post_accumulate_grad_hook(module, hook_fn, param_names=None):

    handles = []

    for name, param in module.named_parameters():
        if param.requires_grad:
            # Filter by parameter names if specified
            if param_names is None or name in param_names:
                if hasattr(param, "is_ramtorch") and param.is_ramtorch:
                    handle = register_ramtorch_hook(
                        param, hook_fn, "_ramtorch_post_accumulate_grad_hooks"
                    )
                else:
                    handle = param.register_post_accumulate_grad_hook(hook_fn)
                # TODO this works but if it not ramtorch then i need to add handles
                handles.append(handle)

    return handles
