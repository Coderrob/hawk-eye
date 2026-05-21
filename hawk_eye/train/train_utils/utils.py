import pathlib
from typing import Any, Dict, List

import torch


def save_model(model: torch.nn.Module, save_path: pathlib.Path) -> None:
    """Save model state to disk.

    Args:
        model: Model whose state should be saved.
        save_path: Destination checkpoint path.
    """
    torch.save(model.state_dict(), save_path)


def _sgd_optimizer(optim_cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        add_weight_decay(model, float(optim_cfg["weight_decay"])),
        lr=float(optim_cfg["lr"]),
        momentum=float(optim_cfg["momentum"]),
        weight_decay=0,
        nesterov=True,
    )


def _rmsprop_optimizer(
    optim_cfg: dict, model: torch.nn.Module
) -> torch.optim.Optimizer:
    return torch.optim.RMSprop(
        add_weight_decay(model, float(optim_cfg["weight_decay"])),
        lr=float(optim_cfg["lr"]),
        momentum=float(optim_cfg["momentum"]),
        weight_decay=0,
    )


def _adamw_optimizer(optim_cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        add_weight_decay(model, float(optim_cfg["weight_decay"])),
        lr=float(optim_cfg["lr"]),
        weight_decay=0,
    )


def create_optimizer(optim_cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Take in optimizer config and create the optimizer for training.

    Args:
        optim_cfg: The parameters for the optimizer generation.
        model: The model to create an optimizer for. We need to read this model's
            parameters.

    Returns:
        An optimizer for the model.
    """

    optimizer_factories = {
        "sgd": _sgd_optimizer,
        "rmsprop": _rmsprop_optimizer,
        "adamw": _adamw_optimizer,
    }
    name = optim_cfg.get("type", "sgd").lower()
    if name not in optimizer_factories:
        raise ValueError(f"Improper optimizer supplied: {name}.")

    return optimizer_factories[name](optim_cfg, model)


def _uses_weight_decay(param: torch.Tensor) -> bool:
    return len(param.shape) != 1


def _weight_decay_group(
    param: torch.Tensor, decay: list, no_decay: list
) -> List[torch.Tensor]:
    if _uses_weight_decay(param):
        return decay
    return no_decay


def add_weight_decay(
    model: torch.nn.Module, weight_decay: float
) -> List[Dict[str, Any]]:
    """Add weight decay to only the convolutional kernels. Do not add weight decay
    to biases and BN. TensorFlow does this automatically, but for some reason this is
    the PyTorch default.

    Args:
        model: The model where the weight decay is applied.
        weight_decay: How much decay to apply.

    Returns:
        A list of dictionaries encapsulating which params need weight decay.
    """
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if param.requires_grad:
            _weight_decay_group(param, decay, no_decay).append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


# TODO(alex): Unwrap DDP models.
def unwrap_model(model: Any) -> torch.nn.Module:
    """Return the underlying model from a potential wrapper.

    Args:
        model: Model or wrapper to unwrap.
    """
    ...
