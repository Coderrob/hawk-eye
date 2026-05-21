""" An expotential moving average implementation for PyTorch. TensorFlow has an
implementation here:
https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage """

import copy
from typing import Union

import torch


class Ema:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA parameters from a base model.

        Args:
            model: Wrapped model whose parameters should update the EMA shadow.
        """

        model = model.module
        # Loop over the state dictionary of the incoming model and update the ema model.
        model_state_dict = model.state_dict()
        for key, shadow_val in self.ema_model.state_dict().items():
            model_val = model_state_dict[key].detach()
            shadow_val.copy_(shadow_val * self.decay + (1.0 - self.decay) * model_val)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.ema_model(x)
