"""Losses shared by all forecasting architectures."""

import torch
from torch import Tensor


def masked_mse(prediction: Tensor, target: Tensor, observed: Tensor) -> Tensor:
    """Mean squared error over observed target entries only, with gradients."""
    if prediction.ndim != 2 or prediction.shape != target.shape or prediction.shape != observed.shape:
        raise ValueError('prediction, target and observed must have matching [batch, targets] shapes.')
    if observed.dtype != torch.bool:
        raise ValueError('observed must be a boolean tensor.')
    if not observed.any():
        raise ValueError('A training batch needs at least one observed target.')
    # Select before subtraction: even NaN placeholders at masked targets cannot
    # contaminate the loss or gradients.
    error = prediction[observed] - target[observed]
    return error.square().mean()
