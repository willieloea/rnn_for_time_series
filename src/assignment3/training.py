"""Shared batching, masked optimisation and prediction for all three RNNs."""

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from assignment3.losses import masked_mse
from assignment3.models import MODEL_TYPES
from assignment3.preprocessing import Examples, ForecastPipeline, TemporalSplit


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-5
    learning_rate: float = 0.001
    batch_size: int = 32
    gradient_clip: float = 1.0
    early_stopping_fraction: float = 0.15
    threads: int = 1
    device: str = 'cpu'

    def __post_init__(self):
        for key in ('max_epochs', 'patience', 'batch_size', 'threads'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{key} must be a positive integer.')
        for key in ('learning_rate', 'gradient_clip'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive and finite.')
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError('min_delta must be nonnegative and finite.')
        if not 0 < self.early_stopping_fraction < 0.5:
            raise ValueError('early_stopping_fraction must be between 0 and 0.5.')


def new_model(architecture, hidden_size, input_size, output_size, seed, config):
    torch.set_num_threads(config.threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return MODEL_TYPES[architecture](input_size, hidden_size, output_size).to(config.device)


def make_inner_split(pipeline, outer, fraction):
    size = outer.train_stop - outer.train_start
    boundary = outer.train_stop - max(1, math.ceil(size * fraction))
    train_stop = boundary - pipeline.windows.horizon + 1
    if train_stop - outer.train_start < pipeline.windows.lookback + pipeline.windows.horizon:
        raise ValueError('Training fold too short for an inner early-stopping split.')
    return TemporalSplit(outer.train_start, train_stop, boundary, outer.train_stop,
                         f'{outer.name}_early_stopping')


def _tensors(examples):
    return (torch.as_tensor(examples.X, dtype=torch.float32),
            torch.as_tensor(examples.y, dtype=torch.float32),
            torch.as_tensor(examples.target_mask, dtype=torch.bool))


def predict(model, examples: Examples, batch_size=256) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(examples.y), batch_size):
            x = torch.as_tensor(examples.X[start:start + batch_size], dtype=torch.float32, device=device)
            chunks.append(model(x).cpu().numpy())
    return np.concatenate(chunks)


def fit(model, train, config, seed, *, stopping=None, epochs=None, log=None):
    """Train for fixed epochs, or select an epoch on an inner stopping block.

    The early-stopping model is discarded by the runner. The returned epoch is
    used to train a fresh model on the full outer training period. Training MSE
    in the history is the online batch loss, not a second full training pass.
    """
    x, y, mask = _tensors(train)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    device = next(model.parameters()).device
    best, best_epoch, stale = float('inf'), 0, 0
    history = []
    started = time.perf_counter()
    limit = config.max_epochs if epochs is None else epochs
    for epoch in range(1, limit + 1):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        squared_error, count = 0.0, 0
        for indices in order.split(config.batch_size):
            xb, yb, mb = x[indices].to(device), y[indices].to(device), mask[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(xb), yb, mb)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss; inspect data or reduce learning rate.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            observed = int(mb.sum())
            squared_error += float(loss.detach()) * observed
            count += observed
        score = None
        if stopping is not None:
            predicted = predict(model, stopping)
            score = float(np.square(predicted - stopping.y)[stopping.target_mask].mean())
            if not math.isfinite(score):
                raise RuntimeError('Nonfinite early-stopping loss.')
            if score < best - config.min_delta:
                best, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        row = {'epoch': epoch, 'train_mse': squared_error / count,
               'stopping_mse': score, 'seconds': time.perf_counter() - started}
        history.append(row)
        if log is not None:
            log(row)
        if stopping is not None and stale >= config.patience:
            break
    return (best_epoch if stopping is not None else limit), history
