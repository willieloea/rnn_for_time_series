"""Single-hidden-layer recurrent forecasters with explicit feedback paths.

Inputs: [batch, time, input_size]. Outputs: [batch, output_size].
Every forward call starts with zero context for each independent input window.
Output feedback always uses model outputs, never ground-truth targets.
"""

import torch
from torch import Tensor, nn


class _RecurrentForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 *, hidden_feedback: bool, output_feedback: bool):
        super().__init__()
        for name, size in [('input_size', input_size), ('hidden_size', hidden_size),
                           ('output_size', output_size)]:
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f'{name} must be a positive integer.')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Exactly one hidden bias, regardless of the number of feedback paths.
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=False) if hidden_feedback else None
        self.output_to_hidden = nn.Linear(output_size, hidden_size, bias=False) if output_feedback else None
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Seed torch before construction to reproduce initial parameters."""
        nn.init.xavier_uniform_(self.input_to_hidden.weight)
        nn.init.zeros_(self.input_to_hidden.bias)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)
        nn.init.zeros_(self.hidden_to_output.bias)
        if self.hidden_to_hidden is not None:
            nn.init.orthogonal_(self.hidden_to_hidden.weight)
        if self.output_to_hidden is not None:
            nn.init.xavier_uniform_(self.output_to_hidden.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, x: Tensor, *, return_sequence: bool = False) -> Tensor:
        """Predict from each window; optionally return all intermediate outputs.

        Intermediate outputs have shape [batch, time, output_size]. The final
        output is the prediction supervised by the preprocessing pipeline's y.
        Earlier outputs are not assigned separate targets by this interface.
        """
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(f'Expected [batch, time, {self.input_size}] inputs, got {tuple(x.shape)}.')
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError('Batch and time dimensions must be nonempty.')
        if not x.is_floating_point():
            raise ValueError('Inputs must be floating-point tensors.')

        hidden = x.new_zeros((x.shape[0], self.hidden_size))
        output = x.new_zeros((x.shape[0], self.output_size))
        # Input projection can be computed for all time steps together.
        projected = self.input_to_hidden(x)
        sequence = []
        for step in projected.unbind(dim=1):
            activation = step
            if self.hidden_to_hidden is not None:
                activation = activation + self.hidden_to_hidden(hidden)
            if self.output_to_hidden is not None:
                activation = activation + self.output_to_hidden(output)
            hidden = torch.tanh(activation)
            output = self.hidden_to_output(hidden)
            # Do not detach context: autograd must propagate through time.
            if return_sequence:
                sequence.append(output)
        return torch.stack(sequence, dim=1) if return_sequence else output


class ElmanRNN(_RecurrentForecaster):
    """h[t] = tanh(Wx x[t] + Wh h[t-1] + b); o[t] = Wy h[t] + c."""

    def __init__(self, input_size: int, hidden_size: int = 16, output_size: int = 1):
        super().__init__(input_size, hidden_size, output_size,
                         hidden_feedback=True, output_feedback=False)


class JordanRNN(_RecurrentForecaster):
    """h[t] = tanh(Wx x[t] + Wo o[t-1] + b); o[t] = Wy h[t] + c.

    Context is a one-step output copy, with no context self-connections.
    """

    def __init__(self, input_size: int, hidden_size: int = 16, output_size: int = 1):
        super().__init__(input_size, hidden_size, output_size,
                         hidden_feedback=False, output_feedback=True)


class MultiRNN(_RecurrentForecaster):
    """h[t] = tanh(Wx x[t] + Wh h[t-1] + Wo o[t-1] + b).

    Combines Elman hidden-state and Jordan output feedback. Context has no
    additional self-connections; outputs remain o[t] = Wy h[t] + c.
    """

    def __init__(self, input_size: int, hidden_size: int = 16, output_size: int = 1):
        super().__init__(input_size, hidden_size, output_size,
                         hidden_feedback=True, output_feedback=True)


MODEL_TYPES = {'elman': ElmanRNN, 'jordan': JordanRNN, 'multi': MultiRNN}
