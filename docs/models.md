# Recurrent forecasting models

`assignment3.models` provides `ElmanRNN`, `JordanRNN` and `MultiRNN`. All use
one hidden layer, `tanh`, a linear output layer and a configurable hidden size
(default 16). They accept `[batch, lookback, input_features]` and return
`[batch, target_features]`, matching the preprocessing pipeline.

Their updates, with initial hidden state and output both zero, are:

```text
Elman:  h[t] = tanh(Wx x[t] + Wh h[t-1]              + b)
Jordan: h[t] = tanh(Wx x[t]              + Wo o[t-1] + b)
Multi:  h[t] = tanh(Wx x[t] + Wh h[t-1] + Wo o[t-1] + b)

All:    o[t] = Wy h[t] + c
```

There are no additional context self-connections or teacher forcing. Feedback
uses the model's own activations and stays attached to the computation graph,
so gradients propagate through all steps of an input window. The same output
projection is used for intermediate feedback and the final prediction. The final
output is supervised; intermediate outputs need not be independently accurate
forecasts. Context resets on every `forward` call and is independent for each
batch element.

Input/output projections and output-feedback weights use Xavier uniform
initialisation; hidden-feedback weights use orthogonal initialisation. Biases
start at zero. Set `torch.manual_seed(seed)` before constructing each model.

## Connect a model to a prepared fold

Install dependencies with `uv sync`. The project selects CPU PyTorch explicitly;
other dependencies continue to use the default package index.

```python
from pathlib import Path
import torch
from assignment3.models import MODEL_TYPES
from assignment3.losses import masked_mse
from assignment3.prepare_data import dataset_defaults
from assignment3.preprocessing import ForecastPipeline, load_dataset, regression_metrics

dataset, windows, validation = dataset_defaults('smart_home', Path('datasets'))
fold = next(ForecastPipeline(load_dataset(dataset), windows, validation).prepare_cv())

torch.manual_seed(42)
model = MODEL_TYPES['elman'](
    input_size=len(dataset.input_columns),
    hidden_size=16,
    output_size=len(dataset.target_columns),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# One training update on a small batch, not a complete training procedure.
X = torch.as_tensor(fold.train.X[:32], dtype=torch.float32)
y = torch.as_tensor(fold.train.y[:32], dtype=torch.float32)
mask = torch.as_tensor(fold.train.target_mask[:32], dtype=torch.bool)
model.train()
optimizer.zero_grad()
prediction = model(X)
loss = masked_mse(prediction, y, mask)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

model.eval()
with torch.no_grad():
    predictions = model(torch.as_tensor(fold.evaluation.X, dtype=torch.float32))
actual = fold.target_scaler.inverse_transform(fold.evaluation.y)
predicted = fold.target_scaler.inverse_transform(predictions.numpy())
scores = regression_metrics(actual, predicted, fold.evaluation.target_mask)
print(model.parameter_count, scores)
```

Use `MODEL_TYPES['jordan']` or `MODEL_TYPES['multi']` for the other architectures.
The classes also support standard PyTorch `.to(device)`, `.double()` and
`state_dict()` save/load operations. Inputs must have the same floating dtype
and device as the model. `model(X, return_sequence=True)` exposes intermediate
outputs with shape `[batch, lookback, target_features]` for diagnostics.

## Comparison and remaining training work

For input size I, hidden size H and output size O, the common projections have
`I*H + H + H*O + O` parameters. Elman adds `H*H`, Jordan adds `O*H`, and Multi
adds both. Report parameter counts alongside hidden sizes. With linear outputs,
Jordan feedback has effective hidden-to-hidden matrix `Wo @ Wy`; its rank is
at most O. Thus its memory representation differs materially from Elman's,
and Multi's two feedback terms can be combined algebraically into a hidden-state
recurrence (apart from initial-state treatment). These are comparisons of
feedback parameterisations, not guaranteed differences in predictive power.

The shared trainer still needs epoch/batch iteration, chronological early
stopping, refitting, CV selection, repeated seeds and experiment logging. This
change implements the models and masked loss; it does not run the assignment's
experiments or select a best model. Never use the final test to select epochs.

Run `uv run python -m unittest discover -s tests -v` to check recurrence,
gradient flow, batching/state reset, loss masking, serialisation and a small
synthetic learning problem, as well as the preprocessing checks.
