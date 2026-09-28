# Forecasting data pipeline

The pipeline reads the cleaned CSVs and creates in-memory NumPy arrays. It does
not modify source files or train models. Run the cleaners from `datasets/` first
if the cleaned files are missing. For AR(2), run `uv run python ar2_generator.py`
from `datasets/`; no cleaning is needed for the generated signal.

## Inspect the data

From the repository root, with project dependencies installed:

```bash
python -m assignment3.prepare_data ar2
python -m assignment3.prepare_data air_passengers
python -m assignment3.prepare_data smart_home --strategy sliding
python -m assignment3.prepare_data smart_home --inputs 2 3 21 22 --targets 2 3 --lookback 96 --horizon 4
```

Use `uv run python` instead of `python` if the project environment is not active.
The commands show fold shapes, eligible evaluation dates, and persistence MAE
and RMSE per target in original units. They do not evaluate the final test period
unless `--include-test` is explicitly supplied. `--help` lists all options.

The presets are starting choices, not results of hyperparameter selection:

| Dataset | Inputs | Targets | Lookback | Reserved test rows | Initial/fixed training rows | Validation rows |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Air passengers | `#Passengers` | `#Passengers` | 12 | 24 | 72 | 12 |
| Minimum temperatures | `Temperature` | `Temperature` | 30 | 730 | 1461 | 365 |
| Smart home | `2`, `3`, `21`, `22` | `2`, `3` | 96 | 672 | 1344 | 288 |
| Sunspots | `sunspot_mean` | `sunspot_mean` | 132 | 264 | 2000 | 264 |
| AR(2) | `value` | `value` | 2 | 1000 | 2000 | 500 |

**Every size is a count of observations, not a calendar duration.** Leap years
and gaps mean equal row counts need not span equal calendar durations. Smart-home
recordings remain separate segments within one chronological dataset; their
boundary is not filled. Review the printed dates before adopting the presets.

## Use from model training code

```python
from pathlib import Path
from assignment3.preprocessing import (
    DatasetConfig, WindowConfig, ValidationConfig,
    ForecastPipeline, load_dataset, regression_metrics,
)

dataset = DatasetConfig(
    files=(Path('datasets/smart_home_1_cleaned.csv'),
           Path('datasets/smart_home_2_cleaned.csv')),
    time_column='Datetime',
    frequency='15min',
    input_columns=('2', '3', '21', '22'),
    target_columns=('2', '3'),
    observed_column='observed',
)
windows = WindowConfig(lookback=96, horizon=1, scaling='standard')
validation = ValidationConfig(
    test_size=672,
    training_window=1344,
    validation_window=288,
    step=288,
    strategy='expanding',  # or 'sliding'
)
pipeline = ForecastPipeline(load_dataset(dataset), windows, validation)

for fold in pipeline.prepare_cv():
    X_train = fold.train.X
    y_train = fold.train.y
    mask_train = fold.train.target_mask
    X_val = fold.evaluation.X
    # Create a fresh model for each architecture, fold and random seed.
    # Train with masked loss; produce predictions matching fold.evaluation.y.
    # Convert predictions back with fold.target_scaler.inverse_transform(...).

# Only after selecting settings:
final = pipeline.prepare_final()
# Train fresh models on final.train and evaluate on final.evaluation.
```

For univariate data, use a one-element input and target tuple. Inputs and targets
may differ: predicting a column does not require including it in the inputs.
Feature order follows the configured tuples. Date fields, `observed` and
`def_prov_mark` are metadata and do not automatically become input features.

## What each fold returns

- `train` and `evaluation`: each contains `X` shaped `[N, lookback, inputs]`,
  `y` and boolean `target_mask` shaped `[N, targets]`, UTC `target_times`,
  `origin_times`, `segment_ids`, and original-scale persistence predictions.
- `input_scaler` and `target_scaler`: separate per-column means and standard
  deviations; both support `transform` and `inverse_transform`.
- `split`: exclusive row boundaries and a fold name for logging.

For target row `j`, the input ends at `j - horizon` and contains `lookback`
observations. The target is a vector at a **single future timestamp**, not a
sequence of all intervening future values. Thus a 96-step lookback and horizon
4 on smart-home data uses 24 hours of history to predict temperatures one hour
ahead. Keep horizon consistent when comparing models.

## Splits, scaling and leakage

The last `test_size` rows are reserved before generating CV folds. The first
validation block follows `training_window` rows; each later block advances by
`step` (default: `validation_window`). Expanding windows begin at row zero;
sliding windows advance the training start too. Only complete validation blocks
are emitted. Any leftover development tail is still available for final fitting.
Setting `step` smaller than the validation size gives overlapping evaluation
periods; their scores are not independent replicates.

With horizon greater than one, the last `horizon - 1` rows of a nominal training
window are excluded from training and scaler fitting: those targets would not
yet be known at the first validation forecast origin. The same rule applies to
final fitting. For sliding windows, this reduces the effective fitting period
below `training_window`. The first validation input may use known training
history. All inputs must stay within the current training-history start and a
single segment, including the interval from the last input to its target.

Standard scaling uses the population standard deviation (`ddof=0`) of observed
training rows, before overlapping examples are constructed. Filled values are
excluded from these statistics. Constant columns use scale 1 and become zero
after centering. `scaling='none'` returns identity scalers. No test or validation
values determine scaling parameters.

These examples implement sequential forecasting with actual observations
available through each origin. Earlier validation observations may therefore
appear in later validation input windows. Model weights and scalers stay fixed
within a fold. This is not a recursive forecast of the whole validation block.
Reset recurrent context for each independent input window in the initial model
implementation; batch elements are separate sequences.

For final training, expanding uses all available development history and sliding
uses the most recent nominal training window, subject to the horizon rule.
`prepare_final()` is deliberately separate from `prepare_cv()`.

## Missing values and scoring

Cleaned selected columns must contain finite numbers. Loading rejects malformed
values, invalid flags, duplicate or out-of-order timestamps and off-grid dates.
Missing timestamps start new segments; the loader does not impute them. Each
file also starts a new segment, even if its first timestamp is adjacent to the
previous file. Windows crossing either kind of boundary are excluded.

The existing row-level `observed` flag is applied to all selected features.
For a future dataset with per-feature missingness, pass e.g.
`observed_columns={'2': 'dining_observed', '3': 'bedroom_observed'}`; these masks
override the shared mask for the named features. They also control scaling.
Without any configured flag, selected values are treated as observed. Sunspot
February 1824 and provisional observations therefore remain included as agreed.

Examples with no observed target are dropped. Examples with some observed
targets are retained with a per-target mask. **Apply this mask to loss**, e.g.
`((predictions - y) ** 2)[target_mask].mean()`. Merely retaining the mask is not
enough; imputed targets must not contribute to gradients. Filled input values
remain usable. Every returned example has at least one observed target.

For evaluation:

```python
actual = fold.target_scaler.inverse_transform(fold.evaluation.y)
# Replace the baseline with inverse-transformed model predictions as needed.
scores = regression_metrics(
    actual, fold.evaluation.persistence, fold.evaluation.target_mask,
)
```

Metrics are per-target MAE/RMSE and observed counts. Do not average raw errors
across datasets with different units. Persistence uses the latest available
target value (including a causal fill), even if that target is not among the
model's input features. Compare models on identical target timestamps and masks;
if tuning lookback changes eligibility near a boundary, use their intersection.

Choose epoch counts/early stopping using a chronological subset of each training
period, with preprocessing fitted on that subset's fitting prefix. Do not use
the scored validation block to both choose the stopping epoch and claim an
independent evaluation. Model fitting and early stopping belong to the training
code and are not implemented by this data pipeline.

Run the focused checks with `python -m unittest discover -s tests -v`.
