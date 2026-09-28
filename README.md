# Time Series Forecasting using Recurrent Neural Networks (Option 4)
Implement and compare various simple recurrent neural networks for time series
prediction.

Assignment requirements:

1. Find 5 time series datasets
   - describe them
   - indicate if they are stationary or non-stationary
2. Know how to implement cross-validation for time series data
   - https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4
   - describe the cross-validation approach used
3. Pre-process the selected datasets
   - describe and justify what you did
4. Implement three recurrent neural networks:
   - Elman RNN, Jordan RNN, multi-RNN
   - Describe each neural network
   - Describe the optimization algorithm and loss function
5. Ensure ideal-fit during model training
   - Describe how underfitting and overfitting was avoided.
6. Describe your empirical process
   - how hyperparameters were set
   - the neural network architectures
   - performance measures
   - process to determine the best neural network for each dataset
7. Discuss results
   - describe which RNN performed best

## 1. Finding datasets
See [this file](./datasets/about.md) to see which datasets I chose and some of
their properties.

Completed cross-validation and final-test experiments cover five datasets: air
passengers, Melbourne minimum temperatures, smart-home sensors, sunspots and a
synthetic stationary AR(2) process.

Generate it from `datasets/` with `uv run python ar2_generator.py`, then inspect
it from the repository root with `uv run python -m assignment3.prepare_data ar2`.
See [generation settings and rationale](datasets/about.md#synthetic-ar2---ar2_cleanedcsv).
A dedicated entry point reuses the shared runner without changing source hashes
required by existing CV runs:

```bash
uv run python -m assignment3.ar2_experiments cv --output runs/cv_ar2 --hidden-sizes 8 16 32 --seeds 42
```

## 2. Cross-validation strategy
The pipeline supports expanding and sliding training windows. The completed
experiments use **expanding-window cross-validation**, followed by a reserved
chronological holdout for final evaluation. Final evaluations are complete for
all five datasets, using initialization seeds 11, 22 and 33. The CV rankings below
remain development results; see the final-test tables for held-out performance.

## 3. Data pre-processing
The shared [preprocessing pipeline](src/assignment3/preprocessing.py) supports
configurable input and target columns, lookbacks and forecast horizons. It
reserves a chronological test period before expanding/sliding cross-validation,
fits scaling on each training fold, and returns sequence arrays with target
masks and timestamps. Smart-home recording boundaries are preserved.

From the repository root:

```bash
uv run python -m assignment3.prepare_data air_passengers
uv run python -m assignment3.prepare_data smart_home --strategy sliding --targets 2 3
uv run python -m unittest discover -s tests -v
```

The inspection commands evaluate a persistence baseline on development folds;
the final test is only evaluated with `--include-test`. See
[pipeline configuration and usage](docs/preprocessing.md) for the Python API,
default split sizes, array shapes, masking rules and final-training procedure.

## 4. Implementing RNNs
The [PyTorch models](src/assignment3/models.py) implement single-hidden-layer
Elman, Jordan and multi-recurrent networks with configurable hidden sizes,
`tanh` activation and linear outputs. All share the pipeline's batch/sequence
interface. The [model guide](docs/models.md) includes equations, a training-update
example and parameter-count considerations. A shared masked MSE loss excludes
unobserved targets. The [shared runner](src/assignment3/experiments.py) performs
early stopping, refitting, cross-validation selection and separate final tests.

## 5. Ensuring ideal-fit
Hidden sizes 8, 16 and 32 are compared using validation performance. Within each
training fold, its final 15% of rows form an inner chronological early-stopping
block. Preprocessing for this inner experiment is fitted only on its fitting
prefix. Training stops after 10 epochs without a sufficient improvement
(`min_delta=1e-5`), up to 100 epochs. A fresh model and scalers are then fitted
on the full outer training period for the selected number of epochs, before
scoring the outer validation block.

This procedure limits overfitting but does not establish an "ideal fit" by
itself. Inspect the saved learning curves and persistence comparisons for
evidence of underfitting or overfitting. A smaller winning hidden size alone is
not proof of overfitting. Some passenger configurations select epochs near the
100-epoch limit, so the training budget may constrain their performance.

## 6. Empirical process
The completed searches contain 207 trials (171 real-data trials and 36 AR(2) trials): three architectures, three hidden
sizes and seed 42, over four folds per dataset except smart home, which has
seven. Each trial includes the inner early-stopping fit and a fresh outer refit.
All models use Adam with learning rate 0.001, batch size 32 and gradient-norm
clipping at 1.0. Context resets for every independent input sequence.

| Dataset | Lookback | Targets | Final holdout rows |
| --- | ---: | --- | ---: |
| Air passengers | 12 months | Passenger count | 24 |
| Minimum temperatures | 30 days | Minimum temperature | 730 |
| Smart home | 96 quarter-hours | Dining-room and bedroom temperatures (`2`, `3`) | 672 |
| Sunspots | 132 months | Mean sunspot number | 264 |

All forecasts are one step ahead. Smart home uses inputs `2`, `3`, `21`, `22`;
the other datasets use their target variable's history. Standard scaling is
fitted independently per training period. Original observations become
available as history for subsequent forecasts; these are not recursive
forecasts of an entire evaluation period.

Candidate selection minimises mean per-target RMSE in training-scaled units,
averaged equally across folds and seeds. This is normalised by each fold's
training standard deviation; it is not RMSE in physical units. Each architecture
gets its own selected hidden size. The final epoch count is the integer median
of selected inner-stopping epochs for that candidate, not a proven optimum.

The saved runs are in `runs/cv_air_passengers/`, `runs/cv_min_temps/`,
`runs/cv_smart_home/` and `runs/cv_sunspots/`:

| File | Contents |
| --- | --- |
| `manifest.json` | Settings, software versions, source-code hashes and data hashes |
| `candidates.csv` | Mean and standard deviation of selection scores per candidate |
| `selected.json` | Selected hidden size and final epoch count per architecture |
| `metrics.csv` | Per-fold, per-target MAE/RMSE in original units, baseline errors and runtime |
| `<dataset>/<architecture>/h<size>/seed<seed>/<fold>/` | Individual trial artifacts |

Each trial directory contains `result.json`, `history.csv`, `predictions.csv`,
`checkpoint.pt`, `learning_curve.svg` and `forecast.svg`. Training curves show
online batch training loss and inner stopping loss; outer validation is scored
after refitting. The candidate score standard deviations in the existing runs
describe variation across chronological folds, **not across random seeds**.

To reproduce a search in a new directory:

```bash
uv run python -m assignment3.experiments cv --datasets air_passengers --output runs/reproduce_air_passengers --hidden-sizes 8 16 32 --seeds 42
```

After freezing the settings, run final holdout evaluations using the saved
selections and three seeds:

```bash
for dataset in air_passengers min_temps smart_home sunspots; do
  uv run python -m assignment3.experiments final --from-cv "runs/cv_${dataset}" --output "runs/final_${dataset}" --seeds 11 22 33 || break
done
```

Final runs produce `metrics.csv` and `summary.csv` with per-target mean errors
and standard deviations across seeds. Do not tune settings after inspecting
these holdouts. Add `--resume` to continue an interrupted run with matching
configuration, data and code. Preserve the existing run directories.

## 7. Discuss results

The following are **cross-validation selections**, not final-test conclusions.
Lower scores are better within each dataset; do not average these scores into
an overall ranking across datasets.

| Dataset | Elman: hidden size / score | Jordan: hidden size / score | Multi: hidden size / score |
| --- | --- | --- | --- |
| Air passengers | **16 / 0.5998** | 8 / 0.7561 | 32 / 0.6766 |
| Minimum temperatures | **8 / 0.5731** | 32 / 0.5994 | 8 / 0.5747 |
| Smart home | 32 / 0.0259 | 32 / 0.0252 | **32 / 0.0208** |
| Sunspots | **16 / 0.3951** | 16 / 0.4107 | 32 / 0.4039 |

For the selected candidates, mean per-fold RMSE in original units gives the
following comparisons with persistence:

- Air passengers: Elman 35.11, Multi 40.63 and Jordan 47.99, versus persistence
  38.26 (thousands of passengers). Only Elman improves on this baseline on this
  aggregate measure.
- Minimum temperatures: Elman 2.4065, Multi 2.4133 and Jordan 2.5166, versus
  persistence 2.7193 degrees Celsius. Elman and Multi are very close; these
  single-seed results do not establish a reliable advantage between them.
- Smart home: Multi gives 0.0607 and 0.0612 degrees Celsius RMSE for dining-room
  and bedroom temperatures, versus persistence 0.1337 and 0.1329. All three
  selected architectures improve on persistence for both targets.
- Sunspots: Elman 25.90, Multi 26.47 and Jordan 26.92, versus persistence 28.32
  in sunspot-index units. The architecture differences are modest.

These observations do not establish why an architecture performs better.
Claims about learning seasonality, long-term memory or overfitting require
additional evidence, such as learning curves and controlled comparisons.
In particular, one-step sunspot accuracy with a 132-month input does not prove
that the network learned an entire solar cycle. Final-test evaluation and
repeated seeds remain the next steps before writing definitive conclusions.

## Final test tables and figures

All 45 final fits are complete (five datasets, three architectures, three seeds).
See [the final results tables and plots](report/tables/final_results.md) for MAE,
RMSE and sample standard deviations across seeds, separate smart-home targets,
and the selected model sizes, epoch counts and parameter counts.

Regenerate from saved results without training models:

```bash
uv run python -m assignment3.results
```

The script checks completion and recomputes saved metrics from predictions before
writing `report/tables/final_errors.tex`, `final_settings.tex`, and PDF/PNG figures
`report/figures/final_rmse.*` and `final_forecasts.*`. These are included in the
report's results section. Compile from `report/` with `latexmk -pdf report.tex`.
The RMSE figure compares mean errors to persistence; error bars show variation
across initialization seeds, not confidence intervals. Forecast examples use
seed 11 and the first 96 test rows (24 for passengers), rather than selecting a
favourable seed or interval. Metrics use the entire eligible test period.

Persistence outperforms every RNN on the passenger holdout. All RNNs improve on
persistence in mean RMSE for the remaining targets. Near-ties, including AR(2),
should not be presented as statistically established architecture advantages.
The optional known-coefficient AR(2) baseline has not been added.
