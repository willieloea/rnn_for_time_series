# Final held-out results

Mean ± sample standard deviation across initialization seeds 11, 22 and 33.
All errors use original units. Bold marks the lowest unrounded mean, not statistical significance.
The variation describes initialization on one fixed test period, not uncertainty across new datasets.

| Dataset / target | Metric | N | Elman | Jordan | Multi | Persistence |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Air passengers (thousand passengers) | MAE | 24 | 66.839 ± 10.259 | 60.587 ± 4.737 | 65.360 ± 3.292 | **44.208** |
| Air passengers (thousand passengers) | RMSE | 24 | 80.303 ± 10.780 | 79.682 ± 6.765 | 76.290 ± 2.299 | **51.782** |
| Minimum temperatures (°C) | MAE | 730 | **1.742 ± 0.007** | 1.854 ± 0.016 | 1.747 ± 0.006 | 1.953 |
| Minimum temperatures (°C) | RMSE | 730 | **2.203 ± 0.004** | 2.346 ± 0.016 | 2.208 ± 0.005 | 2.481 |
| Smart home: dining room (°C) | MAE | 670 | 0.0336 ± 0.0146 | 0.0325 ± 0.0068 | **0.0255 ± 0.0023** | 0.1176 |
| Smart home: dining room (°C) | RMSE | 670 | 0.0421 ± 0.0156 | 0.0417 ± 0.0079 | **0.0332 ± 0.0019** | 0.1345 |
| Smart home: bedroom (°C) | MAE | 670 | 0.0305 ± 0.0013 | 0.0349 ± 0.0029 | **0.0287 ± 0.0038** | 0.1138 |
| Smart home: bedroom (°C) | RMSE | 670 | 0.0395 ± 0.0019 | 0.0475 ± 0.0020 | **0.0391 ± 0.0047** | 0.1347 |
| Sunspots (sunspot index) | MAE | 264 | **11.902 ± 0.400** | 12.097 ± 0.219 | 12.778 ± 0.354 | 12.595 |
| Sunspots (sunspot index) | RMSE | 264 | **16.793 ± 0.452** | 16.912 ± 0.167 | 17.515 ± 0.509 | 17.834 |
| AR(2) (arbitrary units) | MAE | 1000 | 0.8163 ± 0.0022 | **0.8140 ± 0.0037** | 0.8156 ± 0.0069 | 0.8699 |
| AR(2) (arbitrary units) | RMSE | 1000 | 1.0193 ± 0.0019 | **1.0185 ± 0.0074** | 1.0195 ± 0.0060 | 1.0868 |

## Selected settings

Cells contain hidden units / training epochs / trainable parameters.

| Dataset | Elman | Jordan | Multi |
| --- | ---: | ---: | ---: |
| Air passengers | 16 / 99 / 305 | 8 / 99 / 33 | 32 / 69 / 1153 |
| Minimum temperatures | 8 / 36 / 89 | 32 / 11 / 129 | 8 / 29 / 97 |
| Smart home (both targets) | 32 / 60 / 1250 | 32 / 58 / 290 | 32 / 54 / 1314 |
| Sunspots | 16 / 16 / 305 | 16 / 48 / 65 | 32 / 13 / 1153 |
| AR(2) | 32 / 14 / 1121 | 16 / 40 / 65 | 32 / 19 / 1153 |

## Figures

![Test RMSE relative to persistence](../figures/final_rmse.png)

![Fixed-seed forecast examples](../figures/final_forecasts.png)

Forecast examples use seed 11 for every architecture and the first 96 test rows (24 for passengers).
They are individual fits, not ensembles; the tables and RMSE plot use each entire test set.
AR(2) timestamps are artificial labels. Imputed targets are excluded from scores and actual-value traces.

## Reproduce

`uv run python -m assignment3.results` from the repository root.

Sources: `runs/final_<dataset>/metrics.csv`, trial `predictions.csv`, and completion manifests.
The script checks all 45 final trials, recomputes metrics from observed predictions, and checks identical evaluation rows across models and seeds.
No experiments are modified or rerun. No oracle baseline is included.
