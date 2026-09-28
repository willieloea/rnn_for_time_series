"""Inspect prepared folds and a persistence baseline without training an RNN."""

import argparse
from pathlib import Path

from assignment3.preprocessing import (
    DatasetConfig, ForecastPipeline, ValidationConfig, WindowConfig,
    load_dataset, regression_metrics,
)


def dataset_defaults(name: str, directory: Path):
    """Starting configurations, not hyperparameters selected by experiments."""
    settings = {
        'air_passengers': (('air_passengers_cleaned.csv',), 'Month', 'MS',
                           ('#Passengers',), ('#Passengers',), None, 12, 24, 72, 12),
        'min_temps': (('min_temps_cleaned.csv',), 'Date', 'D',
                      ('Temperature',), ('Temperature',), 'observed', 30, 730, 1461, 365),
        'smart_home': (('smart_home_1_cleaned.csv', 'smart_home_2_cleaned.csv'),
                       'Datetime', '15min', ('2', '3', '21', '22'), ('2', '3'),
                       'observed', 96, 672, 1344, 288),
        'sunspots': (('sunspots_cleaned.csv',), 'Date', 'MS',
                     ('sunspot_mean',), ('sunspot_mean',), None, 132, 264, 2000, 264),
        'ar2': (('ar2_cleaned.csv',), 'Date', 'D',
                ('value',), ('value',), None, 2, 1000, 2000, 500),
    }
    files, time, freq, inputs, targets, observed, lookback, test, train, val = settings[name]
    return (DatasetConfig(tuple(directory / f for f in files), time, freq, inputs, targets, observed),
            WindowConfig(lookback), ValidationConfig(test, train, val))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['air_passengers', 'min_temps', 'smart_home', 'sunspots', 'ar2'])
    parser.add_argument('--data-dir', type=Path, default=Path('datasets'))
    parser.add_argument('--strategy', choices=['expanding', 'sliding'], default='expanding')
    parser.add_argument('--inputs', nargs='+', help='Input column names, in feature order.')
    parser.add_argument('--targets', nargs='+', help='Target column names, in output order.')
    for name in ['lookback', 'horizon', 'test-size', 'training-window', 'validation-window', 'step']:
        parser.add_argument(f'--{name}', type=int)
    parser.add_argument('--scaling', choices=['standard', 'none'], default='standard')
    parser.add_argument('--include-test', action='store_true', help='Also prepare/evaluate the final holdout; use only after tuning.')
    args = parser.parse_args()
    dataset, window, validation = dataset_defaults(args.dataset, args.data_dir)
    from dataclasses import replace
    dataset = replace(dataset, input_columns=tuple(args.inputs) if args.inputs else dataset.input_columns,
                      target_columns=tuple(args.targets) if args.targets else dataset.target_columns)
    window = replace(window, scaling=args.scaling, **{
        name: getattr(args, name) for name in ('lookback', 'horizon') if getattr(args, name) is not None
    })
    validation = replace(validation, strategy=args.strategy, **{
        name: getattr(args, name) for name in ('test_size', 'training_window', 'validation_window', 'step')
        if getattr(args, name) is not None
    })
    pipeline = ForecastPipeline(load_dataset(dataset), window, validation)
    print(f'{args.dataset}: {len(pipeline.data.values)} rows; test begins '
          f'{pipeline.data.values.index[pipeline.test_start]}; {validation.strategy} validation')
    print(f'Inputs: {dataset.input_columns}; targets: {dataset.target_columns}')

    def show(fold):
        samples = fold.evaluation
        actual = fold.target_scaler.inverse_transform(samples.y)
        scores = regression_metrics(actual, samples.persistence, samples.target_mask)
        print(f'{fold.split.name}: train X={fold.train.X.shape}, y={fold.train.y.shape}; '
              f'evaluation X={samples.X.shape}, y={samples.y.shape}; '
              f'{samples.target_times[0]} to {samples.target_times[-1]}')
        print(f'  Persistence MAE={scores["mae"].round(4).tolist()}, '
              f'RMSE={scores["rmse"].round(4).tolist()}, observed targets={scores["count"].tolist()}')

    for fold in pipeline.prepare_cv():
        show(fold)
    if args.include_test:
        show(pipeline.prepare_final())


if __name__ == '__main__':
    main()
