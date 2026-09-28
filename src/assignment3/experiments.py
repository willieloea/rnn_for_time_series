"""Run reproducible pilots, chronological CV searches and separate final tests."""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from assignment3.models import MODEL_TYPES
from assignment3.experiment_plots import save_plots
from assignment3.prepare_data import dataset_defaults
from assignment3.preprocessing import (
    DatasetConfig, ForecastPipeline, ValidationConfig, WindowConfig,
    load_dataset, regression_metrics,
)
from assignment3.training import TrainingConfig, fit, make_inner_split, new_model, predict


DATASETS = ('air_passengers', 'min_temps', 'smart_home', 'sunspots')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__,
            'platform': platform.platform(),
            'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                            ['experiments.py', 'experiment_plots.py', 'training.py', 'models.py', 'losses.py', 'preprocessing.py']}}


def open_run(output, manifest, resume):
    output = Path(output)
    manifest = json.loads(json.dumps(manifest, default=str))
    if (output / 'manifest.json').exists():
        if not resume:
            raise ValueError(f'{output} already exists; use --resume or choose a new output directory.')
        stored = read_json(output / 'manifest.json')
        stored.pop('created_utc', None)
        if stored != manifest:
            raise ValueError('Resume configuration, data, code or environment differs from the saved manifest.')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Output directory must be empty or contain a matching run manifest.')
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'manifest.json', dict(manifest, created_utc=datetime.now(timezone.utc).isoformat()))
    return output


def build_pipeline(spec):
    data = dict(spec['dataset'])
    for key in ('files', 'input_columns', 'target_columns'):
        data[key] = tuple(data[key])
    for path, expected in spec['data_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Input file changed since run configuration was saved: {path}')
    return ForecastPipeline(load_dataset(DatasetConfig(**data)),
                            WindowConfig(**spec['windows']), ValidationConfig(**spec['validation']))


def scaler_state(scaler):
    return {'columns': list(scaler.columns), 'mean': scaler.mean.tolist(), 'scale': scaler.scale.tolist()}


def evaluate(model, fold):
    samples = fold.evaluation
    scaled = predict(model, samples)
    actual = fold.target_scaler.inverse_transform(samples.y)
    prediction = fold.target_scaler.inverse_transform(scaled)
    learned = regression_metrics(actual, prediction, samples.target_mask)
    baseline = regression_metrics(actual, samples.persistence, samples.target_mask)
    # Macro-average target RMSE in the fold's scaled units. With standard
    # scaling, each target is normalised by its training standard deviation.
    selection = regression_metrics(samples.y, scaled, samples.target_mask)['rmse'].mean()
    rows = []
    for j, target in enumerate(fold.target_scaler.columns):
        rows.append({'target': target, 'mae': float(learned['mae'][j]),
                     'rmse': float(learned['rmse'][j]),
                     'persistence_mae': float(baseline['mae'][j]),
                     'persistence_rmse': float(baseline['rmse'][j]),
                     'observed_count': int(learned['count'][j])})
    predictions = pd.DataFrame({'origin': samples.origin_times.astype(str),
                                'timestamp': samples.target_times.astype(str),
                                'segment': samples.segment_ids})
    for j, target in enumerate(fold.target_scaler.columns):
        predictions[f'{target}_actual'] = actual[:, j]
        predictions[f'{target}_predicted'] = prediction[:, j]
        predictions[f'{target}_observed'] = samples.target_mask[:, j]
        predictions[f'{target}_persistence'] = samples.persistence[:, j]
    return float(selection), rows, predictions


def run_trial(pipeline, split, architecture, hidden, seed, config, directory, *, fixed_epochs=None):
    directory.mkdir(parents=True, exist_ok=True)
    result_file = directory / 'result.json'
    if result_file.exists():
        required = ('checkpoint.pt', 'predictions.csv', 'history.csv', 'learning_curve.svg', 'forecast.svg')
        if all((directory / name).exists() for name in required):
            print(f'Resume: {directory}', flush=True)
            return read_json(result_file)
        raise ValueError(f'Completed trial has missing artifacts: {directory}')
    started = time.perf_counter()
    fold = pipeline.prepare_split(split)
    inputs, outputs = len(pipeline.data.config.input_columns), len(pipeline.data.config.target_columns)
    inner = None
    with (directory / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['stage', 'epoch', 'train_mse', 'stopping_mse', 'seconds'])
        writer.writeheader()

        def logger(stage):
            def log(row):
                writer.writerow(dict(stage=stage, **row))
                stream.flush()
                if row['epoch'] == 1 or row['epoch'] % 10 == 0:
                    stopping = '' if row['stopping_mse'] is None else f", stopping={row['stopping_mse']:.5g}"
                    print(f'  {stage} epoch {row["epoch"]}: train={row["train_mse"]:.5g}{stopping}', flush=True)
            return log

        if fixed_epochs is None:
            inner = make_inner_split(pipeline, split, config.early_stopping_fraction)
            inner_fold = pipeline.prepare_split(inner)
            selector = new_model(architecture, hidden, inputs, outputs, seed, config)
            epochs, _ = fit(selector, inner_fold.train, config, seed,
                            stopping=inner_fold.evaluation, log=logger('early_stopping'))
            del selector
        else:
            epochs = fixed_epochs
        # Fresh weights, optimiser and scalers; no inner-model state is reused.
        model = new_model(architecture, hidden, inputs, outputs, seed, config)
        fit(model, fold.train, config, seed, epochs=epochs, log=logger('refit'))
    score, metrics, predictions = evaluate(model, fold)
    predictions.to_csv(directory / 'predictions.csv', index=False)
    save_plots(directory, fold.target_scaler.columns)
    result = {'architecture': architecture, 'hidden_size': hidden, 'seed': seed,
              'fold': split.name, 'selected_epochs': epochs, 'parameter_count': model.parameter_count,
              'selection_score': score, 'metrics': metrics, 'seconds': time.perf_counter() - started,
              'train_examples': len(fold.train.y), 'evaluation_examples': len(fold.evaluation.y),
              'split': asdict(split), 'inner_split': asdict(inner) if inner else None,
              'first_target': str(fold.evaluation.target_times[0]),
              'last_target': str(fold.evaluation.target_times[-1])}
    torch.save({'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                'architecture': architecture, 'input_size': inputs, 'hidden_size': hidden, 'output_size': outputs,
                'seed': seed, 'epochs': epochs, 'windows': asdict(pipeline.windows),
                'dataset': json.loads(json.dumps(asdict(pipeline.data.config), default=str)),
                'training': asdict(config), 'input_scaler': scaler_state(fold.input_scaler),
                'target_scaler': scaler_state(fold.target_scaler)}, directory / 'checkpoint.pt')
    # This file is the completion marker: interrupted trials are rerun on resume.
    write_json(result_file, result)
    print(f'  Done: epochs={epochs}, score={score:.5g}, seconds={result["seconds"]:.1f}', flush=True)
    return result


def save_metrics(output, results):
    rows = []
    for result in results:
        common = {key: result[key] for key in ('dataset', 'architecture', 'hidden_size', 'seed', 'fold',
                                               'selected_epochs', 'parameter_count', 'seconds', 'selection_score')}
        rows.extend(dict(common, **metric) for metric in result['metrics'])
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)


def select_candidates(results):
    grouped = {}
    for row in results:
        grouped.setdefault((row['dataset'], row['architecture'], row['hidden_size']), []).append(row)
    candidates = []
    for (dataset, architecture, hidden), trials in grouped.items():
        scores = [trial['selection_score'] for trial in trials]
        candidates.append({'dataset': dataset, 'architecture': architecture, 'hidden_size': hidden,
                           'mean_score': float(np.mean(scores)),
                           'std_score': float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.,
                           'trials': len(trials),
                           'final_epochs': max(1, int(np.median([r['selected_epochs'] for r in trials]))),
                           'mean_seconds': float(np.mean([r['seconds'] for r in trials]))})
    selected = {}
    for row in sorted(candidates, key=lambda x: (x['mean_score'], x['hidden_size'])):
        selected.setdefault(row['dataset'], {}).setdefault(row['architecture'], row)
    return candidates, selected


def run_search(manifest, output, resume=False):
    output = open_run(output, manifest, resume)
    config = TrainingConfig(**manifest['training'])
    all_results = []
    for dataset, spec in manifest['datasets'].items():
        pipeline = build_pipeline(spec)
        splits = list(pipeline.cv_splits())
        if manifest['max_folds'] is not None:
            splits = splits[:manifest['max_folds']]
        for architecture in manifest['architectures']:
            for hidden in manifest['hidden_sizes']:
                for seed in manifest['seeds']:
                    for split in splits:
                        directory = output / dataset / architecture / f'h{hidden}' / f'seed{seed}' / split.name
                        print(f'{dataset}/{architecture}/h{hidden}/seed{seed}/{split.name}', flush=True)
                        result = run_trial(pipeline, split, architecture, hidden, seed, config, directory)
                        all_results.append(dict(result, dataset=dataset))
                        save_metrics(output, all_results)
    candidates, selected = select_candidates(all_results)
    pd.DataFrame(candidates).to_csv(output / 'candidates.csv', index=False)
    write_json(output / 'selected.json', {'mode': manifest['mode'], 'models': selected})
    print(f'Completed {len(all_results)} trials. Results: {output}', flush=True)
    return all_results


def run_final(source, output, seeds, resume=False):
    source = Path(source).resolve()
    original = read_json(source / 'manifest.json')
    selection = read_json(source / 'selected.json')
    if original['mode'] != 'cv' or selection['mode'] != 'cv':
        raise ValueError('Final testing requires a completed CV run, not a pilot.')
    manifest = {'mode': 'final', 'source': str(source),
                'source_manifest_sha256': digest(source / 'manifest.json'),
                'selection_sha256': digest(source / 'selected.json'),
                'seeds': seeds, 'datasets': original['datasets'],
                'training': original['training'], 'environment': environment()}
    if original['environment']['code_sha256'] != manifest['environment']['code_sha256']:
        raise ValueError('Implementation changed since CV; rerun CV before final testing.')
    output = open_run(output, manifest, resume)
    config = TrainingConfig(**manifest['training'])
    results = []
    for dataset, models in selection['models'].items():
        pipeline = build_pipeline(original['datasets'][dataset])
        # Construct the final split without evaluating or selecting on test loss.
        split = pipeline.final_split()
        for architecture, chosen in models.items():
            for seed in seeds:
                print(f'Final: {dataset}/{architecture}/seed{seed}', flush=True)
                directory = output / dataset / architecture / f'seed{seed}'
                result = run_trial(pipeline, split, architecture, chosen['hidden_size'], seed, config,
                                   directory, fixed_epochs=chosen['final_epochs'])
                results.append(dict(result, dataset=dataset))
                save_metrics(output, results)
    frame = pd.read_csv(output / 'metrics.csv')
    summary = frame.groupby(['dataset', 'architecture', 'target'], sort=False).agg(
        seeds=('seed', 'count'), mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        persistence_mae=('persistence_mae', 'first'), persistence_rmse=('persistence_rmse', 'first'),
        parameter_count=('parameter_count', 'first'), seconds_mean=('seconds', 'mean'))
    summary.to_csv(output / 'summary.csv')
    write_json(output / 'completed.json', {'trials': len(results)})
    print(f'Final results saved to {output / "summary.csv"}', flush=True)
    return results


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest='mode', required=True)
    for mode in ('pilot', 'cv'):
        command = commands.add_parser(mode)
        command.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
        command.add_argument('--architectures', nargs='+', choices=list(MODEL_TYPES), default=list(MODEL_TYPES))
        command.add_argument('--hidden-sizes', nargs='+', type=int, default=[16] if mode == 'pilot' else [8, 16, 32])
        command.add_argument('--seeds', nargs='+', type=int, default=[42])
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--resume', action='store_true')
        command.add_argument('--data-dir', type=Path, default=Path('datasets'))
        command.add_argument('--strategy', choices=['expanding', 'sliding'], default='expanding')
        command.add_argument('--max-epochs', type=int, default=5 if mode == 'pilot' else 100)
        command.add_argument('--patience', type=int, default=10)
        command.add_argument('--min-delta', type=float, default=1e-5)
        command.add_argument('--learning-rate', type=float, default=0.001)
        command.add_argument('--batch-size', type=int, default=32)
        command.add_argument('--gradient-clip', type=float, default=1.0)
        command.add_argument('--early-stopping-fraction', type=float, default=0.15)
        command.add_argument('--threads', type=int, default=1)
        command.add_argument('--device', default='cpu')
        command.add_argument('--max-folds', type=int, default=1 if mode == 'pilot' else None)
        command.add_argument('--scaling', choices=['standard', 'none'], default='standard')
        command.add_argument('--inputs', nargs='+')
        command.add_argument('--targets', nargs='+')
        for option in ('lookback', 'horizon', 'test-size', 'training-window', 'validation-window', 'step'):
            command.add_argument(f'--{option}', type=int)
    final = commands.add_parser('final')
    final.add_argument('--from-cv', type=Path, required=True)
    final.add_argument('--output', type=Path, required=True)
    final.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    final.add_argument('--resume', action='store_true')
    return root


def main():
    args = parser().parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError('Seeds must be distinct nonnegative integers.')
    if args.mode == 'final':
        run_final(args.from_cv, args.output, args.seeds, args.resume)
        return
    if args.max_folds is not None and args.max_folds < 1:
        raise ValueError('max-folds must be positive.')
    if any(h < 1 for h in args.hidden_sizes) or len(set(args.hidden_sizes)) != len(args.hidden_sizes):
        raise ValueError('Hidden sizes must be distinct positive integers.')
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.architectures)) != len(args.architectures):
        raise ValueError('Dataset and architecture lists must not contain duplicates.')
    if (args.inputs or args.targets) and len(args.datasets) != 1:
        raise ValueError('Column overrides require a single dataset.')
    training = TrainingConfig(**{key: getattr(args, key) for key in TrainingConfig.__dataclass_fields__})
    specifications = {}
    for name in args.datasets:
        dataset, windows, validation = dataset_defaults(name, args.data_dir.resolve())
        dataset = replace(dataset, input_columns=tuple(args.inputs) if args.inputs else dataset.input_columns,
                          target_columns=tuple(args.targets) if args.targets else dataset.target_columns)
        windows = replace(windows, scaling=args.scaling, **{key: getattr(args, key) for key in
                          ('lookback', 'horizon') if getattr(args, key) is not None})
        validation = replace(validation, strategy=args.strategy, **{key: getattr(args, key) for key in
                             ('test_size', 'training_window', 'validation_window', 'step')
                             if getattr(args, key) is not None})
        specifications[name] = {'dataset': asdict(dataset), 'windows': asdict(windows),
                                'validation': asdict(validation),
                                'data_sha256': {str(path): digest(path) for path in dataset.files}}
    manifest = {'mode': args.mode, 'datasets': specifications, 'training': asdict(training),
                'architectures': args.architectures, 'hidden_sizes': args.hidden_sizes,
                'seeds': args.seeds, 'max_folds': args.max_folds, 'environment': environment(),
                'selection_metric': 'mean per-target RMSE in training-scaled units, averaged equally across folds and seeds'}
    run_search(json.loads(json.dumps(manifest, default=str)), args.output, args.resume)


if __name__ == '__main__':
    main()
