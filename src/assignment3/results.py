"""Build report tables and scientific figures from completed final evaluations.

Run from the repository root: python -m assignment3.results
No models are trained and no experiment artifacts are modified.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ('air_passengers', 'min_temps', 'smart_home', 'sunspots', 'ar2')
ARCHITECTURES = ('elman', 'jordan', 'multi')
SEEDS = (11, 22, 33)
COLORS = ('#0072B2', '#D55E00', '#009E73')
CASES = (
    ('air_passengers', '#Passengers', 'Air passengers', 'thousand passengers'),
    ('min_temps', 'Temperature', 'Minimum temperatures', '°C'),
    ('smart_home', '2', 'Smart home: dining room', '°C'),
    ('smart_home', '3', 'Smart home: bedroom', '°C'),
    ('sunspots', 'sunspot_mean', 'Sunspots', 'sunspot index'),
    ('ar2', 'value', 'AR(2)', 'arbitrary units'),
)


def read_results(runs):
    """Require complete, comparable runs and verify scores against predictions."""
    tables, forecasts = [], {}
    for dataset in DATASETS:
        directory = runs / f'final_{dataset}'
        complete = json.loads((directory / 'completed.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        if complete['trials'] != 9 or sorted(manifest['seeds']) != list(SEEDS):
            raise ValueError(f'{dataset}: expected nine trials with seeds {SEEDS}.')
        frame = pd.read_csv(directory / 'metrics.csv', dtype={'target': str})
        targets = [target for ds, target, _, _ in CASES if ds == dataset]
        expected = {(arch, seed, target) for arch in ARCHITECTURES
                    for seed in SEEDS for target in targets}
        actual = set(frame[['architecture', 'seed', 'target']].itertuples(index=False, name=None))
        if actual != expected or len(frame) != len(expected):
            raise ValueError(f'{dataset}: missing or duplicate metrics.')
        reference = None
        for arch in ARCHITECTURES:
            for seed in SEEDS:
                pred = pd.read_csv(directory / dataset / arch / f'seed{seed}' / 'predictions.csv')
                columns = ['origin', 'timestamp', 'segment'] + [
                    f'{target}_{suffix}' for target in targets
                    for suffix in ('actual', 'observed', 'persistence')]
                if reference is None:
                    reference = pred[columns]
                else:
                    pd.testing.assert_frame_equal(reference, pred[columns])
                for target in targets:
                    observed = pred[f'{target}_observed']
                    if observed.dtype != bool:
                        raise ValueError('Expected boolean observed flags.')
                    row = frame.loc[(frame.architecture == arch) & (frame.seed == seed)
                                    & (frame.target == target)].iloc[0]
                    if observed.sum() != row.observed_count:
                        raise ValueError('Observed count differs from saved metrics.')
                    for prediction, prefix in [('predicted', ''), ('persistence', 'persistence_')]:
                        errors = (pred.loc[observed, f'{target}_{prediction}']
                                  - pred.loc[observed, f'{target}_actual']).to_numpy()
                        np.testing.assert_allclose(
                            [np.abs(errors).mean(), np.sqrt(np.mean(errors ** 2))],
                            [row[f'{prefix}mae'], row[f'{prefix}rmse']], rtol=1e-7, atol=1e-9)
                forecasts[dataset, arch, seed] = pred
        tables.append(frame)
    metrics = pd.concat(tables, ignore_index=True)
    summary = metrics.groupby(['dataset', 'target', 'architecture'], sort=False).agg(
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        persistence_mae=('persistence_mae', 'first'),
        persistence_rmse=('persistence_rmse', 'first'),
        observed_count=('observed_count', 'first'),
    )
    return metrics, summary.sort_index(), forecasts


def format_score(mean, std=None, digits=3, latex=False):
    if std is None:
        return f'{mean:.{digits}f}'
    separator = r' $\pm$ ' if latex else ' ± '
    return f'{mean:.{digits}f}{separator}{std:.{digits}f}'


def save_tables(metrics, summary, output):
    output.mkdir(parents=True, exist_ok=True)
    tex = [r'\begin{table*}[t]', r'\centering', r'\small',
           r'\caption{Final test errors in original units. RNN entries are mean $\pm$ sample standard deviation across initialization seeds 11, 22 and 33. Persistence is deterministic. Bold indicates the lowest unrounded mean in each row, including persistence; it does not indicate statistical significance. $N$ counts observed test targets.}',
           r'\label{tab:final-errors}', r'\begin{tabular}{llrrrrr}', r'\toprule',
           r'Dataset / target & Metric & $N$ & Elman & Jordan & Multi & Persistence \\', r'\midrule']
    md = ['# Final held-out results', '',
          'Mean ± sample standard deviation across initialization seeds 11, 22 and 33.',
          'All errors use original units. Bold marks the lowest unrounded mean, not statistical significance.',
          'The variation describes initialization on one fixed test period, not uncertainty across new datasets.', '',
          '| Dataset / target | Metric | N | Elman | Jordan | Multi | Persistence |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for dataset, target, label, unit in CASES:
        sub = summary.loc[dataset, target]
        digits = 4 if dataset in ('smart_home', 'ar2') else 3
        for metric in ('mae', 'rmse'):
            baseline = sub.iloc[0][f'persistence_{metric}']
            means = [sub.loc[a, f'{metric}_mean'] for a in ARCHITECTURES] + [baseline]
            winner = int(np.argmin(means))
            cells_tex, cells_md = [], []
            for i, arch in enumerate((*ARCHITECTURES, 'persistence')):
                std = sub.loc[arch, f'{metric}_std'] if i < 3 else None
                for cells, latex in [(cells_tex, True), (cells_md, False)]:
                    value = format_score(means[i], std, digits, latex)
                    if i == winner:
                        value = r'\textbf{' + value + '}' if latex else '**' + value + '**'
                    cells.append(value)
            n = int(sub.iloc[0].observed_count)
            tex.append(' & '.join([label if metric == 'mae' else '', metric.upper(), str(n)] + cells_tex) + r' \\')
            md.append('| ' + ' | '.join([f'{label} ({unit})', metric.upper(), str(n)] + cells_md) + ' |')
        tex.append(r'\addlinespace')
    tex.extend([r'\bottomrule', r'\end{tabular}',
                r'\par\smallskip\footnotesize Units: passengers in thousands; temperatures in $^\circ$C; sunspots as an index; AR(2) in arbitrary units.',
                r'\end{table*}'])
    (output / 'final_errors.tex').write_text('\n'.join(tex) + '\n')

    config = [r'\begin{table*}[t]', r'\centering', r'\small',
              r'\caption{Settings selected using development data. Each cell gives hidden units / training epochs / trainable parameters. Final fits use three initialization seeds and the same selected settings.}',
              r'\label{tab:final-settings}', r'\begin{tabular}{lrrr}', r'\toprule',
              r'Dataset & Elman & Jordan & Multi \\', r'\midrule']
    md += ['', '## Selected settings', '', 'Cells contain hidden units / training epochs / trainable parameters.', '',
           '| Dataset | Elman | Jordan | Multi |', '| --- | ---: | ---: | ---: |']
    for dataset in DATASETS:
        label = next(label for ds, _, label, _ in CASES if ds == dataset)
        if dataset == 'smart_home':
            label = 'Smart home (both targets)'
        cells = []
        for arch in ARCHITECTURES:
            values = metrics.loc[(metrics.dataset == dataset) & (metrics.architecture == arch),
                                 ['hidden_size', 'selected_epochs', 'parameter_count']].drop_duplicates()
            if len(values) != 1:
                raise ValueError('Final settings differ between seeds or targets.')
            cells.append(' / '.join(str(int(v)) for v in values.iloc[0]))
        config.append(' & '.join([label] + cells) + r' \\')
        md.append('| ' + ' | '.join([label] + cells) + ' |')
    config += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    (output / 'final_settings.tex').write_text('\n'.join(config) + '\n')
    md += ['', '## Figures', '',
           '![Test RMSE relative to persistence](../figures/final_rmse.png)', '',
           '![Fixed-seed forecast examples](../figures/final_forecasts.png)', '',
           'Forecast examples use seed 11 for every architecture and the first 96 test rows (24 for passengers).',
           'They are individual fits, not ensembles; the tables and RMSE plot use each entire test set.',
           'AR(2) timestamps are artificial labels. Imputed targets are excluded from scores and actual-value traces.', '',
           '## Reproduce', '', '`uv run python -m assignment3.results` from the repository root.', '',
           'Sources: `runs/final_<dataset>/metrics.csv`, trial `predictions.csv`, and completion manifests.',
           'The script checks all 45 final trials, recomputes metrics from observed predictions, and checks identical evaluation rows across models and seeds.',
           'No experiments are modified or rerun. No oracle baseline is included.']
    (output / 'final_results.md').write_text('\n'.join(md) + '\n')


def save_figures(summary, forecasts, output):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(3, 2, figsize=(9, 8), layout='constrained')
    for ax, (dataset, target, label, _) in zip(axes.flat, CASES):
        sub = summary.loc[dataset, target]
        baseline = sub.iloc[0].persistence_rmse
        for i, (arch, color) in enumerate(zip(ARCHITECTURES, COLORS)):
            ax.errorbar(i, sub.loc[arch, 'rmse_mean'] / baseline,
                        yerr=sub.loc[arch, 'rmse_std'] / baseline,
                        fmt='o', color=color, capsize=5, markersize=7)
        ax.axhline(1, color='#555555', linestyle='--', linewidth=1)
        ax.set_xticks(range(3), ['Elman', 'Jordan', 'Multi'])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(bottom=0)
        ax.set_title(label)
        ax.set_ylabel('RMSE / persistence RMSE')
        ax.grid(axis='y', alpha=0.2)
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'final_rmse.{suffix}', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8.5), layout='constrained')
    handles = None
    for ax, (dataset, target, label, unit) in zip(axes.flat, CASES):
        example = forecasts[dataset, 'elman', 11].iloc[:96]
        x = np.arange(1, len(example) + 1)
        actual = example[f'{target}_actual'].where(example[f'{target}_observed'])
        ax.plot(x, actual, color='#222222', linewidth=1.6, label='Actual', zorder=5)
        ax.plot(x, example[f'{target}_persistence'], color='#888888', linestyle=':', linewidth=1.2, label='Persistence')
        for arch, color in zip(ARCHITECTURES, COLORS):
            pred = forecasts[dataset, arch, 11].iloc[:96]
            ax.plot(x, pred[f'{target}_predicted'], color=color, linewidth=1.0, alpha=0.85, label=arch.title())
        ax.set_title(label)
        ax.set_ylabel(unit)
        ax.set_xlabel('Test step (artificial time)' if dataset == 'ar2' else 'Test step')
        ax.grid(alpha=0.15)
        handles = ax.get_legend_handles_labels()
    fig.legend(*handles, loc='outside upper center', ncol=5, frameon=False)
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'final_forecasts.{suffix}', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--report', type=Path, default=Path('report'))
    args = parser.parse_args()
    metrics, summary, forecasts = read_results(args.runs)
    save_tables(metrics, summary, args.report / 'tables')
    save_figures(summary, forecasts, args.report / 'figures')
    print('Verified 45 final trials; wrote tables, Markdown summary, and PDF/PNG figures.')


if __name__ == '__main__':
    main()
