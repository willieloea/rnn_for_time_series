"""Small dependency-free SVG plots for inspecting saved experiment results."""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def _panels(path, panels):
    width, height = 800, 290 * len(panels)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img">',
             '<title>Forecasting experiment diagnostics</title>',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" font-size="12" fill="#202020">']
    for panel, (title, xlabel, ylabel, series, labels) in enumerate(panels):
        top = panel * 290 + 50
        all_values = np.concatenate([np.asarray(values, dtype=float) for _, values, _ in series])
        finite = all_values[np.isfinite(all_values)]
        low, high = float(finite.min()), float(finite.max())
        margin = max((high - low) * .08, 1e-6)
        low, high = low - margin, high + margin
        n = max(len(values) for _, values, _ in series)
        def x(i):
            return 90 + 680 * i / max(1, n - 1)
        def y(value):
            return top + 170 * (high - value) / (high - low)
        parts.append(f'<text x="90" y="{top-30}" font-size="15">{escape(title)}</text>')
        parts.append(f'<rect x="90" y="{top}" width="680" height="170" fill="none" stroke="#bbb"/>')
        for i in range(4):
            value = low + (high - low) * i / 3
            parts.append(f'<text x="82" y="{y(value)+4:.2f}" text-anchor="end">{value:.3g}</text>')
        for i in sorted(set([0, (n-1)//2, n-1])):
            label = labels[i] if labels else str(i + 1)
            anchor = 'start' if i == 0 else 'end' if i == n-1 else 'middle'
            parts.append(f'<text x="{x(i):.2f}" y="{top+190}" text-anchor="{anchor}">{escape(str(label))}</text>')
        parts.append(f'<text x="430" y="{top+215}" text-anchor="middle">{escape(xlabel)}</text>')
        parts.append(f'<text transform="translate(20,{top+85}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>')
        for j, (name, values, color) in enumerate(series):
            parts.append(f'<text x="{90+j*220}" y="{top-10}" fill="{color}">{escape(name)}</text>')
            chunks, chunk = [], []
            for i, value in enumerate(values):
                if np.isfinite(value):
                    chunk.append(f'{x(i):.2f},{y(value):.2f}')
                elif chunk:
                    chunks.append(chunk)
                    chunk = []
            if chunk:
                chunks.append(chunk)
            for chunk in chunks:
                parts.append(f'<polyline points="{" ".join(chunk)}" fill="none" stroke="{color}" stroke-width="1.4"/>')
    parts.append('</g></svg>')
    Path(path).write_text('\n'.join(parts))


def save_plots(directory, target_columns):
    history = pd.read_csv(directory / 'history.csv')
    panels = []
    for stage, rows in history.groupby('stage', sort=False):
        series = [('Training (online batches)', rows.train_mse.to_numpy(), '#2369ad')]
        if rows.stopping_mse.notna().any():
            series.append(('Inner stopping block', rows.stopping_mse.to_numpy(), '#c04b24'))
        panels.append((stage.replace('_', ' ').title(), 'Epoch', 'Scaled MSE', series, None))
    _panels(directory / 'learning_curve.svg', panels)
    predictions = pd.read_csv(directory / 'predictions.csv')
    panels = []
    for target in target_columns:
        actual = predictions[f'{target}_actual'].where(predictions[f'{target}_observed']).to_numpy(dtype=float, copy=True)
        series = [('Observed', actual, '#202020'),
                  ('Prediction', predictions[f'{target}_predicted'].to_numpy(dtype=float, copy=True), '#2369ad'),
                  ('Persistence', predictions[f'{target}_persistence'].to_numpy(dtype=float, copy=True), '#c04b24')]
        # Break lines at recording boundaries rather than visually bridge gaps.
        boundaries = predictions.segment.diff().fillna(0).ne(0).to_numpy()
        for _, values, _ in series:
            values[boundaries] = np.nan
        labels = predictions.timestamp.str.slice(0, 16).tolist()
        panels.append((f'Target {target}', 'Target timestamp (equally spaced forecast rows)',
                       'Original units', series, labels))
    _panels(directory / 'forecast.svg', panels)
