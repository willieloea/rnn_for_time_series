"""Chronological, fold-local preparation for sequence-to-vector forecasting.

All split sizes are counts of rows, not calendar durations. Each example predicts
one future timestamp. Evaluation uses observed history with fixed model weights;
it is not a recursive rollout. No data files are modified.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    files: tuple[str | Path, ...]
    time_column: str
    frequency: str
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    observed_column: str | None = None
    # Per-feature masks override the shared row mask, including for scaler fits.
    observed_columns: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowConfig:
    lookback: int
    horizon: int = 1
    scaling: Literal['standard', 'none'] = 'standard'

    def __post_init__(self):
        _positive_integer('lookback', self.lookback)
        _positive_integer('horizon', self.horizon)
        if self.scaling not in ('standard', 'none'):
            raise ValueError('scaling must be standard or none.')


@dataclass(frozen=True)
class ValidationConfig:
    test_size: int
    training_window: int
    validation_window: int
    step: int | None = None
    strategy: Literal['expanding', 'sliding'] = 'expanding'

    def __post_init__(self):
        for name in ('test_size', 'training_window', 'validation_window'):
            _positive_integer(name, getattr(self, name))
        if self.step is not None:
            _positive_integer('step', self.step)
        if self.strategy not in ('expanding', 'sliding'):
            raise ValueError('strategy must be expanding or sliding.')


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')


@dataclass
class TimeSeriesData:
    values: pd.DataFrame
    observed: pd.DataFrame
    segments: np.ndarray
    config: DatasetConfig


def load_dataset(config: DatasetConfig) -> TimeSeriesData:
    """Validate selected numeric columns and detect file/gap boundaries.

    Selected values must already be finite (the cleaners fill missing inputs).
    Masks record which values are genuine observations. Unknown gaps start new
    segments; this function never fills them or joins separate source files.
    """
    if not config.files or not config.input_columns or not config.target_columns:
        raise ValueError('Provide files and at least one input and target column.')
    for columns in (config.input_columns, config.target_columns):
        if len(set(columns)) != len(columns):
            raise ValueError('Input and target lists must not contain duplicates.')
    columns = list(dict.fromkeys((*config.input_columns, *config.target_columns)))
    offset = pd.tseries.frequencies.to_offset(config.frequency)
    if offset.n <= 0:
        raise ValueError('frequency must be positive.')
    frames, masks, segments = [], [], []
    next_segment = 0
    for path in config.files:
        frame = pd.read_csv(path)
        index = pd.DatetimeIndex(pd.to_datetime(frame[config.time_column], utc=True))
        if frame.empty or index.hasnans or not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError(f'{path}: timestamps must be nonempty, ordered and unique.')
        grid = pd.date_range(index[0], index[-1], freq=offset)
        if not index.isin(grid).all():
            raise ValueError(f'{path}: timestamps do not match {config.frequency}.')
        values = frame[columns].apply(pd.to_numeric, errors='raise').astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f'{path}: selected values must be finite; run the cleaner first.')
        observed = pd.DataFrame(True, index=frame.index, columns=columns)
        for column in columns:
            flag = config.observed_columns.get(column, config.observed_column)
            if flag is not None:
                raw = frame[flag].astype(str).str.lower()
                if not raw.isin(['true', 'false', '1', '0']).all():
                    raise ValueError(f'{path}: {flag} must contain boolean flags.')
                observed[column] = raw.isin(['true', '1'])
        values.index = observed.index = index
        boundaries = np.ones(len(index), dtype=bool)
        boundaries[1:] = [index[i] != index[i - 1] + offset for i in range(1, len(index))]
        ids = np.cumsum(boundaries) - 1 + next_segment
        next_segment = int(ids[-1]) + 1
        frames.append(values)
        masks.append(observed)
        segments.append(ids)
    values = pd.concat(frames)
    if not values.index.is_monotonic_increasing or not values.index.is_unique:
        raise ValueError('Files must be in chronological order with no overlapping timestamps.')
    return TimeSeriesData(values, pd.concat(masks), np.concatenate(segments), config)


@dataclass
class Standardizer:
    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, data, columns, start, stop, method):
        values = data.values.iloc[start:stop].loc[:, list(columns)].to_numpy()
        mask = data.observed.iloc[start:stop].loc[:, list(columns)].to_numpy()
        if (mask.sum(axis=0) == 0).any():
            raise ValueError('Each selected feature needs observed training values.')
        mean, scale = np.zeros(len(columns)), np.ones(len(columns))
        if method == 'standard':
            for j in range(len(columns)):
                real = values[mask[:, j], j]
                mean[j] = real.mean()
                scale[j] = real.std(ddof=0) or 1.0
        return cls(tuple(columns), mean, scale)

    def transform(self, values):
        return (np.asarray(values) - self.mean) / self.scale

    def inverse_transform(self, values):
        return np.asarray(values) * self.scale + self.mean


@dataclass
class Examples:
    X: np.ndarray
    y: np.ndarray
    target_mask: np.ndarray
    target_times: pd.DatetimeIndex
    origin_times: pd.DatetimeIndex
    segment_ids: np.ndarray
    # Original-scale latest target observations, for the persistence baseline.
    persistence: np.ndarray


@dataclass(frozen=True)
class TemporalSplit:
    # All bounds are row positions; stops are exclusive.
    train_start: int
    train_stop: int
    evaluation_start: int
    evaluation_stop: int
    name: str


@dataclass
class PreparedFold:
    train: Examples
    evaluation: Examples
    input_scaler: Standardizer
    target_scaler: Standardizer
    split: TemporalSplit


class ForecastPipeline:
    def __init__(self, data: TimeSeriesData, windows: WindowConfig, validation: ValidationConfig):
        self.data, self.windows, self.validation = data, windows, validation
        self.test_start = len(data.values) - validation.test_size
        if self.test_start < validation.training_window + validation.validation_window:
            raise ValueError('Not enough development rows for a complete validation fold.')
        if validation.training_window < windows.lookback + windows.horizon:
            raise ValueError('training_window is too short for lookback and horizon.')

    def cv_splits(self) -> Iterator[TemporalSplit]:
        """Only complete validation blocks are emitted, wholly before the test."""
        config = self.validation
        step = config.step or config.validation_window
        for number, boundary in enumerate(range(
            config.training_window, self.test_start - config.validation_window + 1, step
        ), start=1):
            yield self._split(boundary, boundary + config.validation_window, f'fold_{number}')

    def _split(self, boundary, stop, name):
        # At the first evaluation origin, targets after boundary-horizon are
        # unknown. Remove those last horizon-1 rows from training/scaler fitting.
        train_stop = boundary - self.windows.horizon + 1
        start = 0 if self.validation.strategy == 'expanding' else boundary - self.validation.training_window
        return TemporalSplit(start, train_stop, boundary, stop, name)

    def prepare_cv(self) -> Iterator[PreparedFold]:
        for split in self.cv_splits():
            yield self._prepare(split)

    def prepare_final(self) -> PreparedFold:
        """Call after tuning: refit preprocessing and expose the reserved test.

        Expanding uses all available development history; sliding retains the
        most recent configured training window (before horizon purging).
        """
        return self._prepare(self.final_split())

    def final_split(self) -> TemporalSplit:
        return self._split(self.test_start, len(self.data.values), 'final_test')

    def _prepare(self, split):
        cfg = self.data.config
        inputs = Standardizer.fit(self.data, cfg.input_columns, split.train_start, split.train_stop, self.windows.scaling)
        targets = Standardizer.fit(self.data, cfg.target_columns, split.train_start, split.train_stop, self.windows.scaling)
        train = self._examples(split.train_start, split.train_stop, split.train_start, inputs, targets)
        evaluation = self._examples(split.evaluation_start, split.evaluation_stop, split.train_start, inputs, targets)
        if not len(train.y) or not len(evaluation.y):
            raise ValueError(f'{split.name}: no usable training or evaluation examples; adjust windows/splits.')
        return PreparedFold(train, evaluation, inputs, targets, split)

    def prepare_split(self, split: TemporalSplit) -> PreparedFold:
        """Prepare an explicit chronological split (e.g. inner early stopping)."""
        if not (0 <= split.train_start < split.train_stop <= split.evaluation_start
                < split.evaluation_stop <= len(self.data.values)):
            raise ValueError('Invalid chronological split boundaries.')
        if split.train_stop > split.evaluation_start - self.windows.horizon + 1:
            raise ValueError('Training targets extend beyond the first forecast origin.')
        return self._prepare(split)

    def _examples(self, start, stop, history_start, input_scaler, target_scaler):
        data, window = self.data, self.windows
        # Never transform test rows when preparing development folds.
        frame = data.values.iloc[:stop]
        inputs = input_scaler.transform(frame.loc[:, list(input_scaler.columns)].to_numpy())
        raw_targets = frame.loc[:, list(target_scaler.columns)].to_numpy()
        targets = target_scaler.transform(raw_targets)
        masks = data.observed.iloc[:stop].loc[:, list(target_scaler.columns)].to_numpy()
        eligible = []
        for target in range(start, stop):
            origin = target - window.horizon
            first = origin - window.lookback + 1
            if first < history_start:
                continue
            # Check the entire input-to-target interval, including the horizon.
            if data.segments[first] != data.segments[target] or not masks[target].any():
                continue
            eligible.append(target)
        idx = np.asarray(eligible, dtype=int)
        origins = idx - window.horizon
        X = np.empty((len(idx), window.lookback, len(input_scaler.columns)), dtype=float)
        for row, origin in enumerate(origins):
            X[row] = inputs[origin - window.lookback + 1:origin + 1]
        return Examples(
            X, targets[idx], masks[idx], frame.index[idx], frame.index[origins],
            data.segments[idx], raw_targets[origins],
        )


def regression_metrics(actual, predicted, mask):
    """Original-scale MAE/RMSE per target, ignoring unobserved target values."""
    actual, predicted, mask = np.asarray(actual), np.asarray(predicted), np.asarray(mask, dtype=bool)
    if actual.ndim != 2 or actual.shape != predicted.shape or actual.shape != mask.shape:
        raise ValueError('actual, predicted and mask must have matching [examples, targets] shapes.')
    if not np.isfinite(actual[mask]).all() or not np.isfinite(predicted[mask]).all():
        raise ValueError('Scored values must be finite.')
    count = mask.sum(axis=0)
    if (count == 0).any():
        raise ValueError('Each target needs at least one observed evaluation value.')
    error = np.where(mask, predicted - actual, 0.0)
    return {'mae': np.abs(error).sum(axis=0) / count,
            'rmse': np.sqrt(np.square(error).sum(axis=0) / count), 'count': count}
