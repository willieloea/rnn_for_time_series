import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from assignment3.preprocessing import (
    DatasetConfig, ForecastPipeline, ValidationConfig, WindowConfig,
    load_dataset, regression_metrics,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.frame = pd.DataFrame({'Date': pd.date_range('2000-01-01', periods=40),
                                   'a': np.arange(40.), 'b': np.arange(40.) * 10,
                                   'observed': True})
        self.config = DatasetConfig((self.directory / 'data.csv',), 'Date', 'D',
                                    ('a', 'b'), ('a', 'b'), 'observed')
        self.validation = ValidationConfig(test_size=8, training_window=16, validation_window=8)

    def pipeline(self, frame=None, window=None, validation=None, config=None):
        (self.frame if frame is None else frame).to_csv(self.config.files[0], index=False)
        return ForecastPipeline(load_dataset(config or self.config), window or WindowConfig(3),
                                validation or self.validation)

    def test_alignment_shapes_and_inverse(self):
        fold = next(self.pipeline().prepare_cv())
        self.assertEqual(fold.train.X.shape, (13, 3, 2))
        self.assertEqual(fold.evaluation.y.shape, (8, 2))
        np.testing.assert_allclose(fold.input_scaler.inverse_transform(fold.evaluation.X[0]),
                                   [[13, 130], [14, 140], [15, 150]])
        np.testing.assert_allclose(fold.target_scaler.inverse_transform(fold.evaluation.y[0]), [16, 160])

    def test_fold_and_test_isolation(self):
        original = next(self.pipeline().prepare_cv())
        changed = self.frame.copy()
        changed.loc[16:, ['a', 'b']] = 1000000
        altered = next(self.pipeline(changed).prepare_cv())
        np.testing.assert_array_equal(original.train.X, altered.train.X)
        np.testing.assert_array_equal(original.input_scaler.mean, altered.input_scaler.mean)
        pipeline = self.pipeline()
        self.assertTrue(all(s.evaluation_stop <= 32 for s in pipeline.cv_splits()))
        self.assertEqual(pipeline.prepare_final().evaluation.target_times[0], pd.Timestamp('2000-02-02', tz='UTC'))

    def test_expanding_and_sliding(self):
        expanding = list(self.pipeline().cv_splits())
        sliding = list(self.pipeline(validation=replace(self.validation, strategy='sliding')).cv_splits())
        self.assertEqual([(s.train_start, s.train_stop) for s in expanding], [(0, 16), (0, 24)])
        self.assertEqual([(s.train_start, s.train_stop) for s in sliding], [(0, 16), (8, 24)])
        final = self.pipeline(validation=replace(self.validation, strategy='sliding')).prepare_final()
        self.assertEqual((final.split.train_start, final.split.train_stop), (16, 32))

    def test_horizon_purges_unknown_training_targets(self):
        fold = next(self.pipeline(window=WindowConfig(3, horizon=3)).prepare_cv())
        self.assertEqual(fold.split.train_stop, 14)
        self.assertLessEqual(fold.train.target_times.max(), fold.evaluation.origin_times.min())
        self.assertEqual(fold.evaluation.target_times[0] - fold.evaluation.origin_times[0], pd.Timedelta(days=3))
        np.testing.assert_allclose(fold.input_scaler.mean, [6.5, 65])

    def test_masks_exclude_filled_values_from_scaling_and_targets(self):
        frame = self.frame.copy()
        frame.loc[[5, 18], 'observed'] = False
        frame.loc[5, ['a', 'b']] = 9999
        fold = next(self.pipeline(frame).prepare_cv())
        self.assertNotIn(pd.Timestamp('2000-01-19', tz='UTC'), fold.evaluation.target_times)
        np.testing.assert_allclose(fold.input_scaler.mean, [(120 - 5) / 15, (1200 - 50) / 15])
        self.assertEqual(len(fold.train.y), 12)

    def test_per_target_masks_and_metrics(self):
        frame = self.frame.copy()
        frame['b_observed'] = True
        frame.loc[18, 'b_observed'] = False
        fold = next(self.pipeline(frame, config=replace(self.config, observed_columns={'b': 'b_observed'})).prepare_cv())
        np.testing.assert_array_equal(fold.evaluation.target_mask[2], [True, False])
        actual = fold.target_scaler.inverse_transform(fold.evaluation.y)
        scores = regression_metrics(actual, fold.evaluation.persistence, fold.evaluation.target_mask)
        np.testing.assert_array_equal(scores['count'], [8, 7])
        np.testing.assert_allclose(scores['rmse'], [1, 10])

    def test_file_boundaries_block_windows_even_without_time_gap(self):
        first, second = self.directory / 'first.csv', self.directory / 'second.csv'
        self.frame.iloc[:18].to_csv(first, index=False)
        self.frame.iloc[18:].to_csv(second, index=False)
        pipeline = ForecastPipeline(load_dataset(replace(self.config, files=(first, second))),
                                    WindowConfig(3), self.validation)
        fold = next(pipeline.prepare_cv())
        expected = [16, 17, 21, 22, 23]
        self.assertEqual(fold.evaluation.target_times.day.tolist(), [i + 1 for i in expected])

    def test_internal_gap_blocks_horizon_and_lookback(self):
        frame = self.frame.copy()
        frame.loc[18:, 'Date'] += pd.Timedelta(days=5)
        fold = next(self.pipeline(frame, window=WindowConfig(3, horizon=2)).prepare_cv())
        self.assertEqual(fold.evaluation.target_times.tolist(),
                         pd.to_datetime(['2000-01-17', '2000-01-18', '2000-01-28', '2000-01-29'], utc=True).tolist())

    def test_constant_and_unscaled_features(self):
        frame = self.frame.copy()
        frame['b'] = 7
        fold = next(self.pipeline(frame).prepare_cv())
        self.assertEqual(fold.input_scaler.scale[1], 1)
        self.assertTrue((fold.train.X[:, :, 1] == 0).all())
        fold = next(self.pipeline(window=WindowConfig(3, scaling='none')).prepare_cv())
        np.testing.assert_array_equal(fold.evaluation.y[0], [16, 160])

    def test_inputs_and_targets_can_differ(self):
        config = replace(self.config, input_columns=('b',), target_columns=('a',))
        fold = next(self.pipeline(config=config).prepare_cv())
        self.assertEqual(fold.evaluation.X.shape, (8, 3, 1))
        np.testing.assert_allclose(fold.target_scaler.inverse_transform(fold.evaluation.y[:, 0:1]).ravel(), np.arange(16, 24))
        np.testing.assert_allclose(fold.evaluation.persistence.ravel(), np.arange(15, 23))

    def test_test_values_cannot_change_cv_arrays(self):
        before = list(self.pipeline().prepare_cv())
        frame = self.frame.copy()
        frame.loc[32:, ['a', 'b']] = -90000
        after = list(self.pipeline(frame).prepare_cv())
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left.evaluation.X, right.evaluation.X)
            np.testing.assert_array_equal(left.evaluation.y, right.evaluation.y)

    def test_empty_masked_evaluation_is_rejected(self):
        frame = self.frame.copy()
        frame.loc[16:23, 'observed'] = False
        with self.assertRaisesRegex(ValueError, 'no usable'):
            next(self.pipeline(frame).prepare_cv())

    def test_invalid_dates_values_masks_and_sizes(self):
        for column, value in [('Date', self.frame.Date[0]), ('a', float('inf')), ('observed', 'unknown')]:
            with self.subTest(column=column):
                frame = self.frame.copy()
                frame[column] = frame[column].astype(object)
                frame.loc[1, column] = value
                with self.assertRaises(ValueError):
                    self.pipeline(frame)
        with self.assertRaises(ValueError):
            ValidationConfig(0, 10, 5)
        with self.assertRaises(ValueError):
            self.pipeline(window=WindowConfig(20))


if __name__ == '__main__':
    unittest.main()
