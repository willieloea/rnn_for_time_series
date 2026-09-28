"""Generate the synthetic AR(2) benchmark. Run from datasets/."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


SEED = 2026
N_SAMPLES = 5000
BURN_IN = 1000
PHI_1 = 0.6
PHI_2 = 0.2
NOISE_STD = 1.0


def generate_ar2(n_samples=N_SAMPLES, burn_in=BURN_IN, seed=SEED):
    """Discard initialization transients from a stable, zero-intercept AR(2)."""
    if n_samples < 1 or burn_in < 0:
        raise ValueError('n_samples must be positive and burn_in non-negative.')
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = rng.normal(0.0, NOISE_STD, n_samples + burn_in)
    values = np.zeros(n_samples + burn_in + 2)
    for t, innovation in enumerate(noise, start=2):
        values[t] = PHI_1 * values[t - 1] + PHI_2 * values[t - 2] + innovation
    return values[burn_in + 2:]


def main():
    directory = Path('time_series/ar2')
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / 'ar2.csv'
    # Dates are arbitrary, equally spaced labels, not physical observations.
    frame = pd.DataFrame({
        'Date': pd.date_range('2000-01-01', periods=N_SAMPLES, freq='D'),
        'value': generate_ar2(),
    })
    frame.to_csv(raw, index=False, float_format='%.17g')
    # No missing data or cleaning transformations: retain the exact generated data.
    shutil.copyfile(raw, 'ar2_cleaned.csv')
    metadata = {
        'equation': 'x[t] = 0.6*x[t-1] + 0.2*x[t-2] + epsilon[t]',
        'phi_1': PHI_1, 'phi_2': PHI_2, 'intercept': 0.0,
        'noise_distribution': 'independent Gaussian', 'noise_std': NOISE_STD,
        'seed': SEED, 'rng': 'numpy.random.Generator(PCG64)',
        'initial_values': [0.0, 0.0], 'burn_in': BURN_IN,
        'n_samples': N_SAMPLES, 'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'time_index': 'artificial daily labels beginning 2000-01-01',
        'sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
    }
    (directory / 'generation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Wrote {N_SAMPLES} rows to {raw} and ar2_cleaned.csv')


if __name__ == '__main__':
    main()
