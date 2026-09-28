"""Stationarity analysis, hypothesis tests, and target feature plotting."""

import argparse
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def compute_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute sample autocorrelation function up to max_lag."""
    n = len(series)
    variance = np.var(series)
    if variance == 0:
        return np.ones(max_lag + 1)
    mean = np.mean(series)
    centered = series - mean
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[0] = 1.0
        else:
            acf[lag] = np.sum(centered[:-lag] * centered[lag:]) / (n * variance)
    return acf


def analyze_series(name: str, values: np.ndarray, dates: pd.DatetimeIndex,
                   cycle_name: str, cycle_lag: int, rolling_window: int) -> dict:
    """Run summary statistics, rolling measures, and unit-root/stationarity tests."""
    clean_mask = np.isfinite(values)
    clean_vals = values[clean_mask]

    n_total = len(values)
    mean_val = float(np.mean(clean_vals))
    std_val = float(np.std(clean_vals))

    # Rolling statistics
    s = pd.Series(clean_vals)
    r_mean = s.rolling(rolling_window, min_periods=max(1, rolling_window // 4)).mean()
    r_std = s.rolling(rolling_window, min_periods=max(1, rolling_window // 4)).std()
    rolling_mean_std = float(r_mean.std())
    rolling_std_min = float(r_std.min())
    rolling_std_max = float(r_std.max())

    # ACF
    max_lag = max(cycle_lag, 40)
    acf_vals = compute_acf(clean_vals, max_lag)
    acf_1 = float(acf_vals[1])
    acf_cycle = float(acf_vals[cycle_lag]) if cycle_lag < len(acf_vals) else None

    # ADF test (H0: unit root present / non-stationary)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        adf_res = adfuller(clean_vals, autolag='AIC')
        adf_stat, adf_pvalue, adf_lags = float(adf_res[0]), float(adf_res[1]), int(adf_res[2])

    # KPSS test (H0: level stationary)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kpss_res = kpss(clean_vals, regression='c', nlags='auto')
        kpss_stat, kpss_pvalue, kpss_lags = float(kpss_res[0]), float(kpss_res[1]), int(kpss_res[2])

    return {
        'dataset': name,
        'n': n_total,
        'mean': mean_val,
        'std': std_val,
        'rolling_window': rolling_window,
        'rolling_mean_std': rolling_mean_std,
        'rolling_std_min': rolling_std_min,
        'rolling_std_max': rolling_std_max,
        'cycle_name': cycle_name,
        'cycle_lag': cycle_lag,
        'acf_1': acf_1,
        'acf_cycle': acf_cycle,
        'adf_stat': adf_stat,
        'adf_pvalue': adf_pvalue,
        'adf_lags': adf_lags,
        'kpss_stat': kpss_stat,
        'kpss_pvalue': kpss_pvalue,
        'kpss_lags': kpss_lags,
    }


def load_dataset_records(data_dir: Path) -> dict:
    """Load cleaned datasets and format targets and timestamps."""
    datasets = {}

    # 1. Air Passengers
    df_ap = pd.read_csv(data_dir / 'air_passengers_cleaned.csv')
    df_ap['Date'] = pd.to_datetime(df_ap['Month'])
    datasets['air_passengers'] = {
        'title': 'International Airline Passengers',
        'y_label': 'Passengers (thousands)',
        'target_col': '#Passengers',
        'dates': pd.DatetimeIndex(df_ap['Date']),
        'values': df_ap['#Passengers'].to_numpy(dtype=float),
        'cycle_name': 'Annual (12M)',
        'cycle_lag': 12,
        'rolling_window': 12,
        'segments': None,
    }

    # 2. Melbourne Min Temps
    df_mt = pd.read_csv(data_dir / 'min_temps_cleaned.csv')
    df_mt['Date'] = pd.to_datetime(df_mt['Date'])
    datasets['min_temps'] = {
        'title': 'Melbourne Daily Minimum Temperatures',
        'y_label': 'Temperature (°C)',
        'target_col': 'Temperature',
        'dates': pd.DatetimeIndex(df_mt['Date']),
        'values': df_mt['Temperature'].to_numpy(dtype=float),
        'cycle_name': 'Annual (365D)',
        'cycle_lag': 365,
        'rolling_window': 365,
        'segments': None,
    }

    # 3. Smart Home (SML2010) - Segments 1 & 2
    df_sh1 = pd.read_csv(data_dir / 'smart_home_1_cleaned.csv')
    df_sh2 = pd.read_csv(data_dir / 'smart_home_2_cleaned.csv')
    df_sh1['Date'] = pd.to_datetime(df_sh1['Datetime'])
    df_sh2['Date'] = pd.to_datetime(df_sh2['Datetime'])

    # Segment 1 alone for formal tests (unbroken contiguous time series)
    datasets['smart_home_seg1'] = {
        'title': 'Smart Home Dining Room Temp (Recording 1)',
        'y_label': 'Temperature (°C)',
        'target_col': '2',
        'dates': pd.DatetimeIndex(df_sh1['Date']),
        'values': df_sh1['2'].to_numpy(dtype=float),
        'cycle_name': 'Diurnal (24H = 96 steps)',
        'cycle_lag': 96,
        'rolling_window': 96,
        'segments': None,
    }

    # Combined with gap for visual representation
    combined_dates = pd.DatetimeIndex(pd.concat([df_sh1['Date'], df_sh2['Date']]))
    combined_values = np.concatenate([df_sh1['2'].to_numpy(dtype=float), df_sh2['2'].to_numpy(dtype=float)])
    segments = np.concatenate([np.ones(len(df_sh1), dtype=int), np.ones(len(df_sh2), dtype=int) * 2])
    datasets['smart_home'] = {
        'title': 'Smart Home Dining Room Temp (SML2010)',
        'y_label': 'Temperature (°C)',
        'target_col': '2',
        'dates': combined_dates,
        'values': combined_values,
        'cycle_name': 'Diurnal (24H = 96 steps)',
        'cycle_lag': 96,
        'rolling_window': 96,
        'segments': segments,
    }

    # 4. Sunspots
    df_ss = pd.read_csv(data_dir / 'sunspots_cleaned.csv')
    df_ss['Date'] = pd.to_datetime(df_ss['Date'])
    datasets['sunspots'] = {
        'title': 'SILSO Monthly Mean Sunspot Number',
        'y_label': 'Mean Sunspot Count',
        'target_col': 'sunspot_mean',
        'dates': pd.DatetimeIndex(df_ss['Date']),
        'values': df_ss['sunspot_mean'].to_numpy(dtype=float),
        'cycle_name': 'Solar Cycle (~11Y = 132M)',
        'cycle_lag': 132,
        'rolling_window': 132,
        'segments': None,
    }

    df_ar2 = pd.read_csv(data_dir / 'ar2_cleaned.csv')
    datasets['ar2'] = {
        'title': 'Synthetic AR(2)',
        'y_label': 'Value (arbitrary units)',
        'dates': pd.DatetimeIndex(pd.to_datetime(df_ar2['Date'])),
        'values': df_ar2['value'].to_numpy(dtype=float),
        # A smoothing window for visualisation, not a seasonal period.
        'rolling_window': 100,
        'segments': None,
    }

    return datasets


def plot_single_series(meta: dict, output_path: Path):
    """Plot an individual time series with rolling mean and variance band."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)

    dates = meta['dates']
    values = meta['values']
    window = meta['rolling_window']

    s = pd.Series(values, index=dates)
    if meta['segments'] is not None:
        # Avoid connecting lines across segment gaps
        s_seg1 = pd.Series(values[meta['segments'] == 1], index=dates[meta['segments'] == 1])
        s_seg2 = pd.Series(values[meta['segments'] == 2], index=dates[meta['segments'] == 2])
        r_mean1 = s_seg1.rolling(window, min_periods=window // 4).mean()
        r_std1 = s_seg1.rolling(window, min_periods=window // 4).std()
        r_mean2 = s_seg2.rolling(window, min_periods=window // 4).mean()
        r_std2 = s_seg2.rolling(window, min_periods=window // 4).std()

        ax.plot(s_seg1.index, s_seg1.values, color='#7f8c8d', alpha=0.6, linewidth=0.8, label='Observations (Seg 1)')
        ax.plot(s_seg2.index, s_seg2.values, color='#95a5a6', alpha=0.6, linewidth=0.8, label='Observations (Seg 2)')
        ax.plot(r_mean1.index, r_mean1.values, color='#2980b9', linewidth=1.5, label=f'Rolling Mean ({window} steps)')
        ax.plot(r_mean2.index, r_mean2.values, color='#2980b9', linewidth=1.5)
        ax.fill_between(r_mean1.index, r_mean1 - r_std1, r_mean1 + r_std1, color='#2980b9', alpha=0.2, label='±1σ Band')
        ax.fill_between(r_mean2.index, r_mean2 - r_std2, r_mean2 + r_std2, color='#2980b9', alpha=0.2)
    else:
        r_mean = s.rolling(window, min_periods=window // 4).mean()
        r_std = s.rolling(window, min_periods=window // 4).std()

        ax.plot(dates, values, color='#7f8c8d', alpha=0.6, linewidth=0.8, label='Observations')
        ax.plot(dates, r_mean, color='#2980b9', linewidth=1.5, label=f'Rolling Mean ({window})')
        ax.fill_between(dates, r_mean - r_std, r_mean + r_std, color='#2980b9', alpha=0.2, label='±1σ Band')

    ax.set_title(meta['title'], fontsize=11, fontweight='bold')
    ax.set_ylabel(meta['y_label'], fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.legend(loc='upper left', frameon=True, fontsize=8)
    ax.tick_params(labelsize=9)

    fig.savefig(output_path.with_suffix('.pdf'))
    fig.savefig(output_path.with_suffix('.png'), dpi=300)
    plt.close(fig)


def plot_combined_grid(datasets: dict, output_path: Path):
    """Plot five datasets, with AR(2) spanning the bottom row."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(7.2, 7.1), constrained_layout=True)
    grid = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    axes.append(fig.add_subplot(grid[2, :]))

    keys = ['air_passengers', 'min_temps', 'smart_home', 'sunspots', 'ar2']
    panels = [('a', 'Air Passengers'), ('b', 'Melbourne Min Temperatures'),
              ('c', 'Smart Home (SML2010)'), ('d', 'SILSO Sunspots'),
              ('e', 'Synthetic AR(2)')]

    for ax, key, (tag, name) in zip(axes, keys, panels):
        meta = datasets[key]
        dates = meta['dates']
        values = meta['values']
        if key == 'ar2':
            dates = np.arange(1, len(values) + 1)
        window = meta['rolling_window']

        if meta['segments'] is not None:
            mask1 = meta['segments'] == 1
            mask2 = meta['segments'] == 2
            s1 = pd.Series(values[mask1], index=dates[mask1])
            s2 = pd.Series(values[mask2], index=dates[mask2])
            m1 = s1.rolling(window, min_periods=window // 4).mean()
            std1 = s1.rolling(window, min_periods=window // 4).std()
            m2 = s2.rolling(window, min_periods=window // 4).mean()
            std2 = s2.rolling(window, min_periods=window // 4).std()

            ax.plot(dates[mask1], values[mask1], color='#95a5a6', alpha=0.5, linewidth=0.7, label='Data')
            ax.plot(dates[mask2], values[mask2], color='#95a5a6', alpha=0.5, linewidth=0.7)
            ax.plot(dates[mask1], m1, color='#2980b9', linewidth=1.3, label=f'Mean (window={window})')
            ax.plot(dates[mask2], m2, color='#2980b9', linewidth=1.3)
            ax.fill_between(dates[mask1], m1 - std1, m1 + std1, color='#2980b9', alpha=0.2, label='±1σ')
            ax.fill_between(dates[mask2], m2 - std2, m2 + std2, color='#2980b9', alpha=0.2)
        else:
            s = pd.Series(values, index=dates)
            m = s.rolling(window, min_periods=window // 4).mean()
            std = s.rolling(window, min_periods=window // 4).std()

            ax.plot(dates, values, color='#95a5a6', alpha=0.5, linewidth=0.7, label='Data')
            ax.plot(dates, m, color='#2980b9', linewidth=1.3, label=f'Mean (window={window})')
            ax.fill_between(dates, m - std, m + std, color='#2980b9', alpha=0.2, label='±1σ')

        ax.set_title(f'({tag}) {name}', fontsize=10, fontweight='bold')
        ax.set_ylabel(meta['y_label'], fontsize=8.5)
        ax.tick_params(labelsize=8)
        if key == 'ar2':
            ax.set_xlabel('Sample index (after warm-up)', fontsize=8.5)
        else:
            locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.legend(loc='upper left', frameon=True, fontsize=7.5)

    fig.savefig(output_path.with_suffix('.pdf'))
    fig.savefig(output_path.with_suffix('.png'), dpi=300)
    plt.close(fig)


def generate_latex_table(results: list[dict]) -> str:
    """Format results into an IEEEtran LaTeX table."""
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\footnotesize',
        r'\caption{Empirical Stationarity and Descriptive Diagnostics for the Four Time Series Datasets}',
        r'\label{tab:stationarity}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccp{4.2cm}}',
        r'\toprule',
        r'\textbf{Dataset} & \textbf{Obs ($N$)} & \textbf{Mean $\pm$ Std} & \textbf{ACF(1)} & \textbf{ACF(Cycle)} & \textbf{ADF ($p$)} & \textbf{KPSS ($p$)} & \textbf{Primary Evidence} \\',
        r'\midrule',
    ]

    for r in results:
        name = r['dataset']
        n = f"{r['n']:,}"
        mean_std = f"{r['mean']:.1f} $\\pm$ {r['std']:.1f}"
        acf1 = f"{r['acf_1']:.3f}"
        acf_cycle = f"{r['acf_cycle']:.3f} ({r['cycle_lag']})" if r['acf_cycle'] is not None else 'N/A'
        adf = f"{r['adf_stat']:.2f} ({r['adf_pvalue']:.3f})"
        kpss_str = f"{r['kpss_stat']:.2f} ({r['kpss_pvalue']:.3f})"

        # Classification and justification
        if name == 'air_passengers':
            display = 'Air Passengers'
            ev = f'\\textbf{{Non-stationary}} (trend and growing variance); ADF $p>0.99$, KPSS $p<0.01$.'
        elif name == 'min_temps':
            display = 'Melbourne Min Temps'
            ev = f'\\textbf{{Non-stationary}} (annual cycle $s=365$, $\\text{{ACF}}(365)={r["acf_cycle"]:.2f}$); time-varying mean.'
        elif name.startswith('smart_home'):
            display = 'Smart Home (Dining)'
            ev = f'\\textbf{{Non-stationary}} (diurnal cycle $s=96$, $\\text{{ACF}}(96)={r["acf_cycle"]:.2f}$); level shifts, KPSS $p<0.01$.'
        elif name == 'sunspots':
            display = 'SILSO Sunspots'
            ev = f'\\textbf{{Cyclical / Mixed}} (solar cycle $s\\approx 132$, $\\text{{ACF}}(132)={r["acf_cycle"]:.2f}$); decadal amplitude modulation.'
        else:
            display = name
            ev = ''

        lines.append(f"{display} & {n} & {mean_std} & {acf1} & {acf_cycle} & {adf} & {kpss_str} & {ev} \\\\")

    lines.extend([
        r'\bottomrule',
        r'\end{tabular*}',
        r'\end{table*}',
    ])
    return '\n'.join(lines)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=repo_root / 'datasets')
    parser.add_argument('--output-dir', type=Path, default=repo_root / 'report/figures')
    parser.add_argument('--table-output', type=Path, default=repo_root / 'report/tables/stationarity.tex')
    parser.add_argument('--figure-only', action='store_true', help='Rebuild Figure 1 without rerunning stationarity tests.')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_output.parent.mkdir(parents=True, exist_ok=True)

    datasets = load_dataset_records(args.data_dir)
    if args.figure_only:
        plot_combined_grid(datasets, args.output_dir / 'target_features_all')
        print('Saved five-panel Figure 1 as PDF and PNG.')
        return

    print('Analyzing stationarity for the 4 time series datasets...')
    results = []

    # Datasets to report in the stationarity table
    keys_to_test = [
        ('air_passengers', 'air_passengers'),
        ('min_temps', 'min_temps'),
        ('smart_home_seg1', 'smart_home_seg1'),
        ('sunspots', 'sunspots'),
    ]

    for key, label in keys_to_test:
        d = datasets[key]
        res = analyze_series(label, d['values'], d['dates'], d['cycle_name'], d['cycle_lag'], d['rolling_window'])
        results.append(res)
        print(f"\n--- {d['title']} ---")
        print(f"  N: {res['n']}, Mean: {res['mean']:.3f}, Std: {res['std']:.3f}")
        print(f"  ACF(1): {res['acf_1']:.3f}, ACF({res['cycle_lag']}): {res['acf_cycle']}")
        print(f"  ADF Stat: {res['adf_stat']:.3f} (p-value: {res['adf_pvalue']:.4g}, lags: {res['adf_lags']})")
        print(f"  KPSS Stat: {res['kpss_stat']:.3f} (p-value: {res['kpss_pvalue']:.4g}, lags: {res['kpss_lags']})")

    # Generate individual plots
    for key in ['air_passengers', 'min_temps', 'smart_home', 'sunspots']:
        out_file = args.output_dir / f'{key}_ts'
        plot_single_series(datasets[key], out_file)
        print(f"Saved plot: {out_file}.pdf and .png")

    # Generate combined five-panel figure
    grid_file = args.output_dir / 'target_features_all'
    plot_combined_grid(datasets, grid_file)
    print(f"Saved combined plot: {grid_file}.pdf and .png")

    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    args.table_output.write_text(latex_table + '\n')
    print(f"Saved LaTeX table to: {args.table_output}")


if __name__ == '__main__':
    main()
