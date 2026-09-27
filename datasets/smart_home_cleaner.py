"""Run from datasets/ to prepare the two SML2010 recordings separately."""

import pandas as pd


for number in (1, 2):
    source = f'time_series/sml2010/NEW-DATA-{number}.T15.txt'
    output = f'smart_home_{number}_cleaned.csv'
    df = pd.read_csv(source, sep=r'\s+', comment='#', header=None)
    if df.empty or df.shape[1] != 24:
        raise ValueError(f'{source}: expected date, time and 22 numeric columns.')

    timestamps = pd.to_datetime(
        df[0] + ' ' + df[1], format='%d/%m/%Y %H:%M', errors='raise', utc=True,
    )
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError(f'{source}: timestamps must be present and unique.')
    if not timestamps.eq(timestamps.dt.floor('15min')).all():
        raise ValueError(f'{source}: timestamps must fall on the 15-minute grid.')

    # Retain the numeric labels 2–23 used in the dataset notes.
    df = df.drop(columns=[0, 1]).apply(pd.to_numeric, errors='raise')
    if df.isna().any().any() or df.isin([float('inf'), float('-inf')]).any().any():
        raise ValueError(f'{source}: source sensor values must be present and finite.')
    df.index = pd.DatetimeIndex(timestamps, name='Datetime')
    df = df.sort_index()
    source_rows = len(df)

    # Each recording has its own calendar: never fill the gap between files.
    grid = pd.date_range(df.index.min(), df.index.max(), freq='15min', name='Datetime')
    df = df.reindex(grid)
    sensor_columns = df.columns
    df['observed'] = df[2].notna()
    df[sensor_columns] = df[sensor_columns].ffill()

    # Filled inputs are usable history, but unobserved targets must be masked
    # from training loss and evaluation. Keep UTC explicit in the output.
    df.to_csv(output, index=True, date_format='%Y-%m-%d %H:%M:%S%z')
    print(
        f'Saved {len(df)} rows to {output}: '
        f'{source_rows} observed, {len(df) - source_rows} forward-filled.'
    )
