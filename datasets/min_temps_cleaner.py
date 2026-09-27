"""Run from datasets/ to restore daily dates and forward-fill missing inputs."""

import pandas as pd

df = pd.read_csv('time_series/melbourne_tempratures/daily-minimum-temperatures-in-me.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='raise')
if df['Date'].isna().any() or df['Date'].duplicated().any():
    raise ValueError('Source dates must be present and unique.')

# Preserve the existing policy of removing question-mark prefixes.
df['Temperature'] = pd.to_numeric(
    df['Temperature'].astype(str).str.replace('?', '', regex=False),
    errors='coerce',
)
df = df.set_index('Date').sort_index()
source_rows = len(df)
daily_index = pd.date_range(df.index.min(), df.index.max(), freq='D', name='Date')
df = df.reindex(daily_index)

# Record actual observations before filling inputs. Unobserved targets must
# still be excluded from training loss and evaluation.
df['observed'] = df['Temperature'].notna()
df['Temperature'] = df['Temperature'].ffill()

output = 'min_temps_cleaned.csv'
df.to_csv(output, index=True, date_format='%Y-%m-%d', na_rep='')
print(
    f'Saved {len(df)} daily rows to {output}: '
    f'{len(df) - source_rows} inserted dates, '
    f'{int((~df["observed"]).sum())} unobserved temperatures.'
)
