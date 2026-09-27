"""Run from datasets/ to prepare the monthly SILSO sunspot series."""

import pandas as pd


columns = [
    'year', 'month', 'year_fraction', 'sunspot_mean',
    'sunspot_sd', 'nr_observations', 'def_prov_mark',
]
df = pd.read_csv('time_series/sunspots/SN_m_tot_V2.0.csv', sep=';', header=None)
if df.empty or df.shape[1] != len(columns):
    raise ValueError('Expected seven columns of monthly sunspot data.')
df.columns = columns
df = df.apply(pd.to_numeric, errors='raise')
if df.isna().any().any() or df.isin([float('inf'), float('-inf')]).any().any():
    raise ValueError('Source values must be finite; missing metadata uses -1.')

for column in ['year', 'month', 'nr_observations', 'def_prov_mark']:
    if df[column].mod(1).ne(0).any():
        raise ValueError(f'{column} must contain integers.')
    df[column] = df[column].astype('int64')

df.insert(0, 'Date', pd.to_datetime(dict(year=df['year'], month=df['month'], day=1)))
if df['Date'].duplicated().any():
    raise ValueError('Monthly dates must be unique.')
df = df.sort_values('Date').reset_index(drop=True)
expected_dates = pd.date_range(df['Date'].min(), df['Date'].max(), freq='MS')
if not pd.DatetimeIndex(df['Date']).equals(expected_dates):
    raise ValueError('The monthly calendar has gaps; inspect the source data.')
if df['sunspot_mean'].lt(0).any():
    raise ValueError('Sunspot targets must be nonnegative; no filling is performed.')
if not df['def_prov_mark'].isin([0, 1]).all():
    raise ValueError('Definitive/provisional flags must be 0 or 1.')

# Missing uncertainty/count metadata does not invalidate the sunspot target.
for column in ['sunspot_sd', 'nr_observations']:
    df[column] = df[column].mask(df[column].eq(-1))
    if df[column].lt(0).any():
        raise ValueError(f'{column} contains an unexpected negative value.')
df['nr_observations'] = df['nr_observations'].astype('Int64')

# Keep genuine zeros, provisional months and the supplied February 1824 value.
# Metadata is retained for reference; sunspot_mean is the univariate target.
output = 'sunspots_cleaned.csv'
df.to_csv(output, index=False, date_format='%Y-%m-%d', na_rep='')
print(
    f'Saved {len(df)} monthly rows to {output}: '
    f'{int(df["def_prov_mark"].eq(0).sum())} provisional values retained.'
)
