# Datasets for recurrent time-series forecasting

These five datasets provide different forecasting challenges for the Elman,
Jordan, and multi-recurrent networks required by [the assignment](../README.md).

Links to the original datasets are stored in this document, and cleaned versions
of the datasets used for this assignment can be created by running the
`*_cleaner.py` scripts inside this `datasets/` directory.

## Air passengers - `air_passengers_cleaned.csv`
Captures: Monthly totals of international airline passengers, measured in
thousands.

Source: [Nick Prock's Kaggle copy](https://www.kaggle.com/datasets/nickprock/airpassenger).

Original: Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (1994). Time Series Analysis: Forecasting and Control (3rd ed.). Holden-Day. (Original work published 1976)

Structure:  
- `Month` is the monthly time index
  - All 144 months are present, ordered and unique, with no missing values.
- `#Passengers` is the forecasting target.
  - Ranges from 104 to 622 thousand passengers.

Stationarity: non-stationary; demand grows over time.

Miscellaneous:  
- This tests whether a network can learn both growth and 12-month seasonality.
- With only 12 years of observations, large networks or long input windows can
  readily overfit.
- Lookbacks of 12 or 24 months are reasonable candidates. Compare against a
  seasonal baseline that predicts the value from the same month of the previous
  year, and report errors after inverting any transformations.

## Mackey-Glass - `mg_ts.csv`
To do.

## Melbourne minimum temperatures - `min_temps_cleaned.csv`
Captures: Daily minimum temperatures in Melbourne, Australia, over 1981–1990.

Source: [Paul Brabban's Kaggle copy](https://www.kaggle.com/datasets/paulbrabban/daily-minimum-temperatures-in-melbourne).

Original: The [IBM dataset catalog](https://dataplatform.cloud.ibm.com/exchange/public/entry/view/de4d953f2a766fbc0469723eba0d93ef?context=wca)
describes this benchmark as degrees Celsius and attributes it to the Australian
Bureau of Meteorology.

Structure:  
- `Date` is the daily time index
  - ordered and unique
  - The source has 3,650 rows; 1984-12-31 and 1988-12-31 are absent.
  - Restore both dates, producing 3,652 consecutive daily rows.
  - Fix by forward-filling missing temperatures from the previous day, and add a
    column `observed` to exclude unobserved targets from training loss and
    evaluation; filled values are not ground truth
- `Temperature` is the forecasting target.
  - Daily minimum temperatures in Melbourne, Australia, 1981-1990
  - The temperatures for three dates are not valid numeric strings:
    - 1982-07-20: `?0.2`
    - 1982-07-21: `?0.8`
    - 1984-07-14: `?0.1`
    - Fix by removing the `?`
- `observed` is `True` when a numeric temperature is available after the above
  cleaning policy but before forward-filling, and `False` for a missing
  temperature. Both inserted dates have forward-filled temperatures and retain
  `observed=False`.

Run `python min_temps_cleaner.py` from `datasets/` to regenerate
`min_temps_cleaned.csv` without modifying the source file.

Stationarity: non-stationary

Miscellaneous:  
- The series combines short-term weather persistence with annual seasonality.
- It tests whether the RNN captures useful history beyond a previous-day
  baseline.
- Preserve the daily calendar when handling gaps, fit any seasonal adjustment
  only on training data, and exclude imputed targets from evaluation.
- Celsius values include zero, so logarithms and percentage errors are poor
  defaults.

## Smart-home sensors - `smart_home_1_cleaned.csv` and `smart_home_2_cleaned.csv`
Captures: Environmental and control-system measurements from a monitored home.

Source:  
The primary source is [UCI SML2010](https://archive.ics.uci.edu/dataset/274/sml2010),
credited to Pablo Romeu-Guallart and Francisco Zamora-Martinez (2014), DOI
[10.24432/C5RS3S](https://doi.org/10.24432/C5RS3S).
UCI describes minute sampling aggregated into 15-minute means, with UTC timestamps.

Structure:  
- `Datetime`
  - the day and time at which the measurement was taken
  - ordered and unique from 2012-03-13 11:45 to 2012-05-02 07:30.
    - 645 missing intervals between observations from the two files.
    - Two isolated missing source records at 2012-05-02 03:00 and 04:45
      are inserted and forward-filled in the cleaned second file.
  - Cleaned timestamps explicitly include the UTC offset (`+0000`).
- 22 numeric columns labelled `2`–`23`.
  - `2`, `3`: Dining-room and bedroom temperature (°C)
  - `4`: Forecast weather temperature (°C)
  - `5`, `6`: Dining-room and bedroom CO2 (ppm)
  - `7`, `8`: Indoor relative humidity (%)
  - `9`, `10`: Indoor lighting (lux)
  - `11`: Fraction of interval with detected rain
  - `12`: Twilight sensor
  - `13`: Wind speed (m/s)
  - `14`-`16`: West, east and south facade light (lux)
  - `17`: Solar irradiance (W/m^2)
  - `18`-`20`: Three ventilation motor indicators
    - constant zero in this copy.
  - `21`, `22`: Outdoor temperature (°C) and relative humidity (%)
  - `23`: Day of week
    - `1.0` = Monday, ..., `7.0` = Sunday
    - Fractional source values are preserved; derive weekday from `Datetime`
      if it is needed as a categorical input.
- `2`, dining-room temperature, will be the target.
- `observed` is recorded before forward-filling: `True` for original records,
  `False` for the two inserted records. Exclude unobserved targets from training
  loss and evaluation, even though their filled values can be used as inputs.

Run `python smart_home_cleaner.py` from `datasets/` (requires pandas).
The script reads the original whitespace-separated files, skips their commented
headers, combines date/time, validates unique timestamps and finite numeric
values, and sorts each recording. It preserves all 22 numeric columns and
forward-fills inserted rows from the preceding record within that file only.
The outputs are `smart_home_1.csv` (2,764 rows, all observed) and
`smart_home_2.csv` (1,375 rows, 1,373 observed). The long gap between recordings
is not filled. Source files remain unchanged.

Stationarity: `2` is non-stationary - changing weather and daily heating/cooling
patterns can change the level.

Miscellaneous:  
- dataset provides short-term thermal persistence, potentially daily dependence
  (96 records per day), and real sensor data issues.
- Additional sensor inputs must be available at the forecast origin; future
  sensor readings would leak information.
- Reset recurrent state at long gaps and use identical eligible target
  timestamps for all architectures.

## Monthly sunspots - `sunspots_cleaned.csv`
Captures: The monthly mean total sunspot number, a solar activity index.

Source: 
This is the Version 2.0 series distributed by SILSO at the Royal Observatory of
Belgium through its [data download page](https://www.sidc.be/SILSO/datafiles).
Monthly values average the daily sunspot index; they are not raw counts of
individual spots accumulated over a month. Definitions and format are documented
on the [SILSO monthly-series page](https://www.sidc.be/SILSO/infosnmtot).

Structure:  
- `Date` First day of the month, used as the monthly time index.
- `year` Gregorian calendar year
- `month` Gregorian calendar month
- `year_fraction` Date in fraction of year for the middle of the corresponding
  month
- `sunspot_mean` Monthly mean total sunspot number.
- `sunspot_sd` Monthly mean standard deviation of the input sunspot numbers from
  individual stations.
  - The first 828 source values (through December 1817) are `-1.0`.
    The cleaner writes these as empty cells, without filling them.
- `nr_observations` Number of observations used to compute the monthly mean
  total sunspot number.
  - The first 828 source values are `-1`; the cleaner writes empty cells.
- `def_prov_mark` Definitive/provisional marker.
  - 1 indicates that the value is definitive. 
  - 0 indicates that the monthly value is still provisional and is subject to a
    possible revision (Usually the last 3 to 6 months)
  - The last five provisional values are retained. This is metadata, not an
    input feature for the univariate model.

The target feature is `sunspot_mean`

SILSO documents historical interpolation of February 1824. For this experiment,
its supplied value is retained and treated like every other target

Stationarity: Uncertain - not strictly either.

Miscellaneous:  
- tests longer temporal dependence and variation in cycle timing and amplitude.
- Include several cycles in validation where possible.
- A 132-month lookback represents approximately one typical cycle, but should be
  tuned rather than treated as a fixed physical law.
- `log1p` is a possible variance transformation (log and MAPE fail at zero).

## Notes for comparing the networks

To Do:
- Use chronological expanding-window or rolling-origin validation, keeping the
  final test period untouched during tuning.
- Fit scaling and other learned transformations on each training fold only.
- Construct each input exclusively from observations available before its
  target, with explicit handling of gaps.
- Support stationarity assessments with
  - training-period plots
  - rolling means/variances
  - autocorrelation
  - optionally supplemented by ADF and KPSS tests with their hypotheses and
    settings stated
- Neither a unit-root test nor a visually repeating cycle proves stationarity.
- RNNs do not require automatic differencing of every series; justify
  transformations through the data and validation results.
