import pandas as pd

# Load the dataset
df = pd.read_csv('time_series/air_passenger/airpassenger.csv')

# No cleaning needed

# Save to file
df.to_csv('air_passengers_cleaned.csv', index=False)