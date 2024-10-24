import pandas as pd

# Load data from the CSV file
df = pd.read_csv('/home/heathknowles/Documents/AINN Project/trends_data/forecast_Nintendo Switch.csv')

# Convert 'yhat' values to integers, effectively removing the decimal
df['yhat'] = df['yhat'].astype(int)

# Save the updated DataFrame back to a CSV file if needed
df.to_csv('updated_file.csv', index=False)

# Show the updated DataFrame
print(df)
